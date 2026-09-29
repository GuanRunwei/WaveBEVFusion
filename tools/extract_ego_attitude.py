"""Per-frame ego roll and pitch for the maritime3d benchmark, from the IMU.

The sea is not a road. A vehicle benchmark can assume the sensor rig stays
level, so a 7-DoF box ``(x, y, z, dx, dy, dz, yaw)`` in the sensor frame is also
a box in the world frame. Here the platform rocks: over sequence 00 the hull
holds sigma = 0.70 deg of roll at a 2.25 s period and sigma = 1.35 deg of pitch
at a 19 s swell period, with excursions to 6.5 deg. One degree of pitch
displaces a target 2.6 m vertically at 150 m, and IoU 0.5 on a 6 m boat allows
about 2 m of centre error in total -- so attitude is not a second-order
correction out there, it is the dominant geometric term.

``dataset/00/lidar_front/imu.txt`` is 213781 rows at 96.4 Hz:

    timestamp  gyro_x gyro_y gyro_z  accel_x accel_y accel_z

The accelerometer is uncalibrated in scale (|a| averages 11.51, not 9.81) but
attitude comes from the *direction* of gravity, which a uniform scale factor
leaves untouched. The two triads are consistent with each other: d(roll)/dt from
the accelerometer correlates +0.79 with gyro_x at slope +0.99, and pitch with
gyro_y at +0.69 / +1.08, while both cross terms sit below 0.19. So the axes are
matched and the signs agree, and the two can be fused.

Fusing matters because neither alone is usable:

  * the accelerometer measures gravity *plus* hull acceleration. Taken raw its
    roll swings 87 deg peak to peak, which is slamming and wave impact, not
    attitude;
  * the gyro integrates cleanly at high rate but carries a -1.79 deg/s bias on
    x, which over a 37-minute record integrates to nearly four thousand degrees.

A complementary filter takes the high-frequency half from the gyro and the
low-frequency half from gravity, which is exactly the split their error
characteristics call for. Bias is removed first as the record mean -- the vessel
has no net rotation over 37 minutes, so the mean rate *is* the bias.

Writes ``data/maritime/ego_attitude.npz`` keyed by the frame's capture
timestamp in nanoseconds -- the LiDAR filename stem -- and consumed by
``MaritimeMetric`` to break AP down by sea state:

    python tools/extract_ego_attitude.py

The key is the timestamp rather than ``sample_idx`` because mmengine's
``BaseDataset.get_data_info`` overwrites ``sample_idx`` with the frame's
position in the split, discarding the value stored in the info pkl. Keying on
it would silently look up the wrong frames -- and still find a hit for every
one, because the positions 0..3883 are themselves valid indices elsewhere in
the record. The filename stem cannot collide that way.

Small-angle note: gyro_x and gyro_y are body rates, and mapping them to roll and
pitch rates neglects a ``tan(pitch)`` coupling term. At the tilts measured here
that correction is under 0.5% -- far below the accelerometer's own noise -- so
it is not applied.

Why sea state is banded on rate, not on tilt
--------------------------------------------

The obvious sea-state variable is how far the hull is tilted. It is the wrong
one, and measurably so. A vessel in a sustained turn is pushed outward by
centripetal acceleration, and an accelerometer cannot tell that from a roll --
so a two-minute turn reads as two minutes of heel. On this record the tilt
estimate correlates +0.30 with sustained yaw rate, and frames in the top tilt
tercile are turning at 1.28 deg/s against 0.63 deg/s in the bottom one.

That contaminates the comparison badly, because turning is not independent of
what is in the scene. Banding the test split on tilt puts 11233 boat boxes at a
median 84 m in the calm third and **194 boat boxes at a median 14 m** in the
rough third. Any AP difference between those two sets is a statement about which
vessels were nearby, not about wave motion.

The RMS of the *rate* over a few seconds has neither problem. A steady turn is
near-DC in roll and pitch rate, and a static trim offset is exactly DC, so both
drop out; what survives is the oscillation that waves actually cause. Banding on
it gives three thirds that are matched in the things that would otherwise
explain an AP gap -- median boat range 81.6 / 81.5 / 77.3 m, 33 / 26 / 35 per
cent of boats beyond 100 m, spread over 23 / 26 / 25 temporal blocks -- so a
difference between them is attributable to the motion.
"""
import argparse
import os
import pickle

import numpy as np

INFOS = ('maritime_infos_train.pkl', 'maritime_infos_val.pkl',
         'maritime_infos_test.pkl')


def load_imu(path):
    """``(t, gyro, accel)`` with the gyro bias removed."""
    d = np.loadtxt(path)
    t, gyro, accel = d[:, 0], d[:, 1:4], d[:, 4:7]
    # the boat does not rotate on net over 37 minutes, so the mean rate is bias
    return t, gyro - gyro.mean(axis=0), accel


def complementary(t, gyro, accel, tau):
    """Fuse gyro rates with the gravity direction into roll and pitch, degrees.

    ``tau`` is the crossover time constant: above ``1/tau`` the estimate follows
    the gyro, below it the accelerometer. 2 s sits between the 2.25 s roll
    oscillation and the drift timescale of the bias-corrected gyro.
    """
    roll_a = np.arctan2(accel[:, 1], accel[:, 2])
    pitch_a = np.arctan2(-accel[:, 0], np.hypot(accel[:, 1], accel[:, 2]))
    roll = np.empty_like(roll_a)
    pitch = np.empty_like(pitch_a)
    roll[0], pitch[0] = roll_a[0], pitch_a[0]
    for k in range(1, len(t)):
        dt = t[k] - t[k - 1]
        if not 0 < dt < 1.0:  # clock glitch: restart from gravity alone
            roll[k], pitch[k] = roll_a[k], pitch_a[k]
            continue
        a = tau / (tau + dt)
        roll[k] = a * (roll[k - 1] + gyro[k, 0] * dt) + (1 - a) * roll_a[k]
        pitch[k] = a * (pitch[k - 1] + gyro[k, 1] * dt) + (1 - a) * pitch_a[k]
    return np.degrees(roll), np.degrees(pitch)


def rocking(t, gyro, half_window):
    """RMS roll+pitch rate over a sliding window, in deg/s: wave motion.

    Insensitive to both a steady turn and a static trim, which are near-DC in
    rate and are exactly what corrupts a tilt-based sea state. ``half_window``
    of 2 s spans about two cycles of the 2.25 s roll oscillation.
    """
    rate2 = np.degrees(gyro[:, 0])**2 + np.degrees(gyro[:, 1])**2
    n = max(1, int(half_window * len(t) / (t[-1] - t[0])))
    kernel = np.ones(2 * n + 1) / (2 * n + 1)
    return np.sqrt(np.convolve(rate2, kernel, 'same'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--imu', default='dataset/00/lidar_front/imu.txt')
    p.add_argument('--data-root', default='data/maritime')
    p.add_argument('--tau', type=float, default=2.0)
    p.add_argument('--rock-window', type=float, default=2.0,
                   help='half-width, in seconds, of the rocking RMS window')
    p.add_argument('--max-dt', type=float, default=0.05,
                   help='reject a frame whose nearest IMU sample is further off')
    p.add_argument('--out', default=None)
    args = p.parse_args()

    t, gyro, accel = load_imu(args.imu)
    roll, pitch = complementary(t, gyro, accel, args.tau)
    rock = rocking(t, gyro, args.rock_window)
    # signed rates, for distinguishing "heeled over" from "moving fast"
    d_roll = np.gradient(roll) / np.gradient(t)
    d_pitch = np.gradient(pitch) / np.gradient(t)
    print(f'{len(t)} IMU samples at {len(t) / (t[-1] - t[0]):.1f} Hz, '
          f'tau = {args.tau} s')
    for nm, v in (('roll', roll), ('pitch', pitch)):
        print(f'  {nm:6s} mean {v.mean():+6.2f}  std {v.std():5.2f}  '
              f'[{v.min():+6.2f}, {v.max():+6.2f}] deg')

    idx, out, missed = [], [], 0
    seen = set()
    for name in INFOS:
        path = os.path.join(args.data_root, name)
        if not os.path.exists(path):
            continue
        with open(path, 'rb') as f:
            data_list = pickle.load(f)['data_list']
        for s in data_list:
            # the LiDAR filename is the capture time in nanoseconds, on the
            # same clock as the IMU, and is the join key on both sides
            stem = os.path.basename(s['lidar_points']['lidar_path'])[:-4]
            if stem in seen:
                continue
            seen.add(stem)
            ts = int(stem) / 1e9
            k = int(np.clip(np.searchsorted(t, ts), 1, len(t) - 1))
            if abs(t[k - 1] - ts) < abs(t[k] - ts):
                k -= 1
            if abs(t[k] - ts) > args.max_dt:
                missed += 1
                continue
            idx.append(stem)
            out.append((roll[k], pitch[k], d_roll[k], d_pitch[k], rock[k]))

    idx = np.asarray(idx)
    out = np.asarray(out, dtype=np.float32)
    order = np.argsort(idx)
    idx, out = idx[order], out[order]
    tilt = np.hypot(out[:, 0], out[:, 1])
    rk = out[:, 4]
    print(f'\n{len(idx)} annotated frames matched, {missed} with no IMU sample '
          f'within {args.max_dt * 1000:.0f} ms')
    print(f'  tilt magnitude  median {np.median(tilt):.2f}  '
          f'p90 {np.percentile(tilt, 90):.2f}  max {tilt.max():.2f} deg')
    print('  induced displacement at 150 m: '
          f'median {np.tan(np.radians(np.median(tilt))) * 150:.2f} m, '
          f'p90 {np.tan(np.radians(np.percentile(tilt, 90))) * 150:.2f} m')
    print(f'  rocking rate    median {np.median(rk):.2f}  '
          f'p90 {np.percentile(rk, 90):.2f}  max {rk.max():.2f} deg/s')
    print('  sea-state terciles (the numbers MaritimeMetric bands on): '
          f'{np.percentile(rk, 33):.2f}, {np.percentile(rk, 67):.2f} deg/s')

    dest = args.out or os.path.join(args.data_root, 'ego_attitude.npz')
    np.savez(dest, token=idx, roll=out[:, 0], pitch=out[:, 1],
             roll_rate=out[:, 2], pitch_rate=out[:, 3], rock=rk)
    print(f'\nwrote {dest}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
