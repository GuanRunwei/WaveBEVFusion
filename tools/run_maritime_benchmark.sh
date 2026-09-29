#!/usr/bin/env bash
# Train and then test every maritime3d baseline, one job per GPU.
#
# Each job runs 40 epochs on maritime_infos_train_nonempty.pkl, validating every
# 5 epochs on the val split, then evaluates the final checkpoint on the held-out
# test split. Logs land in work_dirs/<name>/ and work_dirs/<name>/test.log.
set -u

PY=/data/miniconda/envs/mmdet3d/bin/python
ROOT=/data/xlsun/datasets/maritime3d/mmdetection3d
CFG_DIR=projects/Maritime3D/configs
EPOCHS=40
cd "$ROOT" || exit 1

run() {  # run <name> <gpu>
    local name=$1 gpu=$2
    local cfg=$CFG_DIR/${name}_maritime-3d-4class.py
    local wd=work_dirs/$name
    CUDA_VISIBLE_DEVICES=$gpu $PY tools/train.py "$cfg" --work-dir "$wd" \
        > "$wd.train.log" 2>&1
    echo "TRAIN_EXIT=$? $name" >> "$wd.train.log"
    CUDA_VISIBLE_DEVICES=$gpu $PY tools/test.py "$cfg" \
        "$wd/epoch_${EPOCHS}.pth" --work-dir "$wd" > "$wd.test.log" 2>&1
    echo "TEST_EXIT=$? $name" >> "$wd.test.log"
}

run_dist() {  # run_dist <name> <gpus-csv> <n> <port>
    local name=$1 gpus=$2 n=$3 port=$4
    local cfg=$CFG_DIR/${name}_maritime-3d-4class.py
    local wd=work_dirs/$name
    # BEVFusion is the slowest job; two ranks at batch 2 give it the same
    # effective batch as the single-GPU TransFusion-L baseline it is compared to
    #
    # dist_train.sh invokes a bare `python`, which resolves to the base conda env
    # (no torch). Put the mmdet3d env first on PATH so torchrun and the workers
    # both land in the right interpreter.
    CUDA_VISIBLE_DEVICES=$gpus PORT=$port PATH=$(dirname $PY):$PATH \
        bash tools/dist_train.sh "$cfg" "$n" \
        --work-dir "$wd" > "$wd.train.log" 2>&1
    echo "TRAIN_EXIT=$? $name" >> "$wd.train.log"
    CUDA_VISIBLE_DEVICES=${gpus%%,*} $PY tools/test.py "$cfg" \
        "$wd/epoch_${EPOCHS}.pth" --work-dir "$wd" > "$wd.test.log" 2>&1
    echo "TEST_EXIT=$? $name" >> "$wd.test.log"
}

mkdir -p work_dirs
run pointpillars       0 &
run second             1 &
run transfusion_lidar  2 &
run petr               5 &
run_dist bevfusion_lidar-cam 3,4 2 29511 &
wait
echo ALLDONE
