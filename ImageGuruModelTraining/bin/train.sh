#!/bin/bash

# Usage example: bin/train.sh sofa individual_models 1
# Where the 'sofa' parameter is used to find the directory for input and output files.

source activate pytorch_p36

PT=$1
EXP=$2
DEVICE=$3
BASEDIR=/home/ubuntu/efs/nikhgarg/image_guru
DATADIR=${BASEDIR}/data
FL_CLASSES=${DATADIR}/annotations_v2/furniture/classes.txt
SPLIT_DIR=${DATADIR}/annotations_v2/furniture/splits/${PT}
EXP_DIR=${DATADIR}/experiments/${EXP}/${PT}

mkdir -p ${EXP_DIR}

export PYTHONPATH=${BASEDIR}/ImageGuruModelTraining/src:${BASEDIR}/ImageGuruModelInference/src:${PYTHONPATH}

python3 ${BASEDIR}/ImageGuruModelTraining/src/image_guru_model_training/tools/train.py \
        --train-annotations ${SPLIT_DIR}/train.txt \
        --val-annotations ${SPLIT_DIR}/val.txt \
        --classes ${FL_CLASSES} \
        --image_download_dir ${DATADIR}/images \
        --model-dir ${EXP_DIR}/models \
        --device ${DEVICE} \
        --multi_labels_for_multi_class ignore \
        --batch_size 64 \
        --num_workers 0 \
        --start_lr 0.01 \
        --num_epochs 50 &> ${EXP_DIR}/train.out
