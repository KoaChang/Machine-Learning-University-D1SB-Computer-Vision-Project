#!/bin/bash

source activate pytorch_p36

PT=$1
EXP=$2
BASEDIR=/home/ubuntu/efs/nikhgarg/image_guru
DATADIR=${BASEDIR}/data
FL_CLASSES=${DATADIR}/annotations_v2/furniture/classes.txt
EXP_DIR=${DATADIR}/experiments/${EXP}/${PT}

export PYTHONPATH=${BASEDIR}/ImageGuruModelTraining/src:${BASEDIR}/ImageGuruModelInference/src:${PYTHONPATH}

for partition in train val test; do

  python3 ${BASEDIR}/ImageGuruModelTraining/src/image_guru_model_training/evaluate.py \
          --input ${EXP_DIR}/predictions/${partition}.predictions.txt \
          --classes ${FL_CLASSES} \
          --output-dir ${EXP_DIR}/evaluation/${partition} \
          --multi_labels_for_multi_class ignore

done
