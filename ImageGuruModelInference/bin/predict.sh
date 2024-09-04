#!/bin/bash

# Usage example: bin/predict.sh sofa individual_models 1
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

export PYTHONPATH=${BASEDIR}/ImageGuruModelInference/src:${PYTHONPATH}

for partition in train val test; do

  python3 ${BASEDIR}/ImageGuruModelInference/src/image_guru_model_inference/tools/predict.py \
          --input ${SPLIT_DIR}/${partition}.txt \
          --output ${EXP_DIR}/predictions/${partition}.predictions.txt \
          --classes ${FL_CLASSES} \
          --image_download_dir ${DATADIR}/images \
          --model-path ${EXP_DIR}/models/model-best.pth \
          --model-type resnet50 \
          --device ${DEVICE} \
          --batch_size 32 \
          --num_workers 0

done
