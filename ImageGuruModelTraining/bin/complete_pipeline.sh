#!/bin/bash

# Usage: bin/complete_pipeline.sh <product_type> <exp name> <GPU>
# Usage example: bin/complete_pipeline.sh sofa individual_trained 2
# Where the 'sofa' parameter is used to find the directory for input and output files.

PT=$1
EXP=$2
DEVICE=$3

BASEDIR=/home/ubuntu/efs/nikhgarg/image_guru

${BASEDIR}/ImageGuruModelTraining/bin/train.sh ${PT} ${EXP} ${DEVICE}

${BASEDIR}/ImageGuruModelInference/bin/predict.sh ${PT} ${EXP} ${DEVICE}

${BASEDIR}/ImageGuruModelTraining/bin/eval.sh ${PT} ${EXP}
