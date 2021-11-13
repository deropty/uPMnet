#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
################################################################################################
PATH_ROOT=/home/zangxh/Person-ReID/uPMnet/
################################################################################################
PART=eight_part
N_PART=8
MAX_STEPS=60
SAMPLES=200
###############################################################################################
for MODLE_NAME in 'mobilenet_v1' 'resnet_v1_50'
do
    for RELATION in 'local' 'global'
    do
        if [ ${MODLE_NAME} == 'mobilenet_v1' ];
        then
            if [ ${RELATION} == 'global' ];
            then
                FEATURE_DIM=256
            else
                FEATURE_DIM=1024
            fi
            SUFFIX='_1.0_224'
        else
            if [ ${RELATION} == 'global' ];
            then
                FEATURE_DIM=256
            else
                FEATURE_DIM=2048
            fi
            SUFFIX=''
        fi

	echo "================= using ${MODLE_NAME} with ${RELATION} module ======================"
        NUM_SAMPLES=369656
        NUM_CLASS=2196
        # where to store the results of training
        TRAIN_DIR=${PATH_ROOT}results/DukeMTMC-VideoReID/${MODLE_NAME}/${RELATION}/${PART}/
        # where to store the checkpoints
        CHECKPOINT_PATH=${TRAIN_DIR}models/
        # where the pretrained model is saved
        PRE_TRAIN=${PATH_ROOT}checkpoints/${MODLE_NAME}${SUFFIX}.ckpt
        # dir of training data
        DATASET_DIR=${PATH_ROOT}tfrecords/traindata/DukeMTMC-VideoReID/
        # dir stores the pre-extracted feature for initialisation
        # FEATURE_DIR=${PATH_ROOT}DAL/traindata_feature/MARS/
        
        python train.py \
              --train_dir=${CHECKPOINT_PATH} \
              --dataset_name=DukeMTMC-VideoReID \
              --dataset_dir=${DATASET_DIR} \
              --model_name=${MODLE_NAME} \
              --image_size=256 \
              --max_steps=${MAX_STEPS} \
              --store_interval=${MAX_STEPS} \
              --batch_size=64 \
              --optimizer=sgd \
              --pretrained_model_checkpoint_path=${PRE_TRAIN} \
              --num_gpus=1 \
              --num_readers=4 \
              --num_preprocessing_threads=16 \
              --margin=0.5 \
              --num_classes=${NUM_CLASS} \
              --num_samples=${NUM_SAMPLES} \
              --num_cams=8 \
              --warm_up_epochs=1 \
              --feature_dim=${FEATURE_DIM} \
              --n_part=${N_PART} \
              --relation=${RELATION}
              # --feature_dir=${FEATURE_DIR}
        
        NUM_SAMPLES=445764
        # where the tfrecords are saved: DATA_DIR
        DATA_DIR=${PATH_ROOT}tfrecords/testdata/DukeMTMC-VideoReID/
        # where the results are saved: TRAIN_DIR
        TRAIN_DIR=${PATH_ROOT}results/DukeMTMC-VideoReID/${MODLE_NAME}/${RELATION}/${PART}/
        # where to stored the extracted feature: OUT_DIR
        # name of the checkpoint: CHECKPOINT_INDEX
        OUT_DIR=${TRAIN_DIR}features/
        # where the checkpoint is saved: CHECKPOINT_PATH
        CHECKPOINT_PATH=${TRAIN_DIR}models/
        # name of the activation layer to extract feature: FEATURE_NAME
        FEATURE_NAME=AvgPool_
        
        python test.py \
          --dataset_name=data \
          --model_name=${MODLE_NAME} \
          --feature_type=${FEATURE_NAME} \
          --batch_size=1 \
          --num_readers=1 \
          --image_size=256 \
          --feature_dim=${FEATURE_DIM} \
          --checkpoint_dir=${CHECKPOINT_PATH} \
          --dataset_dir=${DATA_DIR} \
          --feature_dir=${OUT_DIR} \
          --num_classes=${NUM_CLASS} \
          --num_samples=${SAMPLES} \
          --num_matfiles=2 \
          --n_part=${N_PART} \
          --relation=${RELATION}
    done
done
################################################################################################
