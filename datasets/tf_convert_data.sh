#!/bin/bash
export CUDA_VISIBLE_DEVICES=
################################################################################################
# TODO: supply your project path root here
PATH_ROOT=/home/zangxh/Person-ReID/uPMnet
################################################################################################
# # convert training data to tfrecords
# for i in {1..2};
# do
# 	for DATA_TYPE in train
# 	do
# 		# Convert tfrecords of PRID2011
# 		# TODO: supply your image data path here
# 		# Where the original image data is stored
# 		DATA_DIR=/home/zangxh/datasets/prid_2011/multi_shot/
# 		# Where the tfrecords should be stored
# 		OUT_DIR=${PATH_ROOT}/tfrecords/${DATA_TYPE}data/PRID2011/split$i/
# 		# The txt file that lists all the data splits
# 		FILENAME=${PATH_ROOT}/evaluation/datasplits/PRID2011/split$i/${DATA_TYPE}data.txt
# 
#                 cd ${DATA_DIR}; ln -s cam_a cam1; ln -s cam_b cam2; cd ${PATH_ROOT}
# 		python datasets/convert_data_to_tfrecords.py \
# 		    --data_type=${DATA_TYPE} \
# 		    --dataset_dir=${DATA_DIR} \
# 		    --output_dir=${OUT_DIR} \
# 		    --filename=${FILENAME} \
# 		    --num_tfrecords=10
#                 cd ${DATA_DIR}; rm cam1 cam2; cd ${PATH_ROOT}
# 
# 		# Convert tfrecords of iLIDS-VID
# 		# TODO: supply your image data path here
# 		# Where the original image data is stored
# 		# DATA_DIR=YOUR_DATA_PATH/iLIDS-VID/video_data/
# 		DATA_DIR=/home/zangxh/datasets/i-LIDS-VID/sequences/
# 		# Where the tfrecords should be stored
# 		OUT_DIR=${PATH_ROOT}/tfrecords/${DATA_TYPE}data/iLIDS-VID/split$i/
# 		# The txt file that lists all the data information
# 		FILENAME=${PATH_ROOT}/evaluation/datasplits/iLIDS-VID/split$i/${DATA_TYPE}data.txt
# 
# 		python datasets/convert_data_to_tfrecords.py \
# 		    --data_type=${DATA_TYPE} \
# 		    --dataset_dir=${DATA_DIR} \
# 		    --output_dir=${OUT_DIR} \
# 		    --filename=${FILENAME} \
# 		    --num_tfrecords=10
# 
# 	done
# done
# 
# 
# ################################################################################################
# # convert test data to tfrecords
# for i in {1..2};
# do
# 	for DATA_TYPE in test
# 	do
# 		# Convert tfrecords of PRID2011
# 		# TODO: supply your image data path here
# 		# Where the original image data is stored
# 		DATA_DIR=/home/zangxh/datasets/prid_2011/multi_shot/
# 		# Where the tfrecords should be stored
# 		OUT_DIR=${PATH_ROOT}/tfrecords/${DATA_TYPE}data/PRID2011/split$i/
# 		# The txt file that lists all the data splits
# 		FILENAME=${PATH_ROOT}/evaluation/datasplits/PRID2011/split$i/${DATA_TYPE}data.txt
# 
#                 cd ${DATA_DIR}; ln -s cam_a cam1; ln -s cam_b cam2; cd ${PATH_ROOT}
# 		python datasets/convert_data_to_tfrecords.py \
# 		    --data_type=${DATA_TYPE} \
# 		    --dataset_dir=${DATA_DIR} \
# 		    --output_dir=${OUT_DIR} \
# 		    --filename=${FILENAME} \
# 		    --num_tfrecords=1
#                 cd ${DATA_DIR}; rm cam1; rm cam2; cd ${PATH_ROOT}
# 
# 		# Convert tfrecords of iLIDS-VID
# 		# TODO: supply your image data path here
# 		# Where the original image data is stored
# 		# DATA_DIR=YOUR_DATA_PATH/iLIDS-VID/video_data/
# 		DATA_DIR=/home/zangxh/datasets/i-LIDS-VID/sequences/
# 		# Where the tfrecords should be stored
# 		OUT_DIR=${PATH_ROOT}/tfrecords/${DATA_TYPE}data/iLIDS-VID/split$i/
# 		# The txt file that lists all the data information
# 		# The txt file that lists all the training data information
# 		FILENAME=${PATH_ROOT}/evaluation/datasplits/iLIDS-VID/split$i/${DATA_TYPE}data.txt
# 
# 		python datasets/convert_data_to_tfrecords.py \
# 		    --data_type=${DATA_TYPE} \
# 		    --dataset_dir=${DATA_DIR} \
# 		    --output_dir=${OUT_DIR} \
# 		    --filename=${FILENAME} \
# 		    --num_tfrecords=1
# 
# 
# 	done
# done

################################################Duke-MTMC-VideoReID##########################################
# DATA_TYPE=train
# # Convert tfrecords of MARS
# # TODO: supply your image data path here
# # Where the original image data is stored
# DATA_DIR=/home/zangxh/datasets/DukeMTMC-VideoReID/
# # Where the tfrecords should be stored
# OUT_DIR=${PATH_ROOT}/tfrecords/${DATA_TYPE}data/DukeMTMC-VideoReID/
# # The txt file that lists all the data splits
# FILENAME=${PATH_ROOT}/evaluation/datasplits/DukeMTMC-VideoReID/${DATA_TYPE}data.txt
# 
# python datasets/convert_data_to_tfrecords.py \
#     --data_type=${DATA_TYPE} \
#     --dataset_dir=${DATA_DIR} \
#     --output_dir=${OUT_DIR} \
#     --filename=${FILENAME} \
#     --num_tfrecords=100
# 
DATA_TYPE=test
# Convert tfrecords of MARS
# TODO: supply your image data path here
# Where the original image data is stored
DATA_DIR=/home/zangxh/datasets/DukeMTMC-VideoReID/
# Where the tfrecords should be stored
OUT_DIR=${PATH_ROOT}/tfrecords/${DATA_TYPE}data/DukeMTMC-VideoReID/
# The txt file that lists all the data splits
FILENAME=${PATH_ROOT}/evaluation/datasplits/DukeMTMC-VideoReID/${DATA_TYPE}data.txt

python datasets/convert_data_to_tfrecords.py \
    --data_type=${DATA_TYPE} \
    --dataset_dir=${DATA_DIR} \
    --output_dir=${OUT_DIR} \
    --filename=${FILENAME} \
    --num_tfrecords=1
