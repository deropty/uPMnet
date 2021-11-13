python visualize.py 
--model_name=mobilenet_v1 \
--checkpoint_path=./results/iLIDS-VID/mobilenet_v1/global/eight_part/models/split1/ \ 
--input=./images/cam2_person313_01851.png \
--image_size=256 \
--num_classes=178 \
--layer_name=PrePool \
--n_part=2 \
--relation=global \
--steps=15000
