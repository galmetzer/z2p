# $1 export dir

python inference_pc.py \
--model_type regular \
--pc point_clouds/goat.obj \
--rx -1.15 --ry  0.64 --rz -2.46 \
--scale 1.25 --flip_z --dy 331 \
--export_dir $1