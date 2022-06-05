# $1 export dir

python inference_pc.py \
--model_type regular \
--pc point_clouds/chair.obj \
--rx -2.08 --ry  0.13 --rz 1.46 \
--scale 1.09 --flip_z --dy 353 \
--rgb 255 120 120 \
--export_dir $1