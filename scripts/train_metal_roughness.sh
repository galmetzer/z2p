# Arguments
# 1: train data dir
# 2: test data dir
# 3: export dir

python train.py \
--lr 3e-4 \
--data $1 \
--test_data $2 \
--export_dir $3 \
--batch_size 4 --num_workers 4 \
--epochs 100 --log_iter 1000 \
--losses masked_mse intensity masked_pixel_intensity \
--l_weight 1 0.7 1 --splat_size 3 \
--controls colors light_sph_relative metallic roughness \
--log_iter 1000