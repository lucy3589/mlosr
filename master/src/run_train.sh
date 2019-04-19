python main.py --batch_size 64	--iterations 1000 --alpha 0.5 --sigma 1e-12 \
	           --gain 1.0 --no_init_filters 64 --latent_size 600 --betas 0.5 \
	           --depth 92 --growth_rate 24 --stats_frequency 1000 --image_channel 3 \
	           --image_size 32 --gpu True --verbose True --dataset_path ../../datasets/ \
	           --dataset_name cifar10 --model_mode train --method mlosr --dataset_file_format hdf5 \
	           --mlosr_model vggnet --dist_type L1 --lr 3e-4 --no_closed 10 --no_open 1 --no_total 11 \
	           --tail_size 20 --default False --tanh_flag True --separate_flag True 
