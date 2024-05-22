export PYTHONPATH=$PWD/utils:$PYTHONPATH

python main_pretrain.py \
	--batch_size 256 \
	--no_wandb \
	--lr 0.002 \
	--data_path dataset/mnist_polygon_train_10k.npz \
	--weights '' \
	--epochs 100 \
	--no_schedule \
	--weight_decay 0 \
	--fea_dim 7 \
	--d_model 36 \
	--hidden_dim 64 \
	--num_heads 4 \
	--ffn_dim 36 \
	--layer_repeat 3 \
	--dropout 0.2 \
