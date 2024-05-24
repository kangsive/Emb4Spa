python main_finetune.py \
	--batch_size 256 \
	--no_wandb \
	--lr 0.003 \
	--train_data dataset/mnist_polygon_train_10k.npz \
	--eval_data dataset/mnist_polygon_test_2k.npz \
	--pretrain_model weights/potae_repeat2_lr0.0001_dmodel384_bs512_epoch200_runname-mild-darkness-78.pth \
	--epochs 50 \
	--weight_decay 0 \
	--fea_dim 7 \
	--d_model 384 \
	--hidden_dim 64 \
	--num_heads 6 \
	--ffn_dim 1024 \
	--layer_repeat 2 \
	--dropout 0.1 \
	--fine_tune \
	# --no_schedule \
