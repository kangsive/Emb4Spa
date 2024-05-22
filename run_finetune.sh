python main_finetune.py --batch_size 128 \
	--no_wandb \
	--lr 0.003 \
	--train_data dataset/buildings_train_v8.npz \
	--eval_data dataset/buildings_test_v8.npz \
	--pretrain_model no \
	--epochs 50 \
	--fea_dim 5 \
	# --no_schedule \
	# --fine_tune \
