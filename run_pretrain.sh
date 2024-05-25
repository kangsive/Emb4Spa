export PYTHONPATH=$PWD/utils:$PYTHONPATH

python main_pretrain.py \
        --data_path dataset/synthetic_complex_200000.npz \
        --batch_size 512 \
        --lr 0.0001 \
        --weights '' \
        --epochs 200 \
        --no_schedule \
        --weight_decay 0 \
        --fea_dim 7 \
        --d_model 384 \
        --hidden_dim 64 \
        --num_heads 6 \
        --ffn_dim 1024 \
        --layer_repeat 2 \
        --dropout 0.1 \