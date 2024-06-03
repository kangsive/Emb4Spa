import subprocess

batch_sizes = [64, 128, 256]
lrs = [0.0001, 0.00005, 0.00001]
epochs = 50
runs = 5


for batch_size in batch_sizes:
    for lr in lrs:
        group_name = f"finetune_bs{batch_size}_lr{lr}"
        command = ["python",
                   "main_finetune.py",
                    "--epochs",
                    str(epochs),
                    "--runs",
                    str(runs),
                    "--batch_size",
                    str(batch_size),
                    "--group_name",
                    str(group_name),
                    "--lr",
                    str(lr),
                    "--weight_decay",
                    "0",
                    "--pretrain_model",
                    "weights/potae_repeat2_lr0.0001_dmodel384_bs512_epoch200_runname-mild-darkness-78.pth",
                    "--fea_dim",
                    "7",
                    "--d_model",
                    "384",
                    "--hidden_dim",
                    "64",
                    "--num_heads",
                    "6",
                    "--ffn_dim",
                    "1024",
                    "--layer_repeat",
                    "2",
                    "--dropout",
                    "0.1",
                    "--fine_tune",
                    ]

        subprocess.run(command)
