import subprocess

batch_sizes = [64, 128, 256]
lrs = [0.01, 0.005, 0.001]
epochs = 150
early_stop = False # apply early stop
no_schedule = False # no lr schedule
runs = 10


for batch_size in batch_sizes:
    for lr in lrs:
        group_name = f"mlp_bs{batch_size}_lr{lr}"
        command = ["python",
                   "main_+mlp.py",
                    "--epochs",
                    str(epochs),
                    "--runs",
                    str(runs),
                    "--batch_size",
                    str(batch_size),
                    "--no_embedding",
                    "--group_name",
                    str(group_name),
                    "--lr",
                     str(lr)]

        subprocess.run(command)

