import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
from models.cyclegan import CycleGAN
from utils.dataset import UnpairedDataset
from utils.checkpoint_manager import CheckpointManager


epochs = 50
lr = 2e-4
lbda = 10
n_steps_log = 10
n_epochs_log = 1
n_epochs_checkpoint = 2

run_name = "HorseZebra_1"

start_from_checkpoint = True
checkpoint = "checkpoints/HorseZebra_1/13_4676_checkpoint.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = CycleGAN(lr, lbda, device).to(device)
dataset = UnpairedDataset("data/horse2zebra")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
writer = SummaryWriter(log_dir=f"runs/{run_name}")
checkpoint_manager = CheckpointManager(f"checkpoints/{run_name}", max_checkpoints=5)

if start_from_checkpoint:
    epoch_start, step = checkpoint_manager.load(model, checkpoint, device)
else:
    epoch_start, step = 0, 0

for epoch in range(epoch_start, epochs):
    start_time = time.time()
        
    # tqdm progress bar for batches in the epoch
    progress_bar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"Epoch [{epoch+1}/{epochs}]",
        leave=False,
        unit="batch",
        colour="green"
    )

    for batch_idx, (batch) in progress_bar:

        metrics = model.train_step(batch)

        if step % n_steps_log == 0:
            writer.add_scalar("Loss/G/Total", metrics["loss_G"], step)
            writer.add_scalar("Loss/G/G_AB", metrics["loss_G_AB"], step)
            writer.add_scalar("Loss/G/G_BA", metrics["loss_G_BA"], step)
            writer.add_scalar("Loss/D/Total", metrics["loss_D"], step)
            writer.add_scalar("Loss/D/D_A", metrics["loss_D_A"], step)
            writer.add_scalar("Loss/D/D_B", metrics["loss_D_B"], step)
            writer.add_scalar("Loss/Cycle", metrics["loss_cyc"], step)
            writer.add_scalar("Loss/Identity", metrics["loss_id"], step)
            writer.add_scalar("Discriminator_A/Real", metrics["D_A_real"], step)
            writer.add_scalar("Discriminator_A/Fake", metrics["D_A_fake"], step)
            writer.add_scalar("Discriminator_B/Real", metrics["D_B_real"], step)
            writer.add_scalar("Discriminator_B/Fake", metrics["D_B_fake"], step)

        step += 1

    if epoch % n_epochs_log == 0:
        fake_A, fake_B = model.get_fake_images(batch)
        writer.add_images("Fake/B_from_A", (fake_B + 1) / 2, epoch)
        writer.add_images("Fake/A_from_B", (fake_A + 1) / 2, epoch)

    if epoch % n_epochs_checkpoint == 0:
        checkpoint_manager.save(model, epoch, step)

    print(f"Epoch [{epoch+1}/{epochs}] completed in {time.time() - start_time:.2f}s")

# tensorboard --logdir runs/HorseZebra_1