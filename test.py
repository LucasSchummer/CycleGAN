import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import os
from tqdm import tqdm
from models.cyclegan import CycleGAN
from utils.dataset import UnpairedDataset
from utils.checkpoint_manager import CheckpointManager

run_name = "HorseZebra_1"
checkpoint = "200_checkpoint.pth"
save_dir = f"eval_results/{run_name}"

batch_size = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = CycleGAN(device=device).to(device)
dataset = UnpairedDataset("data/horse2zebra", mode="test")
dataloader = DataLoader(dataset, batch_size, shuffle=True)
checkpoint_manager = CheckpointManager(f"checkpoints/{run_name}")

checkpoint_manager.load(model, checkpoint, device)

os.makedirs(os.path.join(save_dir, "A_to_B"), exist_ok=True)
os.makedirs(os.path.join(save_dir, "B_to_A"), exist_ok=True)

progress_bar = tqdm(
    enumerate(dataloader),
    total=len(dataloader),
    leave=False,
    unit="batch",
    colour="green"
)

for i, (batch) in progress_bar:

    with torch.no_grad():
        
        real_A = batch["A"].to(device)
        real_B = batch["B"].to(device)

        # Generate fake and cycle images
        fake_B = model.G_AB(real_A)
        cyc_A = model.G_BA(fake_B)

        fake_A = model.G_BA(real_B)
        cyc_B = model.G_AB(fake_A)

        # Rescale from [-1, 1] → [0, 1] for saving
        def rescale(x):
            return (x + 1) / 2.0

        # Make a grid horizontally: [real | fake | cycle]
        grid_AtoB = make_grid(torch.cat([rescale(real_A), rescale(fake_B), rescale(cyc_A)], dim=0), nrow=3)
        grid_BtoA = make_grid(torch.cat([rescale(real_B), rescale(fake_A), rescale(cyc_B)], dim=0), nrow=3)

        # Save
        save_image(grid_AtoB, os.path.join(save_dir, "A_to_B", f"{i:04d}.png"))
        save_image(grid_BtoA, os.path.join(save_dir, "B_to_A", f"{i:04d}.png"))

print(f"Images saved at {save_dir}")
