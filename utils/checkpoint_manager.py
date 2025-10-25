import os
import torch
from collections import deque

class CheckpointManager:
    def __init__(self, save_dir, max_checkpoints=5):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoints = deque()

    def save(self, model, epoch, step, name="checkpoint.pth"):
        path = os.path.join(self.save_dir, f"{epoch}_{name}")
        model.epoch = epoch
        model.global_step = step
        torch.save(model.get_state(), path)
        self.checkpoints.append(path)
        if len(self.checkpoints) > self.max_checkpoints:
            old = self.checkpoints.popleft()
            if os.path.exists(old):
                os.remove(old)
        print(f"Saved checkpoint {path}")

    def load(self, model, path, device):

        path = os.path.join(self.save_dir, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint found at {path}")
        
        checkpoint = torch.load(path, map_location=device)
        print(f"Loading checkpoint from '{path}' (epoch {checkpoint['epoch']})...")

        # Load model weights
        model.G_AB.load_state_dict(checkpoint["G_AB"])
        model.G_BA.load_state_dict(checkpoint["G_BA"])
        model.D_A.load_state_dict(checkpoint["D_A"])
        model.D_B.load_state_dict(checkpoint["D_B"])

        # Load optimizer states
        model.optimizer_G.load_state_dict(checkpoint["optimizer_G"])
        model.optimizer_D.load_state_dict(checkpoint["optimizer_D"])

        # Restore image buffers
        model.buffer_A.buffer = checkpoint["buffer_A"]
        model.buffer_B.buffer = checkpoint["buffer_B"]

        global_step = checkpoint["step"]
        epoch = checkpoint["epoch"]

        return epoch, global_step