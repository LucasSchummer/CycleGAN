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
        path = os.path.join(self.save_dir, f"{epoch}_{step}_{name}")
        model.epoch = epoch
        model.global_step = step
        torch.save(model.get_state(), path)
        self.checkpoints.append(path)
        if len(self.checkpoints) > self.max_checkpoints:
            old = self.checkpoints.popleft()
            if os.path.exists(old):
                os.remove(old)
        print(f"Saved checkpoint {path}")