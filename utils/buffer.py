import torch
import random

# Image Buffer to store batches of generated images for a particular class
class ImageBuffer():

    def __init__(self, max_size=50, device="cpu"):
        
        self.buffer = []
        self.max_size = max_size
        self.device = device

    def get_size(self):

        return len(self.buffer)
    

    def insert(self, minibatch):
        
        new_images = [torch.unsqueeze(img.detach(), 0).cpu() for img in minibatch]

        # Add until buffer is full
        if len(self.buffer) < self.max_size:
            space_left = self.max_size - len(self.buffer)
            self.buffer.extend(new_images[:space_left])

            return minibatch

        else:
            
            n_new = len(new_images)
            n_half = n_new // 2

            # First randomly choose half of the new images and half of the old for the current batch

            # Randomly choose half of the new images
            new_idxs = random.sample(range(n_new), n_half)
            new_part = [new_images[i] for i in new_idxs]

            # Randomly pick half old images from buffer
            old_idxs = random.sample(range(len(self.buffer)), n_half)
            old_part = [self.buffer[i] for i in old_idxs]

            mixed = torch.cat(new_part + old_part, dim=0).to(self.device)

            # Second select half of the new images to replace some old ones

            replace_new_idxs = random.sample(range(n_new), n_half)
            replace_new_part = [new_images[i] for i in replace_new_idxs]
            replace_idxs  = random.sample(range(len(self.buffer)), n_half)

            # Replace some old entries in the buffer with the selected new ones
            for old_i, new_img in zip(replace_idxs, replace_new_part):
                self.buffer[old_i] = new_img

            # Return a mix of current + buffered images
            return mixed

