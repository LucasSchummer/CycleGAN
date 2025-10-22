import torch
import torch.nn as nn
import torch.nn.functional as F
from models.generator import Generator
from models.discriminator import Discriminator
from utils.buffer import ImageBuffer


class CycleGAN(nn.Module):

    def __init__(self, lr, lbda, device):
        super().__init__()
        
        self.D_A = Discriminator() # Discriminator for class A
        self.D_B = Discriminator() # Discriminator for class B
        self.G_AB = Generator() # Generator from class A to B
        self.G_BA = Generator() # Generator from class B to A

        self.optimizer_G = torch.optim.Adam(list(self.G_AB.parameters()) + list(self.G_BA.parameters()), lr = lr)
        self.optimizer_D = torch.optim.Adam(list(self.D_A.parameters()) + list(self.D_B.parameters()), lr = lr)

        self.buffer_A = ImageBuffer(50, device)
        self.buffer_B = ImageBuffer(50, device)

        self.lbda = lbda
        self.global_step = 0
        self.epoch = 0
        self.device = device

    def train_step(self, batch):

        real_A = batch["A"].to(self.device)
        real_B = batch["B"].to(self.device)

        fake_A = self.G_BA(real_B)
        fake_B = self.G_AB(real_A)

        cyc_A = self.G_BA(fake_B)
        cyc_B = self.G_AB(fake_A)

        # Train Generators

        self.set_requires_grad([self.D_A, self.D_B], False) # No gradient for discriminators while optimizing generators

        pred_fake_A = self.D_A(fake_A)
        pred_fake_B = self.D_B(fake_B)

        loss_G_AB = F.mse_loss(pred_fake_B, torch.ones_like(pred_fake_B))
        loss_G_BA = F.mse_loss(pred_fake_A, torch.ones_like(pred_fake_A))
        loss_id = F.l1_loss(real_A, fake_B) + F.l1_loss(real_B, fake_A)
        loss_cyc = F.l1_loss(real_A, cyc_A) + F.l1_loss(real_B, cyc_B)

        loss_G = loss_G_AB + loss_G_BA + self.lbda * loss_cyc + self.lbda * .5 * loss_id

        self.optimizer_G.zero_grad()
        loss_G.backward()
        self.optimizer_G.step()

        # Train Discriminators

        self.set_requires_grad([self.D_A, self.D_B], True)

        # Insert half of the images to the buffers and get a mixed batch (half new, half old)
        mixed_fake_A = self.buffer_A.insert(fake_A)
        mixed_fake_B = self.buffer_B.insert(fake_B)

        pred_real_A = self.D_A(real_A)
        pred_real_B = self.D_B(real_B)

        loss_D_A = F.mse_loss(pred_real_A, torch.ones_like(pred_real_A)) + F.mse_loss(self.D_A(mixed_fake_A.detach()), torch.zeros_like(pred_real_A))
        loss_D_B = F.mse_loss(pred_real_B, torch.ones_like(pred_real_B)) + F.mse_loss(self.D_B(mixed_fake_B.detach()), torch.zeros_like(pred_real_B))

        loss_D = .5 * (loss_D_A + loss_D_B)

        self.optimizer_D.zero_grad()
        loss_D.backward()
        self.optimizer_D.step()

        return {
            "loss_G": loss_G.item(),
            "loss_D": loss_D.item(),
            "loss_G_AB": loss_G_AB.item(),
            "loss_G_BA": loss_G_BA.item(),
            "loss_D_A": loss_D_A.item(),
            "loss_D_B": loss_D_B.item(),
            "loss_cyc": loss_cyc.item(),
            "loss_id" : loss_id.item(),
            "D_A_real": pred_real_A.mean().item(),
            "D_A_fake": pred_fake_A.mean().item(),
            "D_B_real": pred_real_B.mean().item(),
            "D_B_fake": pred_fake_B.mean().item()
        }
    
    def set_requires_grad(self, nets, requires_grad):

        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad
    
    def get_fake_images(self, batch):

        with torch.no_grad():
            real_A = batch["A"]
            real_B = batch["B"]

            fake_B = self.G_AB(real_A)
            fake_A = self.G_BA(real_B)

            return fake_A, fake_B
        
    def get_state(self):
        return {
            "G_AB": self.G_AB.state_dict(),
            "G_BA": self.G_BA.state_dict(),
            "D_A": self.D_A.state_dict(),
            "D_B": self.D_B.state_dict(),
            "optimizer_G": self.optimizer_G.state_dict(),
            "optimizer_D": self.optimizer_D.state_dict(),
            "buffer_A": self.buffer_A.buffer,
            "buffer_B": self.buffer_B.buffer,
            "step": self.global_step,
            "epoch": self.epoch
        }