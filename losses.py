import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def forward(self, inputs, targets, smooth=1e-5):
        inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        return 1.0 - ((2. * intersection + smooth) / (union + smooth)).mean()


class CompositeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, inputs, targets):
        return self.bce(inputs, targets) + self.dice(inputs, targets)


class CompositeBoundaryLoss(nn.Module):
    def __init__(self, lambda_seg=1.0, lambda_bnd=0.15):
        super().__init__()
        self.lambda_seg = lambda_seg
        self.lambda_bnd = lambda_bnd
        self.seg_criterion = CompositeLoss()
        self.boundary_criterion = nn.MSELoss()

    def get_sobel_boundary(self, mask):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=mask.device).view(1, 1,
                                                                                                                   3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=mask.device).view(1, 1,
                                                                                                                   3, 3)

        grad_x = F.conv2d(mask, sobel_x, padding=1)
        grad_y = F.conv2d(mask, sobel_y, padding=1)
        magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

        # CRITICAL FIX: Clamp to [0, 1] so it perfectly matches the sigmoid output range!
        return torch.clamp(magnitude, 0.0, 1.0)

    def forward(self, seg_preds, bnd_preds, masks):
        seg_loss = self.seg_criterion(seg_preds, masks)
        bnd_targets = self.get_sobel_boundary(masks)

        bnd_loss = self.boundary_criterion(torch.sigmoid(bnd_preds), bnd_targets)
        total_loss = self.lambda_seg * seg_loss + self.lambda_bnd * bnd_loss

        return total_loss, seg_loss, bnd_loss