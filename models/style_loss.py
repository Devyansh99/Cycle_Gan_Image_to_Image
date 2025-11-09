"""Simple Style Loss for handwriting style transfer"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleLoss(nn.Module):
    """Lightweight style loss using texture/pattern matching"""
    
    def __init__(self):
        super(StyleLoss, self).__init__()
        
    def compute_loss(self, generated, style_reference):
        """
        Compute style loss between generated and style reference
        Uses simple L2 distance in pixel space for texture matching
        
        Args:
            generated: Generated image tensor [B, C, H, W]
            style_reference: Style reference image [B, C, H, W]
        
        Returns:
            style_loss: Scalar loss value
        """
        # Simple pixel-level texture loss
        # Encourage generated to have similar pixel distributions as reference
        loss = F.l1_loss(generated, style_reference)
        
        return loss
