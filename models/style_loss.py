"""Simple Style Loss for handwriting style transfer"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleLoss(nn.Module):
    """Lightweight style loss using Gram matrices"""
    
    def __init__(self):
        super(StyleLoss, self).__init__()
        
    def gram_matrix(self, input):
        """Compute Gram matrix for style representation"""
        b, c, h, w = input.size()
        features = input.view(b, c, h * w)
        G = torch.bmm(features, features.transpose(1, 2))
        return G.div(c * h * w)
    
    def compute_loss(self, generated, style_reference):
        """
        Compute style loss between generated and style reference
        
        Args:
            generated: Generated image tensor [B, C, H, W]
            style_reference: Style reference image [B, C, H, W]
        
        Returns:
            style_loss: Scalar loss value
        """
        # Simple feature extraction using input directly
        # Compute Gram matrices
        gram_generated = self.gram_matrix(generated)
        gram_style = self.gram_matrix(style_reference)
        
        # L2 loss between Gram matrices
        loss = F.mse_loss(gram_generated, gram_style)
        
        return loss
