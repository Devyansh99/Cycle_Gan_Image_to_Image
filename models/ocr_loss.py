import torch
import torch.nn.functional as F
import numpy as np


class OCRLoss:
    """
    OCR Consistency Loss for Writer-Aware CycleGAN.
    
    This loss ensures that generated images preserve text readability
    by comparing OCR-extracted features from fake and real images.
    
    Note: EasyOCR is computationally expensive, so we use a lightweight approach
    by extracting text lengths as a proxy for readability.
    """
    
    def __init__(self, device='cuda', lang_list=['en']):
        """
        Initialize OCR Loss.
        
        Args:
            device (str): 'cuda' or 'cpu'
            lang_list (list): List of languages for OCR
        """
        self.device = device
        try:
            import easyocr
            self.reader = easyocr.Reader(lang_list, gpu=(device == 'cuda'), verbose=False)
            self.enabled = True
            print(f"EasyOCR initialized successfully on {device}")
        except ImportError:
            print("Warning: EasyOCR not installed. OCR loss will be disabled.")
            print("Install with: pip install easyocr")
            self.enabled = False
        except Exception as e:
            print(f"Warning: Failed to initialize EasyOCR: {e}")
            print("OCR loss will be disabled.")
            self.enabled = False
    
    def extract_text_features(self, image_tensor):
        """
        Extract text features from image tensor.
        
        Args:
            image_tensor (Tensor): Image tensor, shape [B, C, H, W], range [-1, 1]
        
        Returns:
            Tensor: Text feature vector (e.g., text length, character count)
        """
        if not self.enabled:
            return torch.zeros(image_tensor.shape[0], device=self.device)
        
        batch_size = image_tensor.shape[0]
        text_lengths = []
        
        for i in range(batch_size):
            # Convert tensor to numpy image (denormalize from [-1,1] to [0,255])
            img = image_tensor[i].detach().cpu()
            img = (img + 1.0) / 2.0  # [-1, 1] -> [0, 1]
            img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            
            try:
                # Run OCR
                result = self.reader.readtext(img, detail=0)
                # Calculate total character count as a simple readability metric
                total_chars = sum(len(text) for text in result)
                text_lengths.append(total_chars)
            except Exception:
                # If OCR fails, use 0
                text_lengths.append(0)
        
        return torch.tensor(text_lengths, dtype=torch.float32, device=self.device)
    
    def compute_loss(self, fake_images, real_images):
        """
        Compute OCR consistency loss.
        
        Args:
            fake_images (Tensor): Generated images, shape [B, C, H, W]
            real_images (Tensor): Real images, shape [B, C, H, W]
        
        Returns:
            Tensor: OCR loss (scalar)
        """
        if not self.enabled:
            return torch.tensor(0.0, device=self.device)
        
        # Extract text features
        fake_features = self.extract_text_features(fake_images)
        real_features = self.extract_text_features(real_images)
        
        # Compute L1 loss between feature vectors
        # Normalize by character count to make loss scale-invariant
        max_chars = torch.max(torch.cat([fake_features, real_features])) + 1e-6
        fake_norm = fake_features / max_chars
        real_norm = real_features / max_chars
        
        loss = F.l1_loss(fake_norm, real_norm)
        
        return loss
