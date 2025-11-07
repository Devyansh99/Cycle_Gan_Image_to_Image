import torch
import torch.nn as nn


class StyleEncoder(nn.Module):
    """
    Style Encoder for extracting handwriting style from reference images.
    
    This is useful for production apps where you need to adapt to NEW users
    not seen during training. The user provides 3-5 reference handwriting samples,
    and this encoder extracts their unique style vector.
    
    Usage for iPad Autocomplete:
        1. User writes calibration sentences
        2. StyleEncoder extracts style vector
        3. Use style vector for all autocomplete generations
    """
    
    def __init__(self, embed_dim=128):
        """
        Args:
            embed_dim (int): Dimension of output style vector
        """
        super(StyleEncoder, self).__init__()
        self.embed_dim = embed_dim
        
        # Convolutional feature extractor
        self.conv_layers = nn.Sequential(
            # Input: [B, 3, 256, 256]
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 128, 128]
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 64, 64]
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, 32, 32]
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # [B, 512, 16, 16]
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # [B, 512, 8, 8]
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Global average pooling + FC to get style vector
        self.fc_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, 512, 1, 1]
            nn.Flatten(),  # [B, 512]
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embed_dim)  # [B, embed_dim]
        )
    
    def forward(self, reference_images):
        """
        Extract style from reference handwriting images.
        
        Args:
            reference_images (Tensor): Reference handwriting samples
                                      Shape: [B, N, C, H, W] where N = num_reference_images
                                      OR [B, C, H, W] for single reference
        
        Returns:
            Tensor: Style embedding vector, shape [B, embed_dim]
        """
        # If multiple reference images per user, average their features
        if reference_images.dim() == 5:
            B, N, C, H, W = reference_images.shape
            # Reshape: [B*N, C, H, W]
            reference_images = reference_images.view(B * N, C, H, W)
            
            # Extract features
            features = self.conv_layers(reference_images)
            style_vectors = self.fc_layers(features)  # [B*N, embed_dim]
            
            # Reshape and average: [B, N, embed_dim] -> [B, embed_dim]
            style_vectors = style_vectors.view(B, N, self.embed_dim)
            style_vector = style_vectors.mean(dim=1)
        else:
            # Single reference image per user
            features = self.conv_layers(reference_images)
            style_vector = self.fc_layers(features)
        
        return style_vector


# Example usage for iPad app:
"""
# 1. During calibration (new user)
style_encoder = StyleEncoder(embed_dim=128)
user_reference_images = collect_calibration_samples()  # [1, 5, 3, 256, 256] - 5 samples
user_style = style_encoder(user_reference_images)  # [1, 128]

# Save user_style to database/local storage
save_user_profile(user_id, user_style)

# 2. During autocomplete
user_style = load_user_profile(user_id)  # [1, 128]
next_word = gpt_predict("hello")  # GPT says next word is "world"
word_image = generator(text_to_image("world"), writer_style=user_style)
display_on_ipad(word_image)
"""
