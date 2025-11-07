import torch
import torch.nn as nn


class WriterEmbedding(nn.Module):
    """
    Writer Embedding Module for Writer-Aware CycleGAN.
    
    Maps writer IDs to continuous style embedding vectors that represent
    the unique handwriting characteristics of each writer.
    
    Args:
        num_writers (int): Total number of unique writers in the dataset
        embed_dim (int): Dimension of the embedding vector (default: 128)
        use_fc (bool): Whether to use additional FC layer after embedding (default: True)
    """
    
    def __init__(self, num_writers, embed_dim=128, use_fc=True):
        super(WriterEmbedding, self).__init__()
        self.num_writers = num_writers
        self.embed_dim = embed_dim
        self.use_fc = use_fc
        
        # Embedding layer: maps writer ID (integer) to a dense vector
        self.embedding = nn.Embedding(num_writers, embed_dim)
        
        # Optional FC layer for additional transformation
        if use_fc:
            self.fc = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, embed_dim)
            )
        
        # Initialize embeddings with normal distribution
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
    
    def forward(self, writer_ids):
        """
        Forward pass.
        
        Args:
            writer_ids (Tensor): Batch of writer IDs, shape [batch_size]
        
        Returns:
            Tensor: Writer style embeddings, shape [batch_size, embed_dim]
        """
        # Get embedding vectors for the batch of writer IDs
        style_embedding = self.embedding(writer_ids)
        
        # Apply additional transformation if FC is enabled
        if self.use_fc:
            style_embedding = self.fc(style_embedding)
        
        return style_embedding
