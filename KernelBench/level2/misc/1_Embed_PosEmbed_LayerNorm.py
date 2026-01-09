import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Fused Embed + Position Embed + LayerNorm
    
    Used by: BERT, RoBERTa, encoder models
    
    Token embed + position embed + LayerNorm (BERT-style input).
    """
    
    def __init__(self, vocab_size: int, max_seq_len: int, hidden_size: int):
        super(Model, self).__init__()
        self.token_embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        pos_ids = torch.arange(seq_len, device=input_ids.device)
        x = self.token_embed(input_ids) + self.pos_embed(pos_ids)
        return self.norm(x)


batch_size, seq_len, vocab_size, hidden_size = 32, 512, 30522, 768

def get_inputs():
    return [torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda')]

def get_init_inputs():
    return [vocab_size, seq_len, hidden_size]

