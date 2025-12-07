import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    CTC (Connectionist Temporal Classification) Decoder for OCR.
    
    Enables sequence-to-sequence learning without explicit alignment.
    Used in HunyuanOCR, PaddleOCR, and text recognition systems.
    
    Based on: "Connectionist Temporal Classification" and modern OCR systems
    """
    def __init__(self, input_dim, hidden_dim, vocab_size, num_layers=2, bidirectional=True):
        """
        :param input_dim: Input feature dimension
        :param hidden_dim: LSTM hidden dimension
        :param vocab_size: Size of character vocabulary
        :param num_layers: Number of LSTM layers
        :param bidirectional: Whether to use bidirectional LSTM
        """
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.bidirectional = bidirectional
        
        # Sequence modeling (BiLSTM or Transformer)
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Projection to vocabulary (including blank token at index 0)
        self.fc = nn.Linear(lstm_output_dim, vocab_size + 1)  # +1 for CTC blank
        
        # Optional attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, features, feature_lengths=None):
        """
        Compute CTC logits for text recognition.
        
        :param features: Image features (batch, seq_len, input_dim)
        :param feature_lengths: Actual lengths of each sequence (batch,)
        :return: Dict with 'logits' and 'log_probs'
        """
        batch_size, seq_len, _ = features.shape
        
        # Pack sequences if lengths provided
        if feature_lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                features, feature_lengths.cpu(), 
                batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True, total_length=seq_len
            )
        else:
            lstm_out, _ = self.lstm(features)
        
        # Project to vocabulary logits
        logits = self.fc(lstm_out)  # (batch, seq_len, vocab_size + 1)
        
        # Log softmax for CTC loss
        log_probs = F.log_softmax(logits, dim=-1)
        
        return {
            'logits': logits,
            'log_probs': log_probs
        }
    
    def decode_greedy(self, log_probs):
        """
        Greedy CTC decoding.
        
        :param log_probs: Log probabilities (batch, seq_len, vocab_size + 1)
        :return: List of decoded sequences (without blanks and repeated chars)
        """
        # Get argmax predictions
        preds = log_probs.argmax(dim=-1)  # (batch, seq_len)
        
        decoded = []
        for pred in preds:
            # Remove consecutive duplicates and blanks
            chars = []
            prev_char = -1
            for char in pred:
                char = char.item()
                if char != 0 and char != prev_char:  # 0 is blank
                    chars.append(char)
                prev_char = char
            decoded.append(chars)
        
        return decoded
    
    def decode_beam_search(self, log_probs, beam_width=10):
        """
        Beam search CTC decoding.
        
        :param log_probs: Log probabilities (batch, seq_len, vocab_size + 1)
        :param beam_width: Beam width
        :return: List of decoded sequences
        """
        batch_size, seq_len, vocab_size = log_probs.shape
        
        decoded = []
        for b in range(batch_size):
            # Initialize beams: (prefix, log_prob_blank, log_prob_non_blank)
            beams = [('', 0.0, float('-inf'))]
            
            for t in range(seq_len):
                new_beams = {}
                
                for prefix, log_pb, log_pnb in beams:
                    # Add blank
                    new_key = prefix
                    log_p = log_probs[b, t, 0].item()
                    
                    if new_key in new_beams:
                        new_pb, new_pnb = new_beams[new_key]
                        new_beams[new_key] = (
                            torch.logsumexp(torch.tensor([new_pb, log_pb + log_p, log_pnb + log_p]), dim=0).item(),
                            new_pnb
                        )
                    else:
                        new_beams[new_key] = (log_pb + log_p if log_pb > float('-inf') else log_pnb + log_p, float('-inf'))
                    
                    # Add characters
                    for c in range(1, vocab_size):
                        char = chr(c + 31)  # Simple mapping
                        new_prefix = prefix + char
                        log_p = log_probs[b, t, c].item()
                        
                        if len(prefix) > 0 and prefix[-1] == char:
                            # Same character: only from blank
                            new_log_p = log_pb + log_p
                        else:
                            # Different character: from both
                            new_log_p = torch.logsumexp(torch.tensor([log_pb + log_p, log_pnb + log_p]), dim=0).item()
                        
                        if new_prefix in new_beams:
                            old_pb, old_pnb = new_beams[new_prefix]
                            new_beams[new_prefix] = (old_pb, torch.logsumexp(torch.tensor([old_pnb, new_log_p]), dim=0).item())
                        else:
                            new_beams[new_prefix] = (float('-inf'), new_log_p)
                
                # Prune to beam_width
                sorted_beams = sorted(
                    [(k, pb, pnb) for k, (pb, pnb) in new_beams.items()],
                    key=lambda x: torch.logsumexp(torch.tensor([x[1], x[2]]), dim=0).item(),
                    reverse=True
                )[:beam_width]
                beams = sorted_beams
            
            # Get best beam
            if beams:
                best = max(beams, key=lambda x: torch.logsumexp(torch.tensor([x[1], x[2]]), dim=0).item())
                decoded.append(best[0])
            else:
                decoded.append('')
        
        return decoded


# Test parameters
batch_size = 8
seq_len = 64  # Feature sequence length
input_dim = 512
hidden_dim = 256
vocab_size = 5000  # Character vocabulary

def get_inputs():
    features = torch.randn(batch_size, seq_len, input_dim)
    return [features]

def get_init_inputs():
    return [input_dim, hidden_dim, vocab_size]

