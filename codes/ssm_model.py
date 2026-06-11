"""
Module: ssm_model
Description: Defines the Neural Network architecture and inference generation 
logic for the Bach Chorale model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BalancedMusicClassificationLSTM(nn.Module):
    """
    A lightweight, robust LSTM designed for 4-part polyphonic music generation.
    Features input embedding, deep sequential layers, and layer normalization.
    """
    def __init__(self, input_size: int = 4, hidden_size: int = 256, num_layers: int = 4, dropout: float = 0.2):
        super().__init__()
        
        # 1. Input Projection: Translates raw integers into relationships
        self.input_embed = nn.Linear(input_size, 64)
        
        # 2. Deep Sequential Core
        self.lstm = nn.LSTM(
            input_size=64,           
            hidden_size=hidden_size, 
            num_layers=num_layers,   
            batch_first=True,
            dropout=dropout
        )
        
        # 3. Normalization & Projection Head
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc_hidden = nn.Linear(hidden_size, 256)
        
        # 4. Output: 128 possible pitch classes for each of the 4 voices
        self.fc_out = nn.Linear(256, 4 * 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = F.relu(self.input_embed(x)) 
        out, _ = self.lstm(embedded) 
        
        out = self.layer_norm(out)
        out = F.gelu(self.fc_hidden(out)) 
        out = self.fc_out(out)
        
        # Reshape to (batch_size, seq_len, 4_voices, 128_classes)
        return out.view(out.size(0), out.size(1), 4, 128)


def generate_sequence(model: nn.Module, seed: torch.Tensor, steps: int = 150, temperature: float = 1.0, device: str = "cpu") -> np.ndarray:
    """
    Auto-regressively generates a multi-track musical sequence from a seed.

    Args:
        model (nn.Module): The trained PyTorch model.
        seed (torch.Tensor): The initial normalized sequence block (Seq_len, 4).
        steps (int): Number of time steps to generate.
        temperature (float): Entropy scale. Higher = more creative/random.
        device (str): Compute device ("cpu" or "cuda").

    Returns:
        np.ndarray: The finalized, un-normalized integer pitch array.
    """
    model.eval()
    with torch.no_grad():
        result = seed.clone().to(device) 

        for _ in range(steps):
            # Take the last 64 steps as context
            inp = result[-64:].unsqueeze(0)
            pred = model(inp)

            # Isolate the prediction for the absolute final step
            last_step_logits = pred[0, -1] 
            next_step_notes = []
            
            for voice_idx in range(4):
                voice_logits = last_step_logits[voice_idx] / temperature
                probs = torch.softmax(voice_logits, dim=-1)
                
                # Sample from the probability distribution
                sampled_note = torch.multinomial(probs, 1).item()
                next_step_notes.append(sampled_note)

            # Normalize back to [0, 1] and append to the timeline
            next_step_tensor = torch.tensor(next_step_notes).float().to(device) / 127.0
            result = torch.cat([result, next_step_tensor.unsqueeze(0)], dim=0)

    # Decode floats back to standard MIDI integers (0-127)
    result = result * 127
    return result.detach().cpu().numpy().round().astype(int)