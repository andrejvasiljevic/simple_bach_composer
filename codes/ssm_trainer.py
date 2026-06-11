"""
Module: ssm_trainer
Description: Handles the PyTorch training loop, loss calculation, and checkpoint 
saving for the Bach Chorale LSTM model.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np

def train_chorale_model(
    model: nn.Module, 
    dataloader: DataLoader, 
    epochs: int = 50, 
    learning_rate: float = 0.001, 
    device: str = "cpu", 
    save_dir: str = "../models_classification_balanced/",
    start_epoch: int = 1  
):
    """
    Trains the LSTM model on the music dataset and saves epoch checkpoints.
    
    Args:
        model (nn.Module): The PyTorch LSTM architecture.
        dataloader (DataLoader): PyTorch DataLoader containing (X, y) sequences.
        epochs (int): Total number of training loops.
        learning_rate (float): Step size for the Adam optimizer.
        device (str): "cuda" for GPU or "cpu".
        save_dir (str): Directory path to save the .pth checkpoint files.
    """
    model.to(device)
    model.train()
    
    # CrossEntropyLoss expects unnormalized logits and class integer indices
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Ensure our save folder exists
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"    Starting training on {device.upper()} for {epochs} epochs...")

    for epoch in range(start_epoch, start_epoch + epochs): 
        total_loss = 0.0
        
        # tqdm gives us a beautiful progress bar for the batches!
        batch_iterator = tqdm(dataloader, desc=f"Epoch {epoch:02d}/{epochs}")
        
        for batch_x, batch_y in batch_iterator:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # 1. Reset gradients
            optimizer.zero_grad()
            
            # 2. Forward Pass: (batch, seq_len, 4_voices, 128_classes)
            outputs = model(batch_x)
            
            # 3. Reshape for Loss Calculation
            # CrossEntropyLoss requires 2D predictions (N, Classes) and 1D targets (N)
            # We flatten everything except the 128 pitch classes
            outputs_flat = outputs.view(-1, 128)
            targets_flat = batch_y.view(-1).long() # Ensure targets are integers
            
            # 4. Calculate Loss
            loss = criterion(outputs_flat, targets_flat)
            
            # 5. Backward Pass (Calculate gradients & update weights)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update the progress bar with the real-time loss
            batch_iterator.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        # --- Epoch Metrics & Checkpointing ---
        avg_loss = total_loss / len(dataloader)
        confidence = (1.0 / np.exp(avg_loss)) * 100
        
        print(f"🏁 Epoch {epoch:02d} | Average Loss: {avg_loss:.4f} | Choice Confidence: {confidence:.2f}%")
        
        # Save the checkpoint model
        save_path = os.path.join(save_dir, f"music_classification_lstm_{epoch}.pth")
        torch.save(model.state_dict(), save_path)
        
    print("✨ Training Complete!")