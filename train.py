import torch
import torch.nn as nn
from transformers import AutoTokenizer, TapasTokenizer, TapasForQuestionAnswering, TapasConfig
from torch.utils.data import DataLoader
from torch.optim import AdamW, Optimizer
from dataset import TableDataset
from tqdm import tqdm
import warnings
import wandb
import numpy as np
from params import *

def train_epoch(train_dataloader: DataLoader, model: nn.Module, optimizer: Optimizer):
    """Trains one epoch with a given training set, PyTorch model, and optimizer.

    Args:
        train_dataloader (DataLoader): PyTorch dataloader with training data
        model (nn.Module): PyTorch model to train on
        optimizer (Optimizer): PyTorch optimizer
    
    Returns:
        train_loss (float): total loss over entire training dataset
    """
    model.train()
    losses = 0

    for i, inputs in enumerate(tqdm(train_dataloader)):
        inputs = { k: v.to(DEVICE) for k, v in inputs.items() }

        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        losses += loss.item()

        if (((i + 1) % LOG_FREQUENCY == 0) or (i + 1 == len(train_dataloader))):
            wandb.log({ "loss": loss.item() })
    
    return losses / len(train_dataloader)

def evaluate(val_dataloader: DataLoader, model: nn.Module):
    """Evaluates model with a given validation set.

    Args:
        val_dataloader (DataLoader): PyTorch dataloader with validation data
        model (nn.Module): PyTorch model to evaluate
    
    Returns:
        val_loss (float): total loss over entire validation dataset
    """
    model.eval()
    losses = 0
    
    with torch.no_grad():
        for inputs in tqdm(val_dataloader):
            inputs = { k: v.to(DEVICE) for k, v in inputs.items() }

            outputs = model(**inputs)
            loss = outputs.loss

            losses += loss.item()
    
    return losses / len(val_dataloader)


warnings.simplefilter(action='ignore', category=FutureWarning)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

wandb.init(project="budgetwise")

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)

train_iter = TableDataset(DATA_FILE, USERS_FILE, tokenizer)
train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, shuffle=True)

val_iter = TableDataset(DATA_FILE, USERS_FILE, tokenizer, mode='val')
val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE)

config = TapasConfig(num_aggregation_labels=4, answer_loss_cutoff=1e5)
model = TapasForQuestionAnswering.from_pretrained(PRETRAINED_MODEL, config=config)
model.to(DEVICE)
wandb.watch(model, log_freq=LOG_FREQUENCY)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    train_loss = train_epoch(train_dataloader, model, optimizer)
    val_loss = evaluate(val_dataloader, model)
    print(f"Epoch: {epoch + 1}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}")
    wandb.log({ "val": val_loss })

    if (epoch + 1) % SAVE_FREQUENCY == 0 or (epoch + 1) == EPOCHS:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }, CHECKPOINT_FILE)
        # TODO - add drive.flush_and_unmount() to sync to drive when training on colab
