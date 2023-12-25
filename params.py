import torch

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 5e-5
DEVICE = torch.device('cpu')
LOG_FREQUENCY = 2
SAVE_FREQUENCY = 1
PRETRAINED_MODEL = 'google/tapas-base-finetuned-wtq'
DATA_FILE = 'data.csv'
USERS_FILE = 'users.csv'
CHECKPOINT_FILE = 'model.pt'