import torch

SEED = 2023
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 5e-5
DEVICE = torch.device('cpu')
LOG_FREQUENCY = 2
SAVE_FREQUENCY = 1
PRETRAINED_MODEL = 'google/tapas-base-finetuned-wtq'
DATA_FILE = 'data.csv'
TEST_FILE = 'test.csv'
USERS_FILE = 'users.csv'
CHECKPOINT_FILE = 'model.pt' # TODO - change to /content/drive/My Drive/model.pt for training on colab

CATEGORIES = ['agricultural', 'contracted', 'transportation', 'utility', 'retail', 'clothing',
              'misc', 'business', 'government', 'airlines', 'lodging', 'professional']
SUM, COUNT, AVERAGE, ARGMAX_SUM, ARGMIN_SUM, ARGMAX_STD, ARGMIN_STD, NONE = 0, 1, 2, 3, 4, 5, 6, 7