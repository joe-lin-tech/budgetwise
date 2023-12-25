from transformers import AutoTokenizer, TapasForQuestionAnswering
import torch
import pandas as pd
from params import *

checkpoint = torch.load(CHECKPOINT_FILE)
model = TapasForQuestionAnswering(PRETRAINED_MODEL)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)

model.eval()

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)