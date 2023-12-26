from transformers import AutoTokenizer, TapasForQuestionAnswering
import torch
from torch.utils.data import DataLoader, default_collate
from generate import generate_questions
from dataset import TableDataset
from tqdm import tqdm
import warnings
from params import *

warnings.simplefilter(action='ignore', category=FutureWarning)

checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
model = TapasForQuestionAnswering.from_pretrained(PRETRAINED_MODEL)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

# generate_questions(save_file=TEST_FILE)

def collate_fn(batch):
    encodings, references = zip(*batch)
    encodings, references = list(encodings), list(references)
    return default_collate(encodings), references

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
test_iter = TableDataset(TEST_FILE, USERS_FILE, tokenizer, mode='test')
test_dataloader = DataLoader(test_iter, batch_size=1, collate_fn=collate_fn)

with torch.no_grad():
    for inputs, references in tqdm(test_dataloader):
        inputs = { k: v.to(DEVICE) for k, v in inputs.items() }

        outputs = model(**inputs)
        predicted_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(
            inputs, outputs.logits.detach(), outputs.logits_aggregation.detach()
        )

        predictions = []
        for i, coordinates in enumerate(predicted_coordinates):
            print("COORDINATES:", coordinates)
            if len(coordinates) == 1:
                # only a single cell:
                predictions.append(references[i]['table'].iat[coordinates[0]])
            else:
                # multiple cells
                cell_values = []
                for coordinate in coordinates:
                    cell_values.append(references[i]['table'].iat[coordinate])
                if predicted_aggregation_indices[i] == SUM:
                    aggregated_prediction = str(sum([float(c) for c in cell_values]))
                elif predicted_aggregation_indices[i] == COUNT:
                    aggregated_prediction = str(len(cell_values))
                else:
                    aggregated_prediction = None
                predictions.append(AGGREGATION_OPS[predicted_aggregation_indices[i]] + " > " +
                                   ", ".join(cell_values) + (" > " + aggregated_prediction if aggregated_prediction else ""))
        
        for prediction, reference in zip(predictions, references):
            print(f"Query: {reference['query']['question']}\nPrediction: {prediction}\nAnswer: {reference['query']['answer_text']}")