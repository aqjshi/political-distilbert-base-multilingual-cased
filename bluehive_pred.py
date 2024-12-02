import pandas as pd
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import fasttext
from tqdm import tqdm
import re

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load test data
test_df = pd.read_csv('cleaned_test_language_augmented.csv', encoding='utf8')

# Load model and tokenizer from checkpoint
model_path = "./TTCTrainedModel"
model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

# Prediction function
def predict(texts, tokenizer, model):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=1).detach().cpu().numpy()
    pred_label_idxs = probs.argmax(axis=1)
    pred_labels = [model.config.id2label[idx] for idx in pred_label_idxs]
    return probs, pred_label_idxs, pred_labels


# Prepare results
results = []
for idx, row in tqdm(test_df.iterrows(), total=test_df.shape[0], desc="Processing predictions"):
    probs, pred_label_idx, pred_label = predict(row['cleaned_text'], tokenizer, model)
    results.append({
        'Id': row['Id'],  # Assuming 'id' column exists in the test data
        'pol_spec_user': pred_label,
        'probs': probs.tolist(),
        'pred_label': pred_label
    })

# Save predictions for submission
submission_df = pd.DataFrame(results)[['Id', 'pol_spec_user']]
submission_df.to_csv('predictions.csv', index=False)

# Save detailed predictions for analysis
analysis_df = pd.DataFrame(results)
analysis_df.to_csv('predictions_analysis.csv', index=False)

print("Predictions saved to 'predictions.csv' and detailed analysis to 'predictions_analysis.csv'.")
