
import torch
import pandas as pd
from transformers import TrainingArguments, Trainer
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from torch.utils.data import IterableDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


from transformers import TrainerCallback
import json

class LoggingCallback(TrainerCallback):
    def __init__(self, log_file):
        self.log_file = log_file

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Append logs to file after every logging step
        if logs:
            with open(self.log_file, "a") as f:
                f.write(f"Logging Step {state.global_step}:\n")
                f.write(json.dumps(logs) + "\n\n")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Append evaluation metrics to file after each evaluation step
        if metrics:
            with open(self.log_file, "a") as f:
                f.write(f"Evaluation Step {state.global_step}:\n")
                f.write(json.dumps(metrics) + "\n\n")

                
import torch
torch.cuda.empty_cache()


print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))


device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)

df_org= pd.read_csv("cleaned_train_data_language_augmented.csv", encoding='utf8')


labels = df_org['pol_spec_user'].unique().tolist()
labels = [s.strip() for s in labels ]
labels

NUM_LABELS= len(labels)

id2label={id:label for id,label in enumerate(labels)}

label2id={label:id for id,label in enumerate(labels)}
df_org["pol_spec_user"] = df_org.pol_spec_user.map(lambda x: label2id[x.strip()])

print("Acceess dataset")
# from transformers import XLMRobertaTokenizer, XLMRobertaConfig, XLMRobertaForSequenceClassification
# model_name = 'xlm-roberta-base'

# config = XLMRobertaConfig.from_pretrained(model_name)
# config = BertForSequenceClassification.config.from_pretrained("bert-base-uncased", num_labels=NUM_LABELS)
# config.output_hidden_states = False

# Load XLM-RoBERTa tokenizer
# tokenizer = XLMRobertaTokenizer.from_pretrained(model_name, config=config)

# tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", max_length=128)

# model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=NUM_LABELS, id2label=id2label, label2id=label2id)


from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-multilingual-cased", max_length=512)

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-multilingual-cased", num_labels=NUM_LABELS, id2label=id2label, label2id=label2id)
# model = XLMRobertaForSequenceClassification.from_pretrained(model_name, config=config)
model.to(device)


SIZE= df_org.shape[0]

from sklearn.model_selection import train_test_split

def create_train_val_test_splits(df, text_column, label_column, test_size=0.05, val_size=0.05, random_state=42):
    # Ensure no duplicates in the dataset
    df = df.drop_duplicates(subset=[text_column, label_column])
    
    # Split into train + val and test sets
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df[label_column], 
        random_state=random_state
    )
    
    # Adjust validation size to be relative to the train_val set
    relative_val_size = val_size / (1 - test_size)
    
    # Split the train + val set into separate train and validation sets
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=relative_val_size, 
        stratify=train_val_df[label_column], 
        random_state=random_state
    )
    
    # Extract text and labels
    train_texts = train_df[text_column].tolist()
    train_labels = train_df[label_column].tolist()
    val_texts = val_df[text_column].tolist()
    val_labels = val_df[label_column].tolist()
    test_texts = test_df[text_column].tolist()
    test_labels = test_df[label_column].tolist()
    
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels

# Example usage
train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = create_train_val_test_splits(
    df_org,
    text_column="cleaned_text",
    label_column="pol_spec_user"
)



train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings  = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)
from torch.utils.data import IterableDataset
import torch

class MyIterableDataset(IterableDataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __iter__(self):
        for idx in range(len(self.labels)):
            # Construct the batch dictionary
            batch = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}  # Tokenized data
            batch['labels'] = torch.tensor(self.labels[idx])  # Add labels
            yield batch


train_dataset = MyIterableDataset(train_encodings, train_labels)
val_dataset = MyIterableDataset(val_encodings, val_labels)
test_dataset = MyIterableDataset(test_encodings, test_labels)

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-multilingual-cased",
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id
)
# model = BertForSequenceClassification.from_pretrained(
#     "bert-base-multilingual-cased",
#     num_labels=NUM_LABELS,
#     id2label=id2label,
#     label2id=label2id
# )

model.to(device)

# Define compute_metrics function
def compute_metrics(pred):
    # Extract true labels from the input object
    labels = pred.label_ids
    
    # Obtain predicted class labels by finding the column index with the maximum probability
    preds = pred.predictions.argmax(-1)
    
    # Compute macro precision, recall, and F1 score using sklearn's precision_recall_fscore_support function
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    
    # Calculate the accuracy score using sklearn's accuracy_score function
    acc = accuracy_score(labels, preds)
    
    # Return the computed metrics as a dictionary
    return {
        'Accuracy': acc,
        'F1': f1,
        'Precision': precision,
        'Recall': recall
    }

num_train_samples = len(train_labels)
batch_size = 256
num_epochs = 3
max_steps = (num_train_samples // batch_size) * num_epochs
# Set training arguments
training_args = TrainingArguments(
    learning_rate=2e-5,
    output_dir='./TTCTrainedModel',
    num_train_epochs=num_epochs,  # Still specify for clarity
    per_device_train_batch_size=batch_size,
    warmup_steps=100,
    weight_decay=0.01,
    logging_strategy='steps',
    logging_dir='./multi-class-logs',
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    load_best_model_at_end=True,
    tf32=True,
    max_steps=max_steps  # Add this line
)


log_file = "./training_logs.txt"
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,   
    compute_metrics= compute_metrics,
    callbacks=[LoggingCallback(log_file)]
)
print("Training")
trainer.train()

trainer.save_model("./TTCTrainedModel")
tokenizer.save_pretrained("./TTCTrainedModel")

trainer.evaluate(test_dataset)