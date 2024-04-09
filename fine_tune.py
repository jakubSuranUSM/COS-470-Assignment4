"""
This file uses the 'distilbert/distilroberta-base' model to train a 
text (sequence) classification model on the genre_lyrics.tsv dataset.

Authors: Jackson Thissell, Joseph Rumery, Jakub Suran
Date: 4/8/2024
"""
from transformers import AutoTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer
from datasets import DatasetDict, Dataset
import evaluate
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# data loading and preprocessing
data = pd.read_csv('genre_lyrics.tsv', delimiter='\t')
data = data[data['Lyrics'].str.split().str.len() >= 10] # remove songs with less than 10 words

# balance the dataset
genre_counts = data['Genre'].value_counts()
min_genre_count = genre_counts.min()
balanced_data = data.groupby('Genre')[data.columns].apply(lambda x: x.sample(min_genre_count))
balanced_data.reset_index(drop=True, inplace=True)
data = balanced_data

data.rename(columns={'Genre': 'label', 'Lyrics': 'text'}, inplace=True)

# split the data into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.2, stratify=data['label'], random_state=42)

NUM_LABELS = 6
model_name = 'distilbert/distilroberta-base'

# load the tokenizer and metric
tokenizer = AutoTokenizer.from_pretrained(model_name)
metric = evaluate.load("f1")

# tokenize the data
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# create the train and validation datasets
tds = Dataset.from_pandas(train_data)
vds = Dataset.from_pandas(val_data)

ds = DatasetDict()
ds['train'] = tds
ds['validation'] = vds

# encode the labels
ds = ds.class_encode_column("label")

tokenized_datasets = ds.map(tokenize_function, batched=True)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42)
small_eval_dataset = tokenized_datasets["validation"].shuffle(seed=42)

# function to compute the macro f1 score
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average="macro")


def train_for_hyperparameter(learning_rate, batch_size, adam_beta1, filename):
    """
    Trains a model for a specific set of hyperparameters and evaluates the
    model on the validation set.

    Args:
        learning_rate (float): The learning rate for the optimizer.
        batch_size (int): The batch size for training and evaluation.
        adam_beta1 (float): The beta1 value for the Adam optimizer.
        filename (str): The name of the file to save the trained model.

    Returns:
        float: The F1 score of the trained model.

    """
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=NUM_LABELS)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        adam_beta1=adam_beta1,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )

    print("Training time!")
    print(f"Learning rate: {learning_rate}, Batch size: {batch_size}, Adam beta1: {adam_beta1}")
    trainer.train()
    
    metrics = trainer.evaluate()
    f1 = metrics['eval_f1']
    print(f"F1 score: {f1}")
    
    
    model.save_pretrained(f"./models2/{filename}_f1_{f1}") 
    
    print("--------------------------------------------------")
    return f1


# hyperparameter search
learning_rates = [1e-5, 5e-5, 1e-4]
batch_sizes = [4, 8, 16]
adam_beta1s = [0.85, 0.9, 0.99]

# initialize best hyperparameters
best_f1 = 0
best_learning_rate = learning_rates[0]
best_batch_size = batch_sizes[0]
best_adam_beta1 = adam_beta1s[0]

# search for the best batch size
for i, batch_size in enumerate(batch_sizes):
    print(f"Finding best batch size: {i+1}/{len(batch_sizes)}")
    lr, bs, ab = best_learning_rate, batch_size, best_adam_beta1
    f1 = train_for_hyperparameter(lr, bs, ab, f"lr{lr}_bs{bs}_ab{ab}")
    if f1 > best_f1:
        best_f1 = f1
        best_batch_size = batch_size

# search for the best learning rate
for i, learning_rate in enumerate(learning_rates):
    print(f"Finding best learning rate: {i+1}/{len(learning_rates)}")
    lr, bs, ab = learning_rate, best_batch_size, best_adam_beta1
    f1 = train_for_hyperparameter(lr, bs, ab, f"lr{lr}_bs{bs}_ab{ab}")
    if f1 > best_f1:
        best_f1 = f1
        best_learning_rate = learning_rate
    
# search for the best Adam beta1
for i, adam_beta1 in enumerate(adam_beta1s):
    print(f"Finding best Adam Beta 1: {i+1}/{len(adam_beta1s)}")
    lr, bs, ab = best_learning_rate, best_batch_size, adam_beta1
    f1 = train_for_hyperparameter(lr, bs, ab, f"lr{lr}_bs{bs}_ab{ab}")
    if f1 > best_f1:
        best_f1 = f1
        best_adam_beta1 = adam_beta1
    
# print the best hyperparameters
print("--------------------------------------------------")
print("Best hyperparameters:")
print(f"Learning rate: {best_learning_rate}, Batch size: {best_batch_size}, Adam beta1: {best_adam_beta1}")
print(f"F1 score: {best_f1}")
print("--------------------------------------------------")

# train and save the model with the best hyperparameters
train_for_hyperparameter(best_learning_rate, best_batch_size, best_adam_beta1, "best_model")