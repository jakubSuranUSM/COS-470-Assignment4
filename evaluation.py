"""
This file loads the Test Songs dataset and evaluates the best model trained in main.py

Authors: Jackson Thissell, Joseph Rumery, Jakub Suran
Date: 4/8/2024
"""
import random
from transformers import TextClassificationPipeline, AutoTokenizer, RobertaForSequenceClassification
from datasets import Dataset
import evaluate
import pandas as pd
import os

folder_path = "./Test Songs"
data = []

# Load the test songs
for genre in os.listdir(folder_path):
    genre_path = os.path.join(folder_path, genre)
    for file in os.listdir(genre_path):
        file_path = os.path.join(genre_path, file)
        with open(file_path, "r") as f:
            lyrics = f.read()
        data.append({"genre": genre, "lyrics": lyrics, "title": file.split(".")[0]})

df = pd.DataFrame(data)
# Map the genre to the label used in training
genre_mapping = {
    "Rap": "HipHop",
    "Pop": "Pop",
    "Country": "Country",
    "Blues": "Blues",
    "Metal": "HeavyMetal",
    "Rock": "RockandRoll"
}
df["genre"] = df["genre"].map(genre_mapping)

df.rename(columns={'genre': 'label', 'lyrics': 'text'}, inplace=True)

# Load the tokenizer
model_name = 'distilbert/distilroberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the data
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Create the dataset and encode the labels
tds = Dataset.from_pandas(df)
tds = tds.class_encode_column("label")

# Load the fine-tuned model and the pretrained model
model_path = "./models/lr5e-05_bs4_ab0.85_f1_0.702418983035412/"
model_finetuned = RobertaForSequenceClassification.from_pretrained(model_path)
model_pretrained = RobertaForSequenceClassification.from_pretrained(model_name)

# Create the evaluation pipeline
pipe_finetuned = TextClassificationPipeline(model=model_finetuned, tokenizer=tokenizer)
pipe_pretrained = TextClassificationPipeline(model=model_pretrained, tokenizer=tokenizer)

# get the predictions
predictions_finetuned = pipe_finetuned([p[:512] for p in tds['text']])
predictions_pretrained = pipe_pretrained([p[:512] for p in tds['text']])
 
# get the prediction labels
predictions_finetuned = [int(p['label'].split('_')[-1]) for p in predictions_finetuned]
predictions_pretrained = [int(p['label'].split('_')[-1]) for p in predictions_pretrained]

# Compute the F1 score
eval = evaluate.load("f1")
f1_finetuned = eval.compute(predictions=predictions_finetuned, references=tds['label'], average="macro")
f1_pretrained = eval.compute(predictions=predictions_pretrained, references=tds['label'], average="macro")

# Print the F1 score
print("--------------------------------------------")
print(f"Evaluation of the fine-tuned model on the Test Songs dataset:")
print("F1 Score:", round(f1_finetuned['f1'] * 100, 2))
print("--------------------------------------------")
print(f"Evaluation of the pretrained model on the Test Songs dataset:")
print("F1 Score:", round(f1_pretrained['f1'] * 100, 2))
print("--------------------------------------------")

# Pick elements where fine-tuned model succeeded and pretrained model failed
matching_indices = [i for i in range(len(predictions_finetuned)) if predictions_finetuned[i] == tds['label'][i] and predictions_finetuned[i] != predictions_pretrained[i]]

print("\n\nExamples where the fine-tuned model succeeded and the pretrained model failed:")
print("--------------------------------------------")

id2label = {id:tds.features['label'].int2str(id) for id in range(6)}
for i in range(3):
    element = random.choice(matching_indices)
    print(f"Title: {tds['title'][element]}")
    print(f"Genre: {id2label[tds['label'][element]]}")
    print(f"Fine-tuned Prediction: {id2label[predictions_finetuned[element]]}")
    print(f"Pretrained Prediction: {id2label[predictions_pretrained[element]]}")
    print("--------------------------------------------")
    
# Pick elements where fine-tuned model failed and pretrained model succeeded
matching_indices = [i for i in range(len(predictions_finetuned)) if predictions_pretrained[i] == tds['label'][i] and predictions_finetuned[i] != predictions_pretrained[i]]

print("\n\nExamples where the fine-tuned model failed and the pretrained model succeeded:")
print("--------------------------------------------")

id2label = {id:tds.features['label'].int2str(id) for id in range(6)}
if len(matching_indices) > 0:
    for i in range(min(len(matching_indices), 3)):
        element = random.choice(matching_indices)
        print(f"Title: {tds['title'][element]}")
        print(f"Genre: {id2label[tds['label'][element]]}")
        print(f"Fine-tuned Prediction: {id2label[predictions_finetuned[element]]}")
        print(f"Pretrained Prediction: {id2label[predictions_pretrained[element]]}")
        print("--------------------------------------------")
    
