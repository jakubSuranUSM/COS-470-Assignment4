# COS 470 - Assignment 4
**Team members:** Jackson Thissell, Jakub Suran, Joe Rumery

This repository containes the codes and report for the Assignment 4 which explores the use of DistilRoBERTa for song genre classification 

## Instructions
* To fine-tune the DistilRoBERTa model and tune the hyperparameters, run the `fine_tune.py`, the model uses data from the `genre_lyrics.tsv` file, the fine-tuned models are going to be saved in the `models` directory, the best model will be saved there as well
* To evaluate the fine-tuned DistilRoBERTa and compare it with baseline pre-trained DistilRoBERTa model run the `evaluate.py` file, the script uses the `Test Songs` folder for test data  