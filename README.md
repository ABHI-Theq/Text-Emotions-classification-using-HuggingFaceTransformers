🎯 Text Emotions Detection using HuggingFace Transformers

📌 Overview

This project implements a text emotion detection system using HuggingFace Transformers. The goal is to classify the emotions expressed in a given piece of text using state-of-the-art Natural Language Processing (NLP) models. The notebook leverages the transformers and datasets libraries from HuggingFace and evaluates multiple models for performance comparison.

📂 Dataset

Dataset Used: emotion dataset from HuggingFace Datasets

Labels: 😢 sadness, 😊 joy, ❤️ love, 😡 anger, 😨 fear, 😲 surprise

Structure: Each example in the dataset contains a piece of text and its corresponding emotion label.

🧰 Key Libraries Used

transformers

datasets

sklearn

matplotlib

pandas

seaborn

torch

numpy

🔄 Steps Implemented

1️⃣ Dataset Loading and Exploration

Used load_dataset("emotion") to fetch the dataset.

Explored distribution of emotion labels.

2️⃣ Tokenization

Tokenized the text data using different pretrained models:

distilbert-base-uncased

bert-base-uncased

bert-base-cased

roberta-base

Applied truncation to handle long texts.

3️⃣ Data Preprocessing

Encoded dataset using tokenizers.

Formatted dataset to return PyTorch tensors.

4️⃣ Model Training

Trained models using Trainer and TrainingArguments from transformers.

Parameters included:

Batch sizes

Learning rate scheduler

Number of epochs

Evaluation strategy

5️⃣ Evaluation

Metrics used: accuracy, precision, recall, and F1-score

Visualized:

Confusion matrix

Classification report

Compared performance of different models

6️⃣ Model Testing

Performed inference on test samples

Evaluated predictions vs ground truth labels

7️⃣ Saving and Loading Models

Saved fine-tuned models using .save_pretrained()

Reloaded them with from_pretrained()

8️⃣ Plotting

Visualized label distribution and confusion matrix using matplotlib and seaborn

🏆 Model Performance Summary

Best Performing Model: BERT-base-uncased

Highlights:

High precision and recall for major classes like joy and sadness

Slightly lower performance for minority classes like surprise and love

⚙️ How to Run

Install dependencies:

pip install transformers datasets sklearn matplotlib pandas seaborn torch

Run the notebook: Text_Emotions_detection_using_huggingFaceTransformers.ipynb

✅ Conclusion

This notebook demonstrates the effectiveness of transformer-based models for multi-class emotion detection. Fine-tuning pretrained models like BERT significantly improves classification accuracy compared to traditional methods.

👨‍💻 Author: Abhishek Sharma🌐 GitHub: ABHI-Theq🗓️ Last Updated: (Add Date)

