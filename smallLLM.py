#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, pipeline

# --- 1. Load and Prepare the Dataset from the provided JSON file ---
# The data is already in a perfect Q&A format. We'll load it and
# convert it into a Hugging Face Dataset object.
with open('constitution_qa.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)

print("Dataset loaded and converted to Hugging Face format:")
print(dataset)
print("\nSample data point:")
print(dataset[0])

# --- 2. Load the Model and Tokenizer ---
# We'll use a pre-trained model suitable for question-answering.
# DistilBERT is a good, small option for a project like this.
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# --- 3. Preprocess the Data for the Model ---
# This function tokenizes the questions and answers and prepares them
# in a format the model expects.
def tokenize_function(examples):
    # This is a simplified example. For full fine-tuning on QA,
    # you would need to handle tokenizing contexts and start/end positions.
    # For a simple Q&A dataset like this, we can tokenize the Q and A.
    return tokenizer(
        examples['question'],
        examples['answer'],
        truncation=True,
        padding="max_length"
    )

# Apply the tokenization function to the entire dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# You may also want to split your dataset into training and evaluation sets
# train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
# train_dataset = train_test_split['train']
# eval_dataset = train_test_split['test']

# --- 4. Define Training Arguments and Fine-tune the Model ---
# This sets up the training process. You can adjust these parameters.
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    report_to="none" # Disable logging to external services
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset, # Use the full dataset for this example
    # eval_dataset=eval_dataset
)

# Uncommented: This line starts the training process.
print("\nStarting model training...")
trainer.train()

# --- 5. Example of Using the Fine-tuned Model for a New Query ---
# After training, you can save the model and load it for inference.
# For demonstration purposes, we will use the pre-trained model here.
# In a real project, you would save the fine-tuned model and then load that.
# Uncommented: These lines save the fine-tuned model and tokenizer.
model.save_pretrained("./my_fine_tuned_model")
tokenizer.save_pretrained("./my_fine_tuned_model")

# Use a Hugging Face pipeline for easy inference
loaded_model_path = "./my_fine_tuned_model"
qa_pipeline = pipeline("question-answering", model=loaded_model_path, tokenizer=loaded_model_path)

new_query = "What is the purpose of the industries declared by Parliament by law to be necessary?"
context = "Industries necessary for the purpose of defence or for the prosecution of war. What is the purpose of the industries declared by Parliament by law to be necessary?"

result = qa_pipeline(question=new_query, context=context)

print("\nExample Inference:")
print(f"Question: {new_query}")
print(f"Answer: {result['answer']}")
print(f"Confidence Score: {result['score']:.4f}")

