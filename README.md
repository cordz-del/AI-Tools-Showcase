# AI-Tools-Showcase
A showcase of various AI tools and their applications, including code examples and usage scenarios.

## Introduction

This repository showcases a variety of AI tools and their applications. It includes code examples and usage scenarios that demonstrate how these tools can be utilized in different contexts. The aim is to provide a comprehensive overview of the capabilities and functionalities of each tool.

## AI Tools

### OpenAI GPT
OpenAI's GPT models are powerful language models capable of generating human-like text. They can be used for a variety of applications, including text completion, translation, and summarization.

### Anthropic Claude
Anthropic's Claude is an AI assistant designed to understand and generate natural language. It can assist with tasks such as answering questions, providing explanations, and engaging in conversation.

### Meta's LLaMA
Meta's LLaMA is a large language model that excels in natural language understanding and generation. It can be used for tasks like text classification, sentiment analysis, and language translation.

### Google PaLM / Gemini
Google's PaLM and Gemini are advanced AI models designed for natural language processing. They are capable of understanding context and generating coherent text in various languages.

### Hugging Face Transformers
Hugging Face Transformers is a library that provides access to a wide range of pre-trained models for natural language processing, including BERT, GPT-2, and T5. It simplifies the process of fine-tuning and deploying these models.

## Code Examples

### OpenAI GPT Text Generation Example
```python
import openai

# Set up the OpenAI API client
openai.api_key = "your-api-key"

# Define a prompt for text generation
prompt = "Once upon a time"

# Generate text using GPT-3
response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=prompt,
    max_tokens=50
)

# Print the generated text
print(response.choices[0].text.strip())
```
This example demonstrates how to use OpenAI GPT-3 for text generation by providing a prompt and generating a continuation of the text.

### Anthropic Claude Interaction Example
```python
import anthropic

# Set up the Anthropic API client
anthropic.api_key = "your-api-key"

# Define a question for Claude
question = "What is the capital of France?"

# Get a response from Claude
response = anthropic.Completion.create(
    model="claude-v1",
    prompt=question,
    max_tokens=50
)

# Print the response from Claude
print(response.choices[0].text.strip())
```
This example demonstrates how to interact with Anthropic's Claude to get answers to questions or engage in conversation.

### Meta's LLaMA Text Classification Example
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the tokenizer and model for Meta's LLaMA
tokenizer = AutoTokenizer.from_pretrained("meta-llama")
model = AutoModelForSequenceClassification.from_pretrained("meta-llama")

# Define a text input for classification
text = "This is a great day!"

# Tokenize the input text
inputs = tokenizer(text, return_tensors="pt")

# Perform text classification
outputs = model(**inputs)

# Get the predicted class
predicted_class = torch.argmax(outputs.logits, dim=1).item()
print(f"Predicted class: {predicted_class}")
```
This example demonstrates how to use Meta's LLaMA for text classification, including tokenization and prediction of the class label.

### Google PaLM Sentiment Analysis Example
```python
from google.cloud import language_v1

# Initialize the Google Cloud Language API client
client = language_v1.LanguageServiceClient()

# Define a text input for sentiment analysis
text_content = "I love this product!"

# Prepare the document for analysis
document = language_v1.Document(content=text_content, type_=language_v1.Document.Type.PLAIN_TEXT)

# Perform sentiment analysis
response = client.analyze_sentiment(request={"document": document})
sentiment = response.document_sentiment

# Print the sentiment score and magnitude
print(f"Sentiment score: {sentiment.score}, Magnitude: {sentiment.magnitude}")
```
This example demonstrates how to use Google PaLM for sentiment analysis, including text input preparation and analysis of sentiment score and magnitude.

### Hugging Face Transformers Fine-Tuning Example
```python
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load the tokenizer and model for BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# Load a dataset for fine-tuning
dataset = load_dataset("imdb")

# Tokenize the dataset
tokenized_dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding=True), batched=True)

# Define training arguments
training_args = TrainingArguments(output_dir="./results", num_train_epochs=1, per_device_train_batch_size=8)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"]
)

# Fine-tune the model
trainer.train()
```
This example demonstrates how to use Hugging Face Transformers to fine-tune a pre-trained BERT model for a specific task such as text classification.
