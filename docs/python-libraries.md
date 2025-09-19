# Python Libraries for NLP

This document provides an overview of important Python libraries commonly used in Natural Language Processing (NLP) tasks, from text processing to machine learning and visualization.

## Core NLP Libraries

### NLTK (Natural Language Toolkit)
**Purpose**: Comprehensive platform for building Python programs to work with human language data.

**Key Features**:
- Text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning
- Over 50 corpora and lexical resources
- Suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning

**Common Use Cases**:
- Text preprocessing
- Sentiment analysis
- Part-of-speech tagging
- Named entity recognition
- Text classification

**Installation**: `pip install nltk`

**Example**:
```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

text = "Natural language processing is fascinating!"
tokens = word_tokenize(text)
stop_words = set(stopwords.words('english'))
filtered_tokens = [w for w in tokens if w.lower() not in stop_words]
```

### spaCy
**Purpose**: Industrial-strength Natural Language Processing library designed for production use.

**Key Features**:
- Fast and efficient tokenization, POS tagging, dependency parsing
- Named entity recognition
- Pre-trained statistical models for multiple languages
- Easy integration with deep learning frameworks

**Common Use Cases**:
- Information extraction
- Text analytics
- Chatbots and conversational AI
- Document processing pipelines

**Installation**: `pip install spacy`

**Example**:
```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
for ent in doc.ents:
    print(ent.text, ent.label_)
```

## Text Processing Libraries

### Regex (Regular Expressions)
**Purpose**: Pattern matching and text manipulation using regular expressions.

**Key Features**:
- Advanced pattern matching capabilities
- Text search and replace operations
- String validation and extraction
- Enhanced Unicode support compared to Python's built-in re module

**Common Use Cases**:
- Text cleaning and preprocessing
- Data extraction from unstructured text
- Input validation
- Text pattern matching

**Installation**: `pip install regex`

**Example**:
```python
import regex as re

text = "Contact us at email@example.com or phone: (555) 123-4567"
emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
phones = re.findall(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
```

### ICU (International Components for Unicode)
**Purpose**: Robust and full-featured Unicode and locale support.

**Key Features**:
- Unicode text processing
- Internationalization support
- Collation and sorting for different languages
- Date, time, and number formatting for various locales

**Common Use Cases**:
- Multilingual text processing
- International applications
- Text normalization across languages
- Locale-specific operations

**Installation**: `pip install pyicu`

**Example**:
```python
import icu

# Text normalization
normalizer = icu.Normalizer2.getNFCInstance()
normalized_text = normalizer.normalize("café")

# Collation for sorting
collator = icu.Collator.createInstance(icu.Locale('en_US'))
sorted_words = sorted(['apple', 'café', 'zebra'], key=collator.getSortKey)
```

## Machine Learning Libraries

### Transformers (Hugging Face)
**Purpose**: State-of-the-art Natural Language Processing models and pipelines.

**Key Features**:
- Pre-trained transformer models (BERT, GPT, T5, etc.)
- Easy-to-use pipelines for common NLP tasks
- Model fine-tuning capabilities
- Primary integration with PyTorch (TensorFlow support available but not preferred)

**Common Use Cases**:
- Text classification
- Question answering
- Text generation
- Language translation
- Sentiment analysis

**Installation**: `pip install transformers`

**Example** (using PyTorch backend):
```python
from transformers import pipeline, AutoTokenizer, AutoModel
import torch

# Sentiment analysis with PyTorch backend
classifier = pipeline("sentiment-analysis", framework="pt")
result = classifier("I love this new NLP library!")

# Text generation with PyTorch backend
generator = pipeline("text-generation", model="gpt2", framework="pt")
generated = generator("The future of AI is", max_length=50, num_return_sequences=1)

# Manual model loading with PyTorch
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Vietnamese/English example
text = "My name is John"  # English: "My name is John" → Vietnamese: "Tên tôi là John"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
```

### PyTorch
**Purpose**: Deep learning framework with dynamic computational graphs.

**Key Features**:
- Dynamic neural networks
- Automatic differentiation
- GPU acceleration
- Extensive ecosystem for NLP (torchtext, etc.)

**Common Use Cases**:
- Building custom neural network models
- Research and prototyping
- Deep learning for NLP
- Model training and inference

**Installation**: `pip install torch`

**Example**:
```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, _) = self.rnn(embedded)
        return self.fc(output[:, -1, :])
```

### TensorFlow (Optional - Use PyTorch When Possible)
**Purpose**: End-to-end open source platform for machine learning.

**Note**: This repository primarily uses PyTorch. TensorFlow is only used when PyTorch cannot implement the required functionality.

**Key Features**:
- Static and dynamic computational graphs
- Production-ready deployment
- TensorBoard for visualization
- Extensive ecosystem (TensorFlow Extended, TensorFlow Lite, etc.)

**Common Use Cases**:
- Large-scale machine learning (when PyTorch is insufficient)
- Specific pre-trained models only available in TensorFlow
- Legacy code compatibility

**Installation**: `pip install tensorflow` (only if needed)

**Example** (use PyTorch equivalent when possible):
```python
# TensorFlow example (prefer PyTorch alternatives)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential([
    Embedding(vocab_size, embedding_dim),
    LSTM(128, return_sequences=False),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### Keras (Deprecated in Favor of PyTorch)
**Purpose**: High-level neural networks API, now integrated with TensorFlow.

**Note**: This repository uses PyTorch as the primary framework. Keras examples are included only for reference or when PyTorch alternatives are not available.

**Key Features**:
- User-friendly API for building neural networks
- Modular and composable
- Support for both convolutional and recurrent networks
- Easy prototyping

**Common Use Cases**:
- Legacy code compatibility
- Specific functionality not available in PyTorch

**Installation**: Included with TensorFlow (`pip install tensorflow`)

**Example** (prefer PyTorch equivalents):
```python
# Keras example (prefer PyTorch alternatives)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

model = Sequential([
    Dense(128, activation='relu', input_shape=(input_dim,)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

### Scikit-learn
**Purpose**: Machine learning library for classical ML algorithms.

**Key Features**:
- Wide range of supervised and unsupervised learning algorithms
- Model selection and evaluation tools
- Data preprocessing utilities
- Consistent API across different algorithms

**Common Use Cases**:
- Text classification with traditional ML algorithms
- Feature extraction and selection
- Model evaluation and validation
- Clustering and dimensionality reduction

**Installation**: `pip install scikit-learn`

**Example**:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Create a text classification pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('classifier', MultinomialNB())
])

# Train the model
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

## Data Processing Libraries

### Pandas
**Purpose**: Data manipulation and analysis library.

**Key Features**:
- DataFrame and Series data structures
- Data cleaning and preprocessing tools
- File I/O for various formats (CSV, JSON, Excel, etc.)
- Group operations and data aggregation

**Common Use Cases**:
- Loading and preprocessing text datasets
- Data exploration and analysis
- Handling structured text data
- Data cleaning and transformation

**Installation**: `pip install pandas`

**Example**:
```python
import pandas as pd

# Load text data
df = pd.read_csv('text_dataset.csv')

# Basic text preprocessing
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()
df['clean_text'] = df['text'].str.lower().str.replace('[^\w\s]', '', regex=True)

# Group by category and analyze
category_stats = df.groupby('category')['text_length'].agg(['mean', 'std', 'count'])
```

### NumPy
**Purpose**: Fundamental package for scientific computing with Python.

**Key Features**:
- N-dimensional array objects
- Mathematical functions for arrays
- Linear algebra operations
- Random number generation

**Common Use Cases**:
- Numerical computations for NLP
- Array operations for text processing
- Mathematical operations on embeddings
- Foundation for other scientific libraries

**Installation**: `pip install numpy`

**Example**:
```python
import numpy as np

# Word embeddings operations
embeddings = np.random.rand(1000, 300)  # 1000 words, 300-dim vectors

# Cosine similarity between words
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Find similar words
word1_idx, word2_idx = 0, 1
similarity = cosine_similarity(embeddings[word1_idx], embeddings[word2_idx])

# Compute centroid of word embeddings
centroid = np.mean(embeddings, axis=0)
```

## Visualization Libraries

### Matplotlib
**Purpose**: Comprehensive library for creating static, animated, and interactive visualizations.

**Key Features**:
- Wide variety of plot types
- Highly customizable
- Publication-quality figures
- Integration with NumPy and Pandas

**Common Use Cases**:
- Plotting text analysis results
- Visualizing model performance
- Creating statistical plots
- Displaying word frequency distributions

**Installation**: `pip install matplotlib`

**Example**:
```python
import matplotlib.pyplot as plt
import numpy as np

# Word frequency visualization
words = ['python', 'nlp', 'machine', 'learning', 'data']
frequencies = [150, 120, 95, 85, 110]

plt.figure(figsize=(10, 6))
plt.bar(words, frequencies, color='skyblue')
plt.title('Word Frequency Distribution')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

# Model training loss curve
epochs = np.arange(1, 101)
loss = np.exp(-epochs/30) + 0.1 * np.random.normal(0, 1, 100)

plt.figure(figsize=(10, 6))
plt.plot(epochs, loss, label='Training Loss')
plt.title('Model Training Progress')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
```

### Seaborn
**Purpose**: Statistical data visualization library based on matplotlib.

**Key Features**:
- High-level interface for attractive statistical graphics
- Built-in themes and color palettes
- Statistical plot types
- Easy integration with Pandas

**Common Use Cases**:
- Statistical visualization of text data
- Correlation matrices
- Distribution plots
- Categorical data visualization

**Installation**: `pip install seaborn`

**Example**:
```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Sentiment analysis visualization
data = pd.DataFrame({
    'sentiment': ['positive', 'negative', 'neutral'] * 100,
    'confidence': np.random.beta(2, 5, 300),
    'text_length': np.random.normal(100, 30, 300)
})

# Distribution plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
sns.boxplot(data=data, x='sentiment', y='confidence')
plt.title('Sentiment Confidence Distribution')

plt.subplot(1, 2, 2)
sns.scatterplot(data=data, x='text_length', y='confidence', hue='sentiment')
plt.title('Text Length vs Confidence by Sentiment')
plt.tight_layout()
plt.show()

# Correlation heatmap
correlation_matrix = data[['confidence', 'text_length']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()
```

### Plotly
**Purpose**: Interactive web-based visualization library.

**Key Features**:
- Interactive plots and dashboards
- Web-based visualizations
- 3D plotting capabilities
- Easy sharing and embedding

**Common Use Cases**:
- Interactive data exploration
- Web applications with visualizations
- 3D visualization of embeddings
- Dashboard creation

**Installation**: `pip install plotly`

**Example**:
```python
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Interactive word cloud alternative
word_data = pd.DataFrame({
    'word': ['python', 'nlp', 'machine', 'learning', 'data', 'text', 'analysis'],
    'frequency': [150, 120, 95, 85, 110, 75, 65],
    'category': ['language', 'field', 'concept', 'concept', 'concept', 'concept', 'process']
})

fig = px.scatter(word_data, x='word', y='frequency', 
                 size='frequency', color='category',
                 title='Interactive Word Frequency Analysis')
fig.show()

# 3D visualization of word embeddings (example with random data)
n_words = 100
embeddings_3d = np.random.rand(n_words, 3)
words = [f'word_{i}' for i in range(n_words)]

fig = go.Figure(data=go.Scatter3d(
    x=embeddings_3d[:, 0],
    y=embeddings_3d[:, 1],
    z=embeddings_3d[:, 2],
    mode='markers+text',
    text=words[:20],  # Show labels for first 20 words
    textposition="top center",
    marker=dict(size=5, color=embeddings_3d[:, 0], colorscale='Viridis')
))

fig.update_layout(title='3D Word Embeddings Visualization')
fig.show()
```

## Recommended Library Combinations

### For Beginners
- **NLTK** + **Pandas** + **Matplotlib**: Great for learning NLP fundamentals
- **spaCy** + **Pandas** + **Seaborn**: Production-ready text processing with good visualization

### For Deep Learning
- **Transformers** + **PyTorch** + **NumPy**: State-of-the-art NLP with flexibility
- **TensorFlow/Keras** + **Pandas** + **Plotly**: End-to-end ML pipeline with interactive visualization

### For Data Science
- **scikit-learn** + **Pandas** + **Seaborn**: Traditional ML approaches with excellent data handling
- **spaCy** + **Pandas** + **Plotly**: Industrial NLP with interactive analysis

### For Research
- **PyTorch** + **Transformers** + **Matplotlib**: Cutting-edge research with detailed analysis
- **NLTK** + **NumPy** + **Matplotlib**: Fundamental research with custom implementations

## Installation Guide

### All Libraries at Once (PyTorch-Focused)
```bash
pip install nltk spacy regex pyicu transformers torch torchvision torchaudio scikit-learn pandas numpy matplotlib seaborn plotly

# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Alternative with TensorFlow (Only if Required)
```bash
# Install TensorFlow only if PyTorch cannot meet your needs
pip install nltk spacy regex pyicu transformers torch torchvision torchaudio tensorflow scikit-learn pandas numpy matplotlib seaborn plotly

# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Virtual Environment Setup
```bash
# Create virtual environment
python -m venv nlp_env

# Activate (Linux/Mac)
source nlp_env/bin/activate

# Activate (Windows)
nlp_env\Scripts\activate

# Install libraries (PyTorch-focused)
pip install -r requirements.txt
```

This comprehensive overview should help you choose the right tools for your NLP projects and understand how they work together in the modern PyTorch-focused NLP ecosystem.