# Embeddings in Natural Language Processing

**Embeddings** are one of the most fundamental and transformative concepts in modern Natural Language Processing (NLP). They provide dense, numerical representations of linguistic units (words, sentences, documents) that capture semantic meaning and enable machines to understand and process human language effectively.

> **Note on Examples**: Code examples in this document prioritize offline functionality. Examples requiring pre-trained models include alternative implementations that demonstrate core concepts without internet dependency.

## Table of Contents

1. [What are Embeddings?](#what-are-embeddings)
2. [Why Embeddings Matter](#why-embeddings-matter)
3. [Mathematical Foundation](#mathematical-foundation)
4. [Types of Embeddings](#types-of-embeddings)
5. [Word Embeddings](#word-embeddings)
6. [Sentence and Document Embeddings](#sentence-and-document-embeddings)
7. [Contextual Embeddings](#contextual-embeddings)
8. [Implementation Examples](#implementation-examples)
9. [Practical Applications](#practical-applications)
10. [Best Practices](#best-practices)
11. [Advanced Topics](#advanced-topics)

## What are Embeddings?

**Embeddings** are dense vector representations that map discrete linguistic units (words, phrases, sentences, documents) into continuous vector spaces where semantic relationships are preserved through geometric relationships.

### Core Concept

Think of embeddings as a translation system:
- **Input**: Discrete symbols (words like "king", "queen", "man", "woman")
- **Output**: Dense vectors (e.g., 300-dimensional real-valued vectors)
- **Property**: Semantically similar words have similar vectors

### Mathematical Definition

An embedding is a function `f: V → ℝᵈ` where:
- `V` is a vocabulary of discrete units
- `ℝᵈ` is a d-dimensional vector space
- `f` maps each unit to a dense vector

Example:
```
f("king") = [0.2, -0.1, 0.8, ..., 0.3]    # 300-dimensional vector
f("queen") = [0.1, -0.2, 0.7, ..., 0.4]   # 300-dimensional vector
```

### Geometric Intuition

In the embedding space:
- **Distance** reflects semantic similarity
- **Direction** can capture relationships
- **Clustering** groups related concepts

Famous example: `king - man + woman ≈ queen`

## Why Embeddings Matter

### 1. **Semantic Understanding**

Traditional one-hot encoding treats words as independent symbols:
```
"king" = [1, 0, 0, 0, ...]   # 50,000-dimensional sparse vector
"queen" = [0, 1, 0, 0, ...]  # No relationship captured
```

Embeddings capture semantic relationships:
```
"king" = [0.2, -0.1, 0.8, ...]   # 300-dimensional dense vector
"queen" = [0.1, -0.2, 0.7, ...]  # Similar to "king"
```

### 2. **Dimensionality Efficiency**

- **One-hot**: Vocabulary size dimensions (often 50k-100k+)
- **Embeddings**: Fixed dimensions (typically 100-1000)
- **Result**: Massive reduction in parameters and memory

### 3. **Generalization**

Embeddings enable models to:
- Understand words never seen together during training
- Transfer knowledge between related concepts
- Handle out-of-vocabulary words (with subword embeddings)

### 4. **Foundation for Modern NLP**

Embeddings are the building blocks of:
- **Neural language models** (GPT, BERT)
- **Machine translation systems**
- **Recommendation engines**
- **Search and retrieval systems**

## Mathematical Foundation

### Vector Space Properties

Embeddings live in a **metric space** with key properties:

**1. Distance Measures**
```python
import numpy as np

def cosine_similarity(v1, v2):
    """Measure semantic similarity between embeddings."""
    dot_product = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    return dot_product / norms

def euclidean_distance(v1, v2):
    """Measure dissimilarity between embeddings."""
    return np.linalg.norm(v1 - v2)
```

**2. Linear Relationships**
```python
# Vector arithmetic captures semantic relationships
king = embedding_model["king"]
man = embedding_model["man"]
woman = embedding_model["woman"]

# Compute: king - man + woman
result_vector = king - man + woman

# Find closest word to result_vector
queen_candidate = find_closest_word(result_vector)
# Often returns "queen"
```

### Distributional Hypothesis

Embeddings are based on the **distributional hypothesis**:
> "Words that occur in similar contexts tend to have similar meanings"

This principle drives most embedding algorithms:
- **Word2Vec**: Predicts context from word or word from context
- **GloVe**: Factorizes word co-occurrence statistics
- **FastText**: Extends to character-level patterns

## Types of Embeddings

### 1. **Static Embeddings**
- Fixed representation per word
- Context-independent
- Examples: Word2Vec, GloVe, FastText

### 2. **Contextual Embeddings**
- Dynamic representations based on context
- Same word gets different vectors in different sentences
- Examples: ELMo, BERT, GPT

### 3. **Subword Embeddings**
- Represent parts of words (morphemes, characters, byte-pairs)
- Handle out-of-vocabulary words
- Examples: FastText, BPE, SentencePiece

### 4. **Multilingual Embeddings**
- Shared vector space across languages
- Enable cross-lingual understanding
- Examples: mBERT, XLM-R, LASER

## Word Embeddings

### Word2Vec

**Word2Vec** introduced the modern era of word embeddings with two main architectures:

#### Skip-gram Model
- **Objective**: Predict context words given a target word
- **Intuition**: Words with similar contexts should have similar representations

```python
# Simplified Skip-gram concept (offline implementation)
import numpy as np
from collections import defaultdict, Counter
import random

class SimpleWord2Vec:
    def __init__(self, vector_size=100, window=5, min_count=1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.vocab = {}
        self.word_vectors = None
        
    def build_vocab(self, sentences):
        """Build vocabulary from sentences."""
        word_count = Counter()
        for sentence in sentences:
            words = sentence.lower().split()
            word_count.update(words)
        
        # Filter by minimum count
        vocab_words = [word for word, count in word_count.items() 
                      if count >= self.min_count]
        
        # Create word to index mapping
        self.vocab = {word: idx for idx, word in enumerate(vocab_words)}
        self.vocab_size = len(vocab_words)
        
        # Initialize random word vectors
        self.word_vectors = np.random.uniform(
            -0.5/self.vector_size, 0.5/self.vector_size,
            (self.vocab_size, self.vector_size)
        )
    
    def get_context_pairs(self, sentences):
        """Generate (target, context) pairs for training."""
        pairs = []
        for sentence in sentences:
            words = sentence.lower().split()
            for i, target_word in enumerate(words):
                if target_word not in self.vocab:
                    continue
                    
                # Get context words within window
                start = max(0, i - self.window)
                end = min(len(words), i + self.window + 1)
                
                for j in range(start, end):
                    if i != j and words[j] in self.vocab:
                        pairs.append((target_word, words[j]))
        return pairs
    
    def get_vector(self, word):
        """Get vector for a word."""
        if word in self.vocab:
            return self.word_vectors[self.vocab[word]]
        return None
    
    def most_similar(self, word, top_k=5):
        """Find most similar words to given word."""
        if word not in self.vocab:
            return []
        
        word_vec = self.get_vector(word)
        similarities = []
        
        for other_word, idx in self.vocab.items():
            if other_word != word:
                other_vec = self.word_vectors[idx]
                similarity = self.cosine_similarity(word_vec, other_vec)
                similarities.append((other_word, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def cosine_similarity(self, v1, v2):
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
        return dot_product / (norms + 1e-10)

# Example usage
sentences = [
    "the king rules the kingdom",
    "the queen rules the kingdom",
    "the man walks in the park",
    "the woman walks in the park",
    "kings and queens are royalty",
    "men and women are people"
]

model = SimpleWord2Vec(vector_size=50, window=2)
model.build_vocab(sentences)

print(f"Vocabulary size: {model.vocab_size}")
print(f"Words in vocabulary: {list(model.vocab.keys())}")

# Check similarity (note: this simple model needs more training)
if "king" in model.vocab and "queen" in model.vocab:
    king_vec = model.get_vector("king")
    queen_vec = model.get_vector("queen")
    similarity = model.cosine_similarity(king_vec, queen_vec)
    print(f"Similarity between 'king' and 'queen': {similarity:.3f}")
```

#### CBOW (Continuous Bag of Words)
- **Objective**: Predict target word given context words
- **Intuition**: Context determines meaning

### GloVe (Global Vectors)

**GloVe** combines the advantages of matrix factorization methods with the efficiency of Word2Vec:

```python
# Simplified GloVe concept demonstration
import numpy as np
from collections import defaultdict, Counter

class SimpleGloVe:
    def __init__(self, vector_size=100, x_max=100, alpha=0.75):
        self.vector_size = vector_size
        self.x_max = x_max
        self.alpha = alpha
        
    def build_cooccurrence_matrix(self, sentences, window_size=5):
        """Build word co-occurrence matrix."""
        # Build vocabulary
        word_count = Counter()
        for sentence in sentences:
            words = sentence.lower().split()
            word_count.update(words)
        
        vocab = {word: idx for idx, word in enumerate(word_count.keys())}
        vocab_size = len(vocab)
        
        # Initialize co-occurrence matrix
        cooccur = defaultdict(float)
        
        # Count co-occurrences
        for sentence in sentences:
            words = sentence.lower().split()
            for i, word in enumerate(words):
                if word in vocab:
                    for j in range(max(0, i - window_size), 
                                 min(len(words), i + window_size + 1)):
                        if i != j and words[j] in vocab:
                            distance = abs(i - j)
                            weight = 1.0 / distance
                            cooccur[(vocab[word], vocab[words[j]])] += weight
        
        return vocab, cooccur
    
    def weighting_function(self, x):
        """GloVe weighting function."""
        if x < self.x_max:
            return (x / self.x_max) ** self.alpha
        return 1.0

# Example usage
sentences = [
    "natural language processing is fascinating",
    "machine learning and deep learning are related",
    "neural networks process language naturally",
    "transformers revolutionized natural language processing"
]

glove = SimpleGloVe(vector_size=50)
vocab, cooccur = glove.build_cooccurrence_matrix(sentences, window_size=3)

print(f"Vocabulary: {list(vocab.keys())}")
print(f"Co-occurrence pairs: {len(cooccur)}")

# Show some co-occurrence examples
for (i, j), count in list(cooccur.items())[:5]:
    word_i = [word for word, idx in vocab.items() if idx == i][0]
    word_j = [word for word, idx in vocab.items() if idx == j][0]
    print(f"'{word_i}' co-occurs with '{word_j}': {count:.2f}")
```

### FastText

**FastText** extends Word2Vec to subword level, handling out-of-vocabulary words:

```python
# Simplified FastText subword concept
class SimpleFastText:
    def __init__(self, vector_size=100, min_n=3, max_n=6):
        self.vector_size = vector_size
        self.min_n = min_n  # Minimum n-gram length
        self.max_n = max_n  # Maximum n-gram length
        self.word_vectors = {}
        self.subword_vectors = {}
    
    def get_subwords(self, word):
        """Extract character n-grams from word."""
        word = f"<{word}>"  # Add boundary markers
        subwords = set()
        
        for n in range(self.min_n, self.max_n + 1):
            for i in range(len(word) - n + 1):
                subword = word[i:i + n]
                subwords.add(subword)
        
        return list(subwords)
    
    def get_word_vector(self, word):
        """Get vector for word (including OOV words)."""
        if word in self.word_vectors:
            return self.word_vectors[word]
        
        # For OOV words, average subword vectors
        subwords = self.get_subwords(word)
        subword_vecs = []
        
        for subword in subwords:
            if subword in self.subword_vectors:
                subword_vecs.append(self.subword_vectors[subword])
        
        if subword_vecs:
            return np.mean(subword_vecs, axis=0)
        else:
            # Return random vector for unknown subwords
            return np.random.normal(0, 1, self.vector_size)

# Example usage
fasttext = SimpleFastText(vector_size=50)

# Example: handling out-of-vocabulary words
word = "unbelievable"
subwords = fasttext.get_subwords(word)
print(f"Subwords for '{word}': {subwords[:10]}...")  # Show first 10
```

## Sentence and Document Embeddings

### Averaging Word Embeddings

The simplest approach to sentence embeddings:

```python
def sentence_embedding_average(sentence, word_vectors):
    """Create sentence embedding by averaging word vectors."""
    words = sentence.lower().split()
    vectors = []
    
    for word in words:
        if word in word_vectors:
            vectors.append(word_vectors[word])
    
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(len(next(iter(word_vectors.values()))))

# Example usage
word_vectors = {
    "natural": np.random.rand(100),
    "language": np.random.rand(100),
    "processing": np.random.rand(100),
    "is": np.random.rand(100),
    "interesting": np.random.rand(100)
}

sentence = "natural language processing is interesting"
sent_embedding = sentence_embedding_average(sentence, word_vectors)
print(f"Sentence embedding shape: {sent_embedding.shape}")
```

### Doc2Vec

**Doc2Vec** extends Word2Vec to learn document-level representations:

```python
# Simplified Doc2Vec concept
class SimpleDoc2Vec:
    def __init__(self, vector_size=100):
        self.vector_size = vector_size
        self.doc_vectors = {}
        self.word_vectors = {}
    
    def train_documents(self, documents):
        """Train on a collection of documents."""
        # In real Doc2Vec, this would involve:
        # 1. Adding document vectors to the neural network
        # 2. Training to predict words given document + context
        # 3. Learning both word and document representations
        
        for doc_id, document in enumerate(documents):
            # Simplified: random initialization
            self.doc_vectors[doc_id] = np.random.normal(0, 1, self.vector_size)
    
    def infer_vector(self, document):
        """Infer vector for new document."""
        # In real Doc2Vec, this would involve inference procedure
        # Simplified: average word vectors if available
        words = document.lower().split()
        if hasattr(self, 'word_vectors') and self.word_vectors:
            vectors = [self.word_vectors.get(word, np.zeros(self.vector_size)) 
                      for word in words]
            return np.mean(vectors, axis=0)
        return np.random.normal(0, 1, self.vector_size)

# Example usage
documents = [
    "Natural language processing is a subfield of artificial intelligence",
    "Machine learning algorithms can process human language",
    "Deep neural networks have revolutionized NLP applications"
]

doc2vec = SimpleDoc2Vec(vector_size=100)
doc2vec.train_documents(documents)

print(f"Document 0 vector shape: {doc2vec.doc_vectors[0].shape}")
```

### TF-IDF Embeddings

While not neural embeddings, TF-IDF creates meaningful document representations:

```python
from collections import Counter
import math

class SimpleTFIDF:
    def __init__(self):
        self.vocab = set()
        self.idf = {}
        
    def fit(self, documents):
        """Compute IDF values from document collection."""
        # Build vocabulary
        for doc in documents:
            words = doc.lower().split()
            self.vocab.update(words)
        
        # Compute IDF for each word
        total_docs = len(documents)
        word_doc_count = Counter()
        
        for doc in documents:
            unique_words = set(doc.lower().split())
            word_doc_count.update(unique_words)
        
        for word in self.vocab:
            df = word_doc_count[word]  # Document frequency
            self.idf[word] = math.log(total_docs / (df + 1))
    
    def transform(self, document):
        """Convert document to TF-IDF vector."""
        words = document.lower().split()
        word_count = Counter(words)
        total_words = len(words)
        
        # Create TF-IDF vector
        tfidf_vector = {}
        for word in self.vocab:
            tf = word_count[word] / total_words if total_words > 0 else 0
            tfidf_vector[word] = tf * self.idf.get(word, 0)
        
        return tfidf_vector

# Example usage
documents = [
    "natural language processing tasks",
    "machine learning and data science",
    "neural networks for language understanding"
]

tfidf = SimpleTFIDF()
tfidf.fit(documents)

doc_vector = tfidf.transform("natural language understanding")
print(f"TF-IDF vector keys: {list(doc_vector.keys())[:5]}")
print(f"Sample values: {[(k, f'{v:.3f}') for k, v in list(doc_vector.items())[:3]]}")
```

## Contextual Embeddings

### Evolution from Static to Contextual

**Problem with Static Embeddings**:
```python
# Static embeddings give same vector regardless of context
bank_vector = static_embedding["bank"]  # Same for all contexts

# Sentence 1: "I deposited money in the bank"      (financial institution)
# Sentence 2: "The river bank was muddy"           (riverbank)
# Both get the same "bank" vector!
```

**Solution: Contextual Embeddings**:
```python
# Contextual embeddings provide different vectors per context
bank_financial = contextual_model("I deposited money in the bank")["bank"]
bank_river = contextual_model("The river bank was muddy")["bank"]

# These vectors will be different!
```

### BERT Embeddings

**BERT** (Bidirectional Encoder Representations from Transformers) creates powerful contextual embeddings:

```python
# Conceptual BERT embedding extraction
class ConceptualBERT:
    def __init__(self):
        # Simplified: BERT has 12-24 transformer layers
        self.num_layers = 12
        self.hidden_size = 768
    
    def get_contextual_embeddings(self, sentence, word_position):
        """Get contextual embedding for word at specific position."""
        # In real BERT:
        # 1. Tokenize sentence
        # 2. Add special tokens ([CLS], [SEP])
        # 3. Pass through transformer layers
        # 4. Extract hidden states
        
        # Simplified demonstration
        tokens = sentence.lower().split()
        context_influence = self.calculate_context_influence(tokens, word_position)
        
        # Base word embedding modified by context
        base_embedding = np.random.normal(0, 1, self.hidden_size)
        contextual_embedding = base_embedding * context_influence
        
        return contextual_embedding
    
    def calculate_context_influence(self, tokens, target_pos):
        """Simplified context influence calculation."""
        # Real BERT uses attention mechanisms
        influences = []
        
        for i, token in enumerate(tokens):
            if i != target_pos:
                distance = abs(i - target_pos)
                influence = 1.0 / (distance + 1)  # Closer words have more influence
                influences.append(influence)
        
        return np.mean(influences) if influences else 1.0

# Example usage
bert_model = ConceptualBERT()

sentence1 = "The bank approved my loan application"
sentence2 = "We picnicked by the river bank"

# Get "bank" embeddings from different contexts
bank_financial = bert_model.get_contextual_embeddings(sentence1, 1)  # "bank" at position 1
bank_river = bert_model.get_contextual_embeddings(sentence2, 5)      # "bank" at position 5

print(f"Financial bank embedding shape: {bank_financial.shape}")
print(f"River bank embedding shape: {bank_river.shape}")

# In real scenarios, these would be significantly different
cosine_sim = np.dot(bank_financial, bank_river) / (
    np.linalg.norm(bank_financial) * np.linalg.norm(bank_river)
)
print(f"Similarity between contexts: {cosine_sim:.3f}")
```

### Attention-Based Context

Contextual embeddings use **attention mechanisms** to compute context-aware representations:

```python
def simplified_attention(query, keys, values):
    """Simplified attention mechanism for context."""
    # Compute attention scores
    scores = []
    for key in keys:
        score = np.dot(query, key)  # Simplified: just dot product
        scores.append(score)
    
    # Apply softmax to get attention weights
    exp_scores = np.exp(scores)
    attention_weights = exp_scores / np.sum(exp_scores)
    
    # Compute weighted sum of values
    context_vector = np.zeros_like(values[0])
    for weight, value in zip(attention_weights, values):
        context_vector += weight * value
    
    return context_vector, attention_weights

# Example: computing context for "bank" in sentence
word_embeddings = {
    "the": np.random.rand(50),
    "bank": np.random.rand(50),
    "approved": np.random.rand(50),
    "my": np.random.rand(50),
    "loan": np.random.rand(50)
}

sentence = ["the", "bank", "approved", "my", "loan"]
target_word = "bank"
target_pos = 1

# Query: the word we want context for
query = word_embeddings[target_word]

# Keys and values: all words in sentence (including target)
keys = [word_embeddings[word] for word in sentence]
values = [word_embeddings[word] for word in sentence]

# Compute contextual representation
context_vector, attention_weights = simplified_attention(query, keys, values)

print(f"Attention weights: {[f'{w:.3f}' for w in attention_weights]}")
print(f"Words: {sentence}")
print(f"Contextual embedding shape: {context_vector.shape}")
```

## Implementation Examples

### Using Pre-trained Models (Online)

When internet is available, you can use powerful pre-trained embeddings:

```python
# NLTK Word2Vec-style embeddings (requires download)
"""
import gensim.downloader as api

# Load pre-trained Word2Vec model
word2vec_model = api.load("word2vec-google-news-300")

# Get word vectors
king_vector = word2vec_model["king"]
queen_vector = word2vec_model["queen"]

# Find similar words
similar_to_king = word2vec_model.most_similar("king", topn=5)
print(f"Words similar to 'king': {similar_to_king}")

# Vector arithmetic
result = word2vec_model.most_similar(
    positive=["king", "woman"], 
    negative=["man"], 
    topn=1
)
print(f"king - man + woman = {result}")
"""

# Transformers library for BERT embeddings (requires download)
"""
from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(text):
    # Tokenize and encode
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    
    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract embeddings (last hidden state)
    embeddings = outputs.last_hidden_state
    return embeddings

# Example usage
sentence = "Natural language processing is fascinating"
embeddings = get_bert_embeddings(sentence)
print(f"BERT embeddings shape: {embeddings.shape}")
"""
```

### Offline Embedding Utilities

```python
# Comprehensive offline embedding utilities
class EmbeddingUtils:
    @staticmethod
    def cosine_similarity(vec1, vec2):
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return dot_product / (norms + 1e-10)
    
    @staticmethod
    def euclidean_distance(vec1, vec2):
        """Compute Euclidean distance between two vectors."""
        return np.linalg.norm(vec1 - vec2)
    
    @staticmethod
    def find_closest_words(target_vector, word_vectors, top_k=5):
        """Find words with vectors closest to target vector."""
        similarities = []
        
        for word, vector in word_vectors.items():
            similarity = EmbeddingUtils.cosine_similarity(target_vector, vector)
            similarities.append((word, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    @staticmethod
    def vector_arithmetic(word_vectors, positive=None, negative=None, top_k=5):
        """Perform vector arithmetic: positive words - negative words."""
        if positive is None:
            positive = []
        if negative is None:
            negative = []
        
        # Start with zero vector
        result_vector = np.zeros_like(next(iter(word_vectors.values())))
        
        # Add positive vectors
        for word in positive:
            if word in word_vectors:
                result_vector += word_vectors[word]
        
        # Subtract negative vectors
        for word in negative:
            if word in word_vectors:
                result_vector -= word_vectors[word]
        
        # Find closest words (excluding input words)
        excluded_words = set(positive + negative)
        filtered_vectors = {w: v for w, v in word_vectors.items() 
                          if w not in excluded_words}
        
        return EmbeddingUtils.find_closest_words(result_vector, filtered_vectors, top_k)

# Example usage
# Create sample word vectors
sample_vectors = {
    "king": np.array([1.0, 0.8, 0.2, 0.1]),
    "queen": np.array([0.9, 0.9, 0.3, 0.2]),
    "man": np.array([0.8, 0.1, 0.7, 0.3]),
    "woman": np.array([0.7, 0.2, 0.8, 0.4]),
    "royal": np.array([0.9, 0.7, 0.1, 0.1]),
    "noble": np.array([0.8, 0.6, 0.2, 0.1])
}

# Test vector arithmetic
utils = EmbeddingUtils()
result = utils.vector_arithmetic(
    sample_vectors, 
    positive=["king", "woman"], 
    negative=["man"],
    top_k=3
)
print(f"king - man + woman = {result}")

# Test similarity
similarity = utils.cosine_similarity(
    sample_vectors["king"], 
    sample_vectors["queen"]
)
print(f"Similarity between king and queen: {similarity:.3f}")
```

## Practical Applications

### 1. **Information Retrieval**

```python
def semantic_search(query, documents, embedding_model):
    """Search documents using semantic similarity."""
    # Get query embedding
    query_embedding = embedding_model.get_sentence_embedding(query)
    
    # Get document embeddings
    doc_embeddings = []
    for doc in documents:
        doc_embedding = embedding_model.get_sentence_embedding(doc)
        doc_embeddings.append(doc_embedding)
    
    # Compute similarities
    similarities = []
    for i, doc_embedding in enumerate(doc_embeddings):
        similarity = cosine_similarity(query_embedding, doc_embedding)
        similarities.append((i, similarity, documents[i]))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities

# Example usage (conceptual)
documents = [
    "Machine learning algorithms for natural language processing",
    "Deep neural networks in computer vision applications",
    "Statistical methods for text analysis and mining",
    "Transformer architectures in modern NLP systems"
]

query = "NLP deep learning"
# results = semantic_search(query, documents, embedding_model)
```

### 2. **Recommendation Systems**

```python
def content_based_recommendations(user_profile, item_embeddings, top_k=5):
    """Recommend items based on user profile embedding."""
    similarities = []
    
    for item_id, item_embedding in item_embeddings.items():
        similarity = cosine_similarity(user_profile, item_embedding)
        similarities.append((item_id, similarity))
    
    # Sort and return top recommendations
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# Example: Book recommendations based on reading history
book_embeddings = {
    "nlp_book": np.random.rand(100),
    "ml_book": np.random.rand(100),
    "stats_book": np.random.rand(100),
    "cooking_book": np.random.rand(100)
}

# User profile: average of books they've liked
user_profile = (book_embeddings["nlp_book"] + book_embeddings["ml_book"]) / 2

recommendations = content_based_recommendations(user_profile, book_embeddings)
print(f"Recommended books: {recommendations}")
```

### 3. **Clustering and Classification**

```python
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

def cluster_documents(document_embeddings, num_clusters=3):
    """Cluster documents based on their embeddings."""
    # Convert to numpy array
    embeddings_array = np.array(list(document_embeddings.values()))
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings_array)
    
    # Create clusters dictionary
    clusters = {}
    for i, (doc_id, embedding) in enumerate(document_embeddings.items()):
        cluster_id = cluster_labels[i]
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(doc_id)
    
    return clusters

def classify_with_embeddings(train_embeddings, train_labels, test_embeddings):
    """Classify documents using embeddings as features."""
    # Train classifier
    classifier = LogisticRegression(random_state=42)
    classifier.fit(train_embeddings, train_labels)
    
    # Predict on test data
    predictions = classifier.predict(test_embeddings)
    probabilities = classifier.predict_proba(test_embeddings)
    
    return predictions, probabilities

# Example usage (conceptual)
doc_embeddings = {
    f"doc_{i}": np.random.rand(100) for i in range(10)
}

clusters = cluster_documents(doc_embeddings, num_clusters=3)
print(f"Document clusters: {clusters}")
```

## Best Practices

### 1. **Choosing Embedding Dimensions**

```python
def analyze_embedding_dimensions():
    """Guidelines for choosing embedding dimensions."""
    guidelines = {
        "Small vocabularies (< 10K words)": "50-100 dimensions",
        "Medium vocabularies (10K-100K words)": "100-300 dimensions", 
        "Large vocabularies (> 100K words)": "300-1000 dimensions",
        "Computational constraints": "Use smaller dimensions (50-200)",
        "High precision tasks": "Use larger dimensions (300-1000)"
    }
    
    return guidelines

print("Embedding dimension guidelines:")
for context, recommendation in analyze_embedding_dimensions().items():
    print(f"- {context}: {recommendation}")
```

### 2. **Preprocessing for Embeddings**

```python
import re
import unicodedata

def preprocess_for_embeddings(text):
    """Preprocessing pipeline optimized for embeddings."""
    # 1. Unicode normalization
    text = unicodedata.normalize('NFKD', text)
    
    # 2. Lowercase conversion
    text = text.lower()
    
    # 3. Remove or replace special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # 4. Handle multiple whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 5. Remove very short words (optional)
    words = text.split()
    words = [word for word in words if len(word) > 2]
    
    return ' '.join(words)

# Example
raw_text = "Hello, World! This is a test... 123 #hashtag"
processed = preprocess_for_embeddings(raw_text)
print(f"Original: {raw_text}")
print(f"Processed: {processed}")
```

### 3. **Evaluation Metrics**

```python
def evaluate_embeddings(embedding_model, test_cases):
    """Evaluate embedding quality using various metrics."""
    
    def evaluate_similarity(word_pairs):
        """Evaluate word similarity task."""
        correlations = []
        for word1, word2, human_score in word_pairs:
            if word1 in embedding_model and word2 in embedding_model:
                vec1 = embedding_model[word1]
                vec2 = embedding_model[word2]
                model_score = cosine_similarity(vec1, vec2)
                correlations.append((model_score, human_score))
        return correlations
    
    def evaluate_analogy(analogies):
        """Evaluate word analogy task."""
        correct = 0
        total = 0
        
        for a, b, c, expected_d in analogies:
            if all(word in embedding_model for word in [a, b, c]):
                # Compute: b - a + c
                result_vec = (embedding_model[b] - 
                            embedding_model[a] + 
                            embedding_model[c])
                
                # Find closest word
                similarities = []
                for word, vec in embedding_model.items():
                    if word not in [a, b, c]:
                        sim = cosine_similarity(result_vec, vec)
                        similarities.append((word, sim))
                
                similarities.sort(key=lambda x: x[1], reverse=True)
                predicted_d = similarities[0][0]
                
                if predicted_d == expected_d:
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0
    
    # Example test cases
    similarity_pairs = [
        ("king", "queen", 0.8),
        ("man", "woman", 0.7),
        ("car", "automobile", 0.9)
    ]
    
    analogies = [
        ("king", "queen", "man", "woman"),
        ("paris", "france", "london", "england")
    ]
    
    # Note: This is conceptual - would need actual embedding model
    print("Embedding evaluation framework:")
    print("- Similarity correlation with human judgments")
    print("- Analogy task accuracy")
    print("- Clustering coherence")
    print("- Downstream task performance")

evaluate_embeddings(None, None)  # Conceptual example
```

### 4. **Handling Out-of-Vocabulary Words**

```python
class OOVHandler:
    """Strategies for handling out-of-vocabulary words."""
    
    @staticmethod
    def subword_averaging(word, subword_embeddings):
        """Use subword embeddings for OOV words."""
        subwords = []
        # Extract character n-grams
        for n in range(3, 7):  # 3-gram to 6-gram
            for i in range(len(word) - n + 1):
                subword = word[i:i+n]
                if subword in subword_embeddings:
                    subwords.append(subword_embeddings[subword])
        
        if subwords:
            return np.mean(subwords, axis=0)
        return None
    
    @staticmethod
    def nearest_neighbor(word, word_embeddings, distance_threshold=2):
        """Find similar known words for OOV handling."""
        def edit_distance(s1, s2):
            """Compute edit distance between two strings."""
            if len(s1) < len(s2):
                return edit_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        # Find closest known words
        candidates = []
        for known_word in word_embeddings:
            distance = edit_distance(word, known_word)
            if distance <= distance_threshold:
                candidates.append((known_word, distance))
        
        if candidates:
            # Return embedding of closest word
            closest_word = min(candidates, key=lambda x: x[1])[0]
            return word_embeddings[closest_word]
        
        return None
    
    @staticmethod
    def random_initialization(vector_size, seed=42):
        """Random initialization for OOV words."""
        np.random.seed(seed)
        return np.random.normal(0, 1, vector_size)

# Example usage
oov_handler = OOVHandler()

# Simulate word embeddings
word_embeddings = {
    "running": np.random.rand(100),
    "walking": np.random.rand(100),
    "jumping": np.random.rand(100)
}

# Handle OOV word "jogging"
oov_word = "jogging"
fallback_embedding = oov_handler.nearest_neighbor(oov_word, word_embeddings)

if fallback_embedding is not None:
    print(f"Found similar word for '{oov_word}'")
else:
    print(f"No similar word found, using random initialization")
    fallback_embedding = oov_handler.random_initialization(100)
```

## Advanced Topics

### 1. **Multilingual Embeddings**

Multilingual embeddings enable cross-lingual understanding:

```python
class MultilingualEmbeddings:
    """Conceptual multilingual embedding system."""
    
    def __init__(self):
        # Shared vector space for multiple languages
        self.shared_space_dim = 512
        self.language_specific_dim = 128
    
    def create_shared_representation(self, word, language):
        """Create language-agnostic representation."""
        # In real systems, this involves:
        # 1. Language-specific encoding
        # 2. Mapping to shared space
        # 3. Cross-lingual alignment
        
        # Simulate shared representation
        shared_vector = np.random.normal(0, 1, self.shared_space_dim)
        return shared_vector
    
    def cross_lingual_similarity(self, word1, lang1, word2, lang2):
        """Compute similarity across languages."""
        vec1 = self.create_shared_representation(word1, lang1)
        vec2 = self.create_shared_representation(word2, lang2)
        
        return cosine_similarity(vec1, vec2)

# Example: Cross-lingual word similarity
multilingual = MultilingualEmbeddings()

# Compare English "dog" with Spanish "perro"
similarity = multilingual.cross_lingual_similarity("dog", "en", "perro", "es")
print(f"Cross-lingual similarity (dog/perro): {similarity:.3f}")
```

### 2. **Domain Adaptation**

Adapting embeddings to specific domains:

```python
class DomainAdaptation:
    """Techniques for domain-specific embeddings."""
    
    @staticmethod
    def fine_tune_embeddings(base_embeddings, domain_corpus, learning_rate=0.01):
        """Fine-tune pre-trained embeddings on domain data."""
        # Conceptual fine-tuning process:
        # 1. Start with pre-trained embeddings
        # 2. Continue training on domain-specific data
        # 3. Adjust embeddings to capture domain vocabulary
        
        adapted_embeddings = base_embeddings.copy()
        
        # Simulate adaptation (in practice, use gradient descent)
        for word in adapted_embeddings:
            if word in domain_corpus:
                # Add domain-specific noise/adjustment
                noise = np.random.normal(0, learning_rate, 
                                       adapted_embeddings[word].shape)
                adapted_embeddings[word] += noise
        
        return adapted_embeddings
    
    @staticmethod
    def domain_specific_vocabulary(general_vocab, domain_texts):
        """Identify domain-specific vocabulary."""
        # Count word frequencies in domain
        domain_word_count = Counter()
        for text in domain_texts:
            words = text.lower().split()
            domain_word_count.update(words)
        
        # Identify words that are disproportionately frequent in domain
        domain_specific = []
        for word, count in domain_word_count.items():
            if word in general_vocab:
                # Compare domain frequency with general frequency
                domain_freq = count / len(domain_texts)
                general_freq = general_vocab.get(word, 0)
                
                if domain_freq > 2 * general_freq:  # Threshold
                    domain_specific.append(word)
        
        return domain_specific

# Example usage
base_embeddings = {f"word_{i}": np.random.rand(100) for i in range(1000)}
domain_corpus = ["medical", "patient", "diagnosis", "treatment"]

adaptation = DomainAdaptation()
adapted = adaptation.fine_tune_embeddings(base_embeddings, domain_corpus)
print(f"Adapted {len(adapted)} embeddings for domain")
```

### 3. **Embedding Visualization**

Techniques for visualizing high-dimensional embeddings:

```python
# Conceptual t-SNE visualization for embeddings
class EmbeddingVisualizer:
    """Tools for visualizing embeddings."""
    
    @staticmethod
    def prepare_for_visualization(embeddings, max_words=500):
        """Prepare embeddings for 2D visualization."""
        # Select subset of words for visualization
        word_list = list(embeddings.keys())[:max_words]
        embedding_matrix = np.array([embeddings[word] for word in word_list])
        
        return word_list, embedding_matrix
    
    @staticmethod
    def simulated_tsne(embedding_matrix, n_components=2):
        """Simulate t-SNE dimensionality reduction."""
        # In practice, use sklearn.manifold.TSNE
        n_samples, n_features = embedding_matrix.shape
        
        # Simulate 2D projection
        np.random.seed(42)
        projection = np.random.normal(0, 1, (n_samples, n_components))
        
        return projection
    
    @staticmethod
    def create_visualization_data(words, projections):
        """Create data structure for plotting."""
        viz_data = []
        for i, word in enumerate(words):
            viz_data.append({
                'word': word,
                'x': projections[i, 0],
                'y': projections[i, 1]
            })
        return viz_data

# Example usage
sample_embeddings = {f"word_{i}": np.random.rand(100) for i in range(50)}

visualizer = EmbeddingVisualizer()
words, matrix = visualizer.prepare_for_visualization(sample_embeddings)
projections = visualizer.simulated_tsne(matrix)
viz_data = visualizer.create_visualization_data(words, projections)

print(f"Prepared {len(viz_data)} words for visualization")
print(f"Sample point: {viz_data[0]}")
```

### 4. **Embedding Quality Assessment**

```python
class EmbeddingQualityAssessment:
    """Methods for assessing embedding quality."""
    
    @staticmethod
    def intrinsic_evaluation(embeddings):
        """Evaluate embeddings using intrinsic measures."""
        metrics = {}
        
        # 1. Average pairwise distance
        distances = []
        words = list(embeddings.keys())
        
        for i in range(min(100, len(words))):  # Sample for efficiency
            for j in range(i+1, min(100, len(words))):
                vec1, vec2 = embeddings[words[i]], embeddings[words[j]]
                distance = np.linalg.norm(vec1 - vec2)
                distances.append(distance)
        
        metrics['avg_pairwise_distance'] = np.mean(distances)
        
        # 2. Dimensionality utilization
        embedding_matrix = np.array(list(embeddings.values()))
        variance_per_dim = np.var(embedding_matrix, axis=0)
        metrics['effective_dimensions'] = np.sum(variance_per_dim > 0.01)
        
        # 3. Isotropy (how evenly distributed in space)
        # Simplified: standard deviation of vector norms
        norms = [np.linalg.norm(vec) for vec in embeddings.values()]
        metrics['norm_std'] = np.std(norms)
        
        return metrics
    
    @staticmethod
    def semantic_coherence_test(embeddings, word_categories):
        """Test semantic coherence within categories."""
        coherence_scores = {}
        
        for category, words in word_categories.items():
            # Get embeddings for words in category
            category_embeddings = []
            for word in words:
                if word in embeddings:
                    category_embeddings.append(embeddings[word])
            
            if len(category_embeddings) > 1:
                # Compute average pairwise similarity within category
                similarities = []
                for i in range(len(category_embeddings)):
                    for j in range(i+1, len(category_embeddings)):
                        sim = cosine_similarity(
                            category_embeddings[i], 
                            category_embeddings[j]
                        )
                        similarities.append(sim)
                
                coherence_scores[category] = np.mean(similarities)
        
        return coherence_scores

# Example usage
test_embeddings = {
    "king": np.array([1.0, 0.8, 0.2]),
    "queen": np.array([0.9, 0.9, 0.3]),
    "man": np.array([0.8, 0.1, 0.7]),
    "woman": np.array([0.7, 0.2, 0.8]),
    "dog": np.array([0.2, 0.3, 0.9]),
    "cat": np.array([0.3, 0.4, 0.8])
}

word_categories = {
    "royalty": ["king", "queen"],
    "gender": ["man", "woman"],
    "animals": ["dog", "cat"]
}

assessor = EmbeddingQualityAssessment()
quality_metrics = assessor.intrinsic_evaluation(test_embeddings)
coherence = assessor.semantic_coherence_test(test_embeddings, word_categories)

print(f"Quality metrics: {quality_metrics}")
print(f"Semantic coherence: {coherence}")
```

## Conclusion

Embeddings represent one of the most significant breakthroughs in Natural Language Processing, transforming how machines understand and process human language. This comprehensive guide has covered:

### Key Takeaways:

1. **Fundamental Concept**: Embeddings map discrete linguistic units to dense vector spaces where semantic relationships are preserved geometrically.

2. **Evolution**: From simple one-hot encodings to sophisticated contextual embeddings like BERT and GPT.

3. **Types**: Static (Word2Vec, GloVe), contextual (BERT, ELMo), and specialized embeddings for different granularities.

4. **Applications**: Information retrieval, recommendation systems, machine translation, sentiment analysis, and many more.

5. **Best Practices**: Proper preprocessing, dimension selection, evaluation methods, and handling of edge cases.

### Future Directions:

- **Multimodal Embeddings**: Combining text with images, audio, and other modalities
- **Efficient Architectures**: Reducing computational requirements while maintaining quality
- **Multilingual Understanding**: Better cross-lingual representations
- **Domain Specialization**: Tailored embeddings for specific fields and applications

Understanding embeddings is crucial for anyone working in NLP, as they form the foundation for virtually all modern language understanding systems. Whether you're building chatbots, search engines, or language models, embeddings provide the mathematical framework that enables machines to work with human language effectively.

The examples and implementations provided in this guide offer both theoretical understanding and practical tools for working with embeddings in real-world applications. As the field continues to evolve, these fundamental concepts will remain essential for building more sophisticated and capable NLP systems.
