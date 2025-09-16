# Chunking in Natural Language Processing (NLP)

Chunking is a fundamental technique in Natural Language Processing that involves grouping consecutive words or tokens into meaningful units called "chunks." These chunks represent linguistic phrases or segments that carry semantic or syntactic significance, making text analysis more structured and interpretable.

## Table of Contents

1. [What is Chunking?](#what-is-chunking)
2. [Types of Chunking](#types-of-chunking)
3. [Chunking vs. Other NLP Techniques](#chunking-vs-other-nlp-techniques)
4. [Technical Approaches](#technical-approaches)
5. [Implementation with Popular Libraries](#implementation-with-popular-libraries)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Real-World Applications](#real-world-applications)
8. [Advanced Techniques](#advanced-techniques)
9. [Best Practices](#best-practices)
10. [Further Reading](#further-reading)

## What is Chunking?

**Chunking** (also known as shallow parsing or phrase chunking) is the process of identifying and grouping sequences of words that form syntactic units, such as noun phrases (NP), verb phrases (VP), or prepositional phrases (PP). Unlike full parsing, chunking focuses on finding non-overlapping, non-recursive phrases that capture important linguistic structures without building complete parse trees.

### Core Concepts:

**Phrase Identification**: Chunking identifies meaningful word groups that function as single units in a sentence.

**Non-recursive**: Chunks do not contain nested structures; they represent flat, non-overlapping segments.

**Syntactic Foundation**: Chunks are based on syntactic relationships and grammatical rules rather than just semantic similarity.

### Example:
```
Input: "The quick brown fox jumps over the lazy dog"
Chunked Output: 
[NP: The quick brown fox] [VP: jumps] [PP: over] [NP: the lazy dog]
```

## Types of Chunking

### 1. Syntactic Chunking

**Noun Phrase (NP) Chunking**
- Identifies noun phrases: determiners, adjectives, and nouns that form a unit
- Example: "the tall building" → [NP: the tall building]

**Verb Phrase (VP) Chunking**
- Identifies verb phrases: main verbs with auxiliaries and modifiers
- Example: "has been running" → [VP: has been running]

**Prepositional Phrase (PP) Chunking**
- Identifies prepositional phrases: preposition + noun phrase
- Example: "in the morning" → [PP: in the morning]

### 2. Semantic Chunking

**Named Entity Chunking**
- Groups words that form named entities (persons, organizations, locations)
- Example: "New York City" → [LOCATION: New York City]

**Topic-based Chunking**
- Groups sentences or paragraphs by thematic similarity
- Used in document segmentation and topic modeling

### 3. Text Segmentation Chunking

**Sentence Chunking**
- Divides text into sentence-level chunks
- Essential for document processing and analysis

**Document Chunking**
- Breaks large documents into manageable segments
- Critical for processing long texts in language models

## Chunking vs. Other NLP Techniques

### Chunking vs. Full Parsing
| Aspect | Chunking | Full Parsing |
|--------|----------|--------------|
| **Complexity** | Shallow, non-recursive | Deep, hierarchical |
| **Speed** | Fast | Slower |
| **Output** | Flat phrase structures | Complete parse trees |
| **Accuracy** | High for identified chunks | Variable, more complex |

### Chunking vs. Named Entity Recognition (NER)
| Aspect | Chunking | NER |
|--------|----------|-----|
| **Focus** | Syntactic phrases | Semantic entities |
| **Types** | NP, VP, PP, etc. | PERSON, ORG, LOC, etc. |
| **Overlap** | Can complement NER | Can use chunking results |

### Chunking vs. Tokenization
| Aspect | Chunking | Tokenization |
|--------|----------|--------------|
| **Granularity** | Phrase-level | Word/subword-level |
| **Purpose** | Syntactic grouping | Text segmentation |
| **Input** | Tokenized text | Raw text |

## Technical Approaches

### 1. Rule-Based Chunking

Uses hand-crafted rules and regular expressions to identify chunks based on POS tags and linguistic patterns.

**Advantages:**
- Interpretable and controllable
- Fast execution
- Domain-specific customization

**Disadvantages:**
- Requires linguistic expertise
- Limited coverage
- Language-specific rules

### 2. Statistical Chunking

Employs machine learning models trained on annotated corpora to predict chunk boundaries.

**Common Models:**
- Hidden Markov Models (HMM)
- Conditional Random Fields (CRF)
- Support Vector Machines (SVM)

### 3. Neural Chunking

Uses deep learning models to learn chunk patterns from data automatically.

**Popular Architectures:**
- Recurrent Neural Networks (RNN/LSTM)
- Bidirectional LSTM (BiLSTM)
- Transformer-based models

### 4. IOB Tagging Scheme

Chunking is often formulated as a sequence labeling task using the IOB (Inside-Outside-Begin) format:

- **B-**: Beginning of a chunk
- **I-**: Inside a chunk
- **O**: Outside any chunk

**Example:**
```
Word:  The  quick  brown  fox  jumps  over  the  lazy  dog
Tag:   B-NP I-NP  I-NP   I-NP B-VP   B-PP  B-NP I-NP  I-NP
```

## Implementation with Popular Libraries

### 1. NLTK (Natural Language Toolkit)

```python
import nltk
from nltk import word_tokenize, pos_tag
from nltk.chunk import ne_chunk, RegexpParser

# Download required data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

def nltk_chunking_example():
    # Sample text
    text = "The quick brown fox jumps over the lazy dog in New York"
    
    # Tokenize and POS tag
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    
    # Named Entity Chunking
    named_entities = ne_chunk(pos_tags)
    print("Named Entity Chunks:")
    print(named_entities)
    
    # Rule-based Noun Phrase Chunking
    grammar = r"""
        NP: {<DT|PP\$>?<JJ>*<NN>}   # Noun phrase
        PP: {<IN><NP>}               # Prepositional phrase
        VP: {<VB.*><NP|PP|CLAUSE>+$} # Verb phrase
    """
    
    cp = RegexpParser(grammar)
    result = cp.parse(pos_tags)
    print("\nRule-based Chunks:")
    result.draw()  # Visualize the tree
    
    return result

# Run the example
chunks = nltk_chunking_example()
```

### 2. spaCy

```python
import spacy

def spacy_chunking_example():
    # Load English model
    nlp = spacy.load("en_core_web_sm")
    
    text = "The quick brown fox jumps over the lazy dog in New York"
    doc = nlp(text)
    
    # Noun Phrases
    print("Noun Phrases:")
    for chunk in doc.noun_chunks:
        print(f"  {chunk.text} --> {chunk.label_}")
    
    # Named Entities
    print("\nNamed Entities:")
    for ent in doc.ents:
        print(f"  {ent.text} --> {ent.label_}")
    
    # Custom chunking with dependency parsing
    print("\nCustom Phrase Chunks:")
    for token in doc:
        if token.dep_ == "ROOT":  # Find main verb
            # Get verb phrase
            verb_phrase = [token.text]
            for child in token.children:
                if child.dep_ in ["aux", "auxpass", "neg"]:
                    verb_phrase.append(child.text)
            print(f"  Verb Phrase: {' '.join(verb_phrase)}")

# Run the example
spacy_chunking_example()
```

### 3. Sklearn with Custom Features

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np

def sklearn_chunking_example():
    # Prepare training data (IOB format)
    # This is a simplified example - real datasets would be much larger
    sentences = [
        [("The", "DT", "B-NP"), ("quick", "JJ", "I-NP"), ("fox", "NN", "I-NP"), ("runs", "VB", "B-VP")],
        [("A", "DT", "B-NP"), ("big", "JJ", "I-NP"), ("dog", "NN", "I-NP"), ("sleeps", "VB", "B-VP")]
    ]
    
    def extract_features(sentence, i):
        """Extract features for word at position i"""
        word, pos, _ = sentence[i]
        features = {
            'word': word,
            'pos': pos,
            'word.lower': word.lower(),
            'word.isupper': word.isupper(),
            'word.istitle': word.istitle(),
            'word.isdigit': word.isdigit(),
        }
        
        # Previous word features
        if i > 0:
            prev_word, prev_pos, _ = sentence[i-1]
            features.update({
                'prev_word': prev_word,
                'prev_pos': prev_pos,
            })
        
        # Next word features
        if i < len(sentence) - 1:
            next_word, next_pos, _ = sentence[i+1]
            features.update({
                'next_word': next_word,
                'next_pos': next_pos,
            })
        
        return features
    
    # Prepare training data
    X_train = []
    y_train = []
    
    for sentence in sentences:
        for i in range(len(sentence)):
            features = extract_features(sentence, i)
            X_train.append(features)
            y_train.append(sentence[i][2])  # IOB tag
    
    # Convert features to format suitable for sklearn
    from sklearn.feature_extraction import DictVectorizer
    vectorizer = DictVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    
    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    
    print("Model trained successfully!")
    print(f"Feature names: {len(vectorizer.feature_names_)}")
    
    return model, vectorizer

# Run the example
model, vectorizer = sklearn_chunking_example()
```

### 4. Modern Transformer Approach

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

def transformer_chunking_example():
    # Use a pre-trained NER model for chunking
    # This can be adapted for custom chunking tasks
    
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
    model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
    
    # Create NER pipeline
    ner_pipeline = pipeline("ner", 
                           model=model, 
                           tokenizer=tokenizer,
                           aggregation_strategy="simple")
    
    text = "The quick brown fox jumps over the lazy dog in New York"
    
    # Get predictions
    predictions = ner_pipeline(text)
    
    print("Transformer-based Entity Chunks:")
    for pred in predictions:
        print(f"  {pred['word']} --> {pred['entity_group']} (confidence: {pred['score']:.3f})")
    
    return predictions

# Run the example (requires internet connection for model download)
# predictions = transformer_chunking_example()
```

## Evaluation Metrics

### 1. Precision, Recall, and F1-Score

```python
def evaluate_chunking(true_chunks, predicted_chunks):
    """
    Evaluate chunking performance
    
    Args:
        true_chunks: List of true chunk boundaries
        predicted_chunks: List of predicted chunk boundaries
    
    Returns:
        Dict with precision, recall, and F1 scores
    """
    true_set = set(true_chunks)
    pred_set = set(predicted_chunks)
    
    if len(pred_set) == 0:
        precision = 0.0
    else:
        precision = len(true_set & pred_set) / len(pred_set)
    
    if len(true_set) == 0:
        recall = 0.0
    else:
        recall = len(true_set & pred_set) / len(true_set)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Example usage
true_chunks = [(0, 3), (4, 5), (6, 9)]  # (start, end) positions
pred_chunks = [(0, 3), (4, 5), (7, 9)]  # Slightly different prediction

scores = evaluate_chunking(true_chunks, pred_chunks)
print(f"Precision: {scores['precision']:.3f}")
print(f"Recall: {scores['recall']:.3f}")
print(f"F1-Score: {scores['f1']:.3f}")
```

### 2. Boundary Detection Metrics

- **Exact Match**: Chunks must match exactly
- **Overlap**: Partial overlap counts as correct
- **Boundary Precision/Recall**: Focus on chunk boundaries

## Real-World Applications

### 1. Information Extraction

**Use Case**: Extracting structured information from unstructured text
```python
def extract_company_info(text):
    """Extract company-related information using chunking"""
    # This would use chunking to identify:
    # - Company names (NP chunks + NER)
    # - Financial figures (NP chunks with numbers)
    # - Action phrases (VP chunks)
    pass
```

**Example Applications**:
- Resume parsing for HR systems
- Financial document analysis
- Legal document processing

### 2. Question Answering Systems

**Use Case**: Identifying question and answer phrases
```python
def chunk_qa_pairs(question, context):
    """Use chunking to improve QA performance"""
    # Chunk question to identify:
    # - Question type (what, where, when)
    # - Key noun phrases
    # - Context chunks for answer extraction
    pass
```

### 3. Machine Translation

**Use Case**: Phrase-based translation systems
```python
def phrase_based_translation(source_text, target_language):
    """Use chunking for phrase-based MT"""
    # 1. Chunk source text into phrases
    # 2. Translate each chunk
    # 3. Reorder and combine translations
    pass
```

### 4. Text Summarization

**Use Case**: Identifying important text segments
```python
def extractive_summarization(document):
    """Use chunking for better summarization"""
    # 1. Chunk document into meaningful segments
    # 2. Score chunks by importance
    # 3. Select top-scored chunks for summary
    pass
```

### 5. Document Processing for LLMs

**Use Case**: Preparing long documents for transformer models
```python
def chunk_for_llm(document, max_tokens=512, overlap=50):
    """
    Chunk documents for LLM processing with context preservation
    
    Args:
        document: Input text document
        max_tokens: Maximum tokens per chunk
        overlap: Number of overlapping tokens between chunks
    
    Returns:
        List of document chunks with metadata
    """
    # Intelligent chunking that:
    # 1. Respects sentence boundaries
    # 2. Maintains semantic coherence
    # 3. Preserves context with overlap
    
    chunks = []
    # Implementation would go here
    return chunks
```

## Advanced Techniques

### 1. Hierarchical Chunking

Building multi-level chunk structures:
```python
def hierarchical_chunking(text):
    """
    Create hierarchical chunk structure:
    Document -> Sections -> Paragraphs -> Sentences -> Phrases
    """
    hierarchy = {
        'document': text,
        'sections': [],
        'paragraphs': [],
        'sentences': [],
        'phrases': []
    }
    # Implementation would build this hierarchy
    return hierarchy
```

### 2. Context-Aware Chunking

Chunking that considers surrounding context:
```python
def context_aware_chunking(text, context_window=5):
    """
    Chunk text considering broader context for better accuracy
    """
    # Consider surrounding words/sentences when making chunking decisions
    pass
```

### 3. Multi-Language Chunking

Handling different languages with varying syntactic rules:
```python
def multilingual_chunking(text, language='auto'):
    """
    Perform language-specific chunking
    """
    language_specific_rules = {
        'english': english_chunker,
        'chinese': chinese_chunker,
        'arabic': arabic_chunker,
    }
    # Select appropriate chunking strategy
    pass
```

### 4. Domain-Specific Chunking

Specialized chunking for specific domains:
```python
def biomedical_chunking(text):
    """Chunking specialized for biomedical texts"""
    # Identify medical terms, drug names, symptoms, etc.
    pass

def legal_chunking(text):
    """Chunking specialized for legal documents"""
    # Identify legal entities, clauses, citations, etc.
    pass
```

## Best Practices

### 1. Data Preprocessing
- **Clean text consistently**: Remove or standardize special characters
- **Normalize case**: Convert to lowercase unless case is meaningful
- **Handle abbreviations**: Expand or standardize common abbreviations
- **Token boundary handling**: Ensure proper tokenization before chunking

### 2. Feature Engineering
- **Use linguistic features**: POS tags, dependency relations, named entities
- **Context windows**: Include surrounding word/tag information
- **Orthographic features**: Capitalization, punctuation, word shape
- **Semantic features**: Word embeddings, topic information

### 3. Model Selection
- **Start simple**: Begin with rule-based approaches for baseline
- **Consider domain**: Choose techniques appropriate for your text type
- **Evaluate thoroughly**: Use cross-validation and domain-specific test sets
- **Balance accuracy vs. speed**: Consider deployment requirements

### 4. Error Analysis
- **Analyze failure cases**: Understand where your chunker fails
- **Check boundary errors**: Focus on chunk start/end detection
- **Evaluate chunk types**: Some phrase types may be harder than others
- **Consider annotation consistency**: Ensure training data quality

### 5. Integration with Other NLP Tasks
- **Pipeline design**: Consider how chunking fits in your NLP pipeline
- **Error propagation**: Handle errors from upstream tasks (tokenization, POS tagging)
- **Task synergy**: Use chunking to improve other NLP tasks

## Further Reading

### Academic Papers
- **"Text Chunking using Transformation-Based Learning"** by Lance A. Ramshaw and Mitchell P. Marcus (1995)
- **"Chunking with Support Vector Machines"** by Tong Zhang et al. (2002)
- **"Introduction to the CoNLL-2000 Shared Task: Chunking"** by Erik F. Tjong Kim Sang and Sabine Buchholz (2000)

### Books
- **"Natural Language Processing with Python"** by Steven Bird, Ewan Klein, and Edward Loper
- **"Speech and Language Processing"** by Daniel Jurafsky and James H. Martin
- **"Foundations of Statistical Natural Language Processing"** by Christopher Manning and Hinrich Schütze

### Online Resources
- **NLTK Chunking Tutorial**: [https://www.nltk.org/book/ch07.html](https://www.nltk.org/book/ch07.html)
- **spaCy Linguistic Features**: [https://spacy.io/usage/linguistic-features](https://spacy.io/usage/linguistic-features)
- **CoNLL-2000 Shared Task**: Historical chunking evaluation benchmark

### Related NLP Concepts
- **[Named Entity Recognition (NER)](./NER.md)**: Closely related to semantic chunking
- **[Part-of-Speech Tagging](./tagging.md)**: Essential preprocessing for syntactic chunking
- **[Parsing](./key-concepts.md)**: Full syntactic analysis that builds on chunking
- **[Tokenization](./key-concepts.md)**: Required preprocessing step for chunking

---

*This documentation provides a comprehensive overview of chunking in NLP. For hands-on practice, explore the example notebooks in the `examples/` directory, particularly those focusing on tokenization, POS tagging, and NER, which form the foundation for effective chunking implementations.*