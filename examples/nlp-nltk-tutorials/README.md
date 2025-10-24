# NLP-NLTK Tutorials Collection

A comprehensive collection of 10 hands-on Jupyter notebook tutorials for learning Natural Language Processing (NLP) using the NLTK (Natural Language Toolkit) library. These tutorials are designed to provide practical, step-by-step guidance for essential NLP tasks.

## 📚 Tutorial Series

### Task 01: Text Preprocessing and Cleaning
**File:** `task-01-text-preprocessing.ipynb`  
**Topics:** Lowercase conversion, HTML removal, URL/email filtering, punctuation handling, whitespace normalization, contraction expansion

Learn the fundamentals of cleaning and preparing raw text data for NLP tasks. Essential for any text processing pipeline.

### Task 02: Tokenization Techniques
**File:** `task-02-tokenization.ipynb`  
**Topics:** Word tokenization, sentence tokenization, custom tokenizers, regex tokenization, tweet tokenization, multilingual tokenization

Master different tokenization methods and understand when to use each approach for optimal results.

### Task 03: Stopword Removal and Filtering
**File:** `task-03-stopword-removal.ipynb`  
**Topics:** Built-in stopword lists, custom stopwords, language-specific filtering, frequency-based filtering

Reduce noise in text data by identifying and removing common words that don't carry significant meaning.

### Task 04: Stemming and Lemmatization
**File:** `task-04-stemming-lemmatization.ipynb`  
**Topics:** Porter Stemmer, Lancaster Stemmer, Snowball Stemmer, WordNet Lemmatizer, comparison of techniques

Normalize words to their base or root forms to improve text analysis and reduce dimensionality.

### Task 05: Part-of-Speech Tagging
**File:** `task-05-pos-tagging.ipynb`  
**Topics:** POS tagging, Penn Treebank tagset, Universal Dependencies, multilingual tagging, applications

Identify grammatical roles of words in sentences for syntactic analysis and feature extraction.

### Task 06: Named Entity Recognition
**File:** `task-06-named-entity-recognition.ipynb`  
**Topics:** Entity recognition, ne_chunk function, entity types, custom chunking, multilingual NER

Extract and classify named entities such as persons, organizations, and locations from text.

### Task 07: Text Classification
**File:** `task-07-text-classification.ipynb`  
**Topics:** Feature extraction, Naive Bayes classifier, Decision Trees, training/testing, evaluation metrics

Build machine learning models to automatically categorize text documents into predefined classes.

### Task 08: Sentiment Analysis
**File:** `task-08-sentiment-analysis.ipynb`  
**Topics:** VADER sentiment analyzer, lexicon-based approaches, ML approaches, negation handling, multilingual sentiment

Determine the emotional tone and polarity of text for opinion mining and customer feedback analysis.

### Task 09: Frequency Analysis and Collocations
**File:** `task-09-frequency-collocations.ipynb`  
**Topics:** Word frequency distribution, collocations, bigrams, PMI, chi-square tests, n-gram analysis

Discover patterns and common word combinations in text through statistical analysis.

### Task 10: WordNet and Semantic Similarity
**File:** `task-10-wordnet-similarity.ipynb`  
**Topics:** WordNet synsets, hypernyms/hyponyms, synonyms/antonyms, similarity measures, applications

Explore semantic relationships between words and calculate similarity scores for various NLP applications.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Basic understanding of Python programming

### Installation

1. Navigate to this directory:
```bash
cd examples/nlp-nltk-tutorials
```

2. Install required packages:
```bash
pip install nltk jupyter
```

3. Download NLTK data:
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words')"
```

4. Launch Jupyter:
```bash
jupyter notebook
```

### Running on Google Colab

Each notebook includes a "Open in Colab" badge at the top. Click it to run the notebook directly in Google Colab without any local setup.

## 📖 Learning Path

We recommend following the tutorials in order, as each builds on concepts from previous ones:

1. **Foundation** (Tasks 01-02): Text preprocessing and tokenization
2. **Text Normalization** (Tasks 03-04): Stopword removal and stemming/lemmatization
3. **Linguistic Analysis** (Tasks 05-06): POS tagging and named entity recognition
4. **Machine Learning** (Tasks 07-08): Text classification and sentiment analysis
5. **Statistical Analysis** (Tasks 09-10): Frequency analysis and semantic similarity

## 🌍 Vietnamese/English Focus

These tutorials emphasize Vietnamese-English language pairs, providing parallel examples for:
- Translation tasks
- Cross-lingual NLP
- Southeast Asian language processing

Example pairs used throughout:
- English: "My name is" → Vietnamese: "Tên tôi là"
- English: "Hello" → Vietnamese: "Xin chào"
- English: "Thank you" → Vietnamese: "Cảm ơn"

## 🎯 Key Features

- **Hands-on Code**: Every concept includes practical, runnable code examples
- **Bilingual Examples**: English and Vietnamese text for comparative learning
- **Environment Detection**: Automatic setup for Colab, Kaggle, and local environments
- **Best Practices**: Industry-standard approaches and techniques
- **Real-world Applications**: Practical use cases and scenarios

## 📚 Additional Resources

- [NLTK Official Documentation](https://www.nltk.org/)
- [NLTK Book - Natural Language Processing with Python](https://www.nltk.org/book/)
- [Main Repository Documentation](../../docs/)

## 🤝 Contributing

This is part of the NLP Learning Journey repository. For suggestions or improvements:
1. Open an issue in the main repository
2. Fork and submit a pull request
3. Share your feedback and learning experiences

## 📝 License

These tutorials are part of the nlp-learning-journey repository and are licensed under the MIT License.

## 🙏 Acknowledgments

These tutorials are inspired by:
- The NLTK community and documentation
- Packt Publishing's "Mastering Natural Language Processing with Python"
- Various open-source NLP resources and best practices

---

**Happy Learning!** 🎓

Start with [Task 01: Text Preprocessing](task-01-text-preprocessing.ipynb) and work your way through the series.
