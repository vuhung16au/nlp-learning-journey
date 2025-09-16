# NLP Definitions and Glossary

A comprehensive glossary of terms, concepts, and definitions used in Natural Language Processing, from basic text processing to advanced deep learning techniques.

## Table of Contents

1. [Basic Text Processing](#basic-text-processing)
2. [Linguistic Concepts](#linguistic-concepts)
3. [Statistical NLP](#statistical-nlp)
4. [Machine Learning for NLP](#machine-learning-for-nlp)
5. [Neural Networks](#neural-networks)
6. [Word Embeddings](#word-embeddings)
7. [Transformers and Attention](#transformers-and-attention)
8. [Language Models](#language-models)
9. [NLP Tasks](#nlp-tasks)
10. [Evaluation Metrics](#evaluation-metrics)
11. [Advanced Concepts](#advanced-concepts)

## Basic Text Processing

**Corpus (plural: Corpora)**
A large collection of written or spoken texts used for linguistic analysis and machine learning training.

**Document**
A single unit of text, such as an article, email, or social media post.

**Token**
The basic unit of text processing, typically a word, punctuation mark, or special symbol after tokenization.

**Tokenization**
The process of breaking down text into individual tokens (words, subwords, or characters).

**Type**
A unique token in a vocabulary. The number of types is always less than or equal to the number of tokens.

**Vocabulary (Vocab)**
The set of all unique words or tokens known to a model or system.

**Stemming**
Reducing words to their root form by removing suffixes (e.g., "running" → "run").

**Lemmatization**
Reducing words to their base or dictionary form considering context and part of speech (e.g., "better" → "good").

**Stop Words**
Common words (like "the", "and", "is") that are often filtered out because they carry little semantic meaning.

**N-gram**
A contiguous sequence of n items from text. Unigrams (1-gram), bigrams (2-gram), trigrams (3-gram), etc.

**Bag of Words (BoW)**
A text representation that ignores word order and grammar, treating text as a collection of words.

**Term Frequency (TF)**
The number of times a term appears in a document, often normalized by document length.

**Inverse Document Frequency (IDF)**
A measure of how much information a word provides across a collection of documents.

**TF-IDF**
Term Frequency-Inverse Document Frequency; a numerical statistic reflecting how important a word is to a document in a collection.

## Linguistic Concepts

**Morphology**
The study of word structure and formation, including prefixes, suffixes, and roots.

**Syntax**
The rules governing the structure of sentences and how words combine to form phrases and sentences.

**Semantics**
The study of meaning in language, including word meanings and sentence meanings.

**Pragmatics**
The study of how context affects meaning and how language is used in real situations.

**Phonology**
The study of sound patterns in language.

**Part-of-Speech (POS)**
Grammatical categories like noun, verb, adjective, adverb, etc.

**Named Entity**
Real-world objects that can be denoted with a proper name, such as people, locations, organizations.

**Dependency**
Grammatical relationships between words in a sentence, often represented as a tree structure.

**Constituency**
How words group together to form phrases and clauses in hierarchical structures.

**Anaphora**
Reference to a previously mentioned entity in text (e.g., pronouns referring to nouns).

**Coreference**
When two or more expressions refer to the same entity.

**Ambiguity**
When a word, phrase, or sentence has multiple possible meanings.

**Polysemy**
A single word having multiple related meanings.

**Homonymy**
Words that share the same spelling or pronunciation but have different meanings.

**Synonymy**
Words with similar or identical meanings.

**Antonymy**
Words with opposite meanings.

## Statistical NLP

**Language Model**
A statistical model that assigns probabilities to sequences of words.

**Markov Assumption**
The assumption that the probability of a word depends only on a limited number of previous words.

**Maximum Likelihood Estimation (MLE)**
A method for estimating model parameters by maximizing the likelihood of observed data.

**Smoothing**
Techniques to handle zero probabilities in statistical models, such as Laplace smoothing.

**Perplexity**
A measure of how well a language model predicts text; lower perplexity indicates better performance.

**Cross-Entropy**
A measure of the difference between two probability distributions, often used as a loss function.

**Entropy**
A measure of uncertainty or randomness in information content.

**Mutual Information**
A measure of the mutual dependence between two variables.

**Pointwise Mutual Information (PMI)**
A measure of association between two specific outcomes of random variables.

**Chi-square Test**
A statistical test for independence between categorical variables.

**Expectation-Maximization (EM)**
An iterative algorithm for finding maximum likelihood estimates in models with latent variables.

## Machine Learning for NLP

**Feature**
An individual measurable property of an observed phenomenon.

**Feature Engineering**
The process of selecting and transforming variables for machine learning models.

**Feature Vector**
A numerical representation of an object's features.

**Classification**
Predicting discrete categories or classes for input data.

**Regression**
Predicting continuous numerical values.

**Clustering**
Grouping similar objects together without predefined categories.

**Supervised Learning**
Learning with labeled training data.

**Unsupervised Learning**
Learning patterns from data without labels.

**Semi-supervised Learning**
Learning with a combination of labeled and unlabeled data.

**Training Set**
Data used to train a machine learning model.

**Validation Set**
Data used to tune hyperparameters and validate model performance during training.

**Test Set**
Data used to evaluate final model performance.

**Overfitting**
When a model performs well on training data but poorly on new data.

**Underfitting**
When a model is too simple to capture underlying patterns.

**Cross-Validation**
A technique for assessing model performance by partitioning data into subsets.

**Regularization**
Techniques to prevent overfitting by adding constraints or penalties to models.

**Hyperparameter**
Configuration settings for learning algorithms that are set before training.

## Neural Networks

**Artificial Neural Network (ANN)**
A computing system inspired by biological neural networks.

**Neuron/Unit**
A basic processing unit in a neural network that applies a function to its inputs.

**Layer**
A collection of neurons that process inputs in parallel.

**Hidden Layer**
Layers between input and output layers that learn intermediate representations.

**Activation Function**
A function that determines the output of a neuron given its inputs.

**ReLU (Rectified Linear Unit)**
An activation function: f(x) = max(0, x).

**Sigmoid**
An activation function: f(x) = 1/(1 + e^(-x)).

**Tanh**
An activation function: f(x) = (e^x - e^(-x))/(e^x + e^(-x)).

**Softmax**
A function that converts a vector of numbers into a probability distribution.

**Backpropagation**
An algorithm for training neural networks by computing gradients through the chain rule.

**Gradient Descent**
An optimization algorithm that iteratively adjusts parameters to minimize a loss function.

**Learning Rate**
A hyperparameter controlling the step size in gradient descent.

**Epoch**
One complete pass through the entire training dataset.

**Batch**
A subset of training data processed together in one iteration.

**Dropout**
A regularization technique that randomly sets some neurons to zero during training.

**Weight Decay**
A regularization technique that penalizes large weights.

## Word Embeddings

**Word Embedding**
Dense vector representations of words that capture semantic relationships.

**Word2Vec**
A neural network approach for learning word embeddings using skip-gram or CBOW.

**Skip-gram**
A Word2Vec variant that predicts context words from a target word.

**CBOW (Continuous Bag of Words)**
A Word2Vec variant that predicts a target word from context words.

**GloVe (Global Vectors)**
A word embedding method that combines global statistical information with local context.

**FastText**
An extension of Word2Vec that considers subword information.

**Subword**
Parts of words, such as character n-grams or morphological units.

**Embedding Dimension**
The size of the vector representing each word.

**Semantic Similarity**
The degree to which words have similar meanings.

**Analogical Reasoning**
The ability to complete word analogies (e.g., king - man + woman = queen).

**Contextualized Embeddings**
Word representations that change based on the surrounding context.

**Static Embeddings**
Word representations that remain the same regardless of context.

## Transformers and Attention

**Attention Mechanism**
A technique that allows models to focus on relevant parts of the input.

**Self-Attention**
Attention applied within a single sequence, allowing positions to attend to each other.

**Multi-Head Attention**
Multiple attention mechanisms applied in parallel to capture different types of relationships.

**Query, Key, Value**
Three vectors used in attention mechanisms to compute attention weights and outputs.

**Attention Weights**
Scores indicating how much focus to place on each input position.

**Transformer**
A neural network architecture based entirely on attention mechanisms.

**Encoder**
The part of a transformer that processes input sequences.

**Decoder**
The part of a transformer that generates output sequences.

**Position Encoding**
A method for incorporating positional information into transformer models.

**Layer Normalization**
A normalization technique applied within neural network layers.

**Residual Connection**
Skip connections that add the input of a layer to its output.

**Feed-Forward Network**
Fully connected layers applied pointwise in transformer blocks.

## Language Models

**Autoregressive Model**
A model that generates sequences one token at a time, conditioning on previous tokens.

**Masked Language Model**
A model trained to predict masked tokens using bidirectional context.

**BERT (Bidirectional Encoder Representations from Transformers)**
A pre-trained transformer model using masked language modeling.

**GPT (Generative Pre-trained Transformer)**
An autoregressive transformer model for text generation.

**T5 (Text-to-Text Transfer Transformer)**
A model that treats all NLP tasks as text-to-text problems.

**RoBERTa**
An optimized version of BERT with improved training procedures.

**ALBERT**
A lightweight version of BERT with parameter sharing and factorized embeddings.

**Pre-training**
Training a model on a large corpus for general language understanding.

**Fine-tuning**
Adapting a pre-trained model to a specific task with task-specific data.

**Transfer Learning**
Using knowledge gained from one task to improve performance on another task.

**Zero-shot Learning**
Performing tasks without task-specific training examples.

**Few-shot Learning**
Learning from a small number of examples.

**In-context Learning**
Learning to perform tasks from examples provided in the input prompt.

## NLP Tasks

**Text Classification**
Categorizing text into predefined classes or categories.

**Sentiment Analysis**
Determining the emotional tone or opinion expressed in text.

**Named Entity Recognition (NER)**
Identifying and classifying named entities in text.

**Part-of-Speech Tagging**
Assigning grammatical categories to words.

**Dependency Parsing**
Analyzing grammatical relationships between words.

**Constituency Parsing**
Analyzing the hierarchical structure of sentences.

**Coreference Resolution**
Determining which expressions refer to the same entity.

**Machine Translation**
Automatically translating text from one language to another.

**Question Answering**
Automatically answering questions based on given context.

**Text Summarization**
Creating a shorter version of text while preserving key information.

**Text Generation**
Automatically producing human-like text.

**Information Extraction**
Extracting structured information from unstructured text.

**Relation Extraction**
Identifying relationships between entities in text.

**Event Extraction**
Identifying events and their participants from text.

**Text Similarity**
Measuring how similar two pieces of text are.

**Paraphrase Detection**
Determining if two texts express the same meaning.

**Natural Language Inference (NLI)**
Determining logical relationships between text pairs.

**Reading Comprehension**
Understanding and answering questions about a passage.

**Dialogue Systems**
Computer systems designed to converse with humans.

**Chatbot**
A computer program designed to simulate conversation.

## Evaluation Metrics

**Accuracy**
The proportion of correct predictions among all predictions.

**Precision**
The proportion of true positive predictions among all positive predictions.

**Recall**
The proportion of true positive predictions among all actual positives.

**F1-Score**
The harmonic mean of precision and recall.

**BLEU (Bilingual Evaluation Understudy)**
A metric for evaluating machine translation quality.

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**
A set of metrics for evaluating text summarization.

**Exact Match**
A strict metric requiring predictions to match references exactly.

**METEOR**
A metric for machine translation that considers synonyms and paraphrases.

**CIDEr**
A metric for image captioning that uses TF-IDF weights.

**BERTScore**
An evaluation metric using BERT embeddings to measure semantic similarity.

**Human Evaluation**
Assessment of model outputs by human judges.

**Inter-annotator Agreement**
The degree to which different annotators agree on labels.

**Cohen's Kappa**
A statistic measuring inter-annotator agreement.

**Krippendorff's Alpha**
A reliability coefficient for agreement among multiple annotators.

## Advanced Concepts

**Attention Head**
Individual attention mechanisms in multi-head attention.

**Gradient Clipping**
A technique to prevent exploding gradients by limiting gradient magnitudes.

**Learning Rate Scheduling**
Adjusting the learning rate during training.

**Warm-up**
Gradually increasing the learning rate at the beginning of training.

**Knowledge Distillation**
Training a smaller model to mimic a larger, more complex model.

**Model Compression**
Techniques to reduce model size while maintaining performance.

**Quantization**
Reducing the precision of model weights to decrease memory usage.

**Pruning**
Removing less important neurons or connections from neural networks.

**Adversarial Examples**
Inputs designed to fool machine learning models.

**Adversarial Training**
Training models with adversarial examples to improve robustness.

**Data Augmentation**
Techniques to artificially increase training data diversity.

**Active Learning**
Selecting the most informative examples for annotation.

**Curriculum Learning**
Training models by gradually increasing task difficulty.

**Multi-task Learning**
Training a single model on multiple related tasks simultaneously.

**Meta-learning**
Learning to learn; algorithms that can quickly adapt to new tasks.

**Domain Adaptation**
Adapting models trained on one domain to work on another domain.

**Out-of-Domain**
Data or tasks that differ from the training distribution.

**Out-of-Vocabulary (OOV)**
Words not present in the training vocabulary.

**Catastrophic Forgetting**
When a model forgets previously learned information while learning new tasks.

**Continual Learning**
Learning new tasks without forgetting previous ones.

**Federated Learning**
Training models across decentralized data sources without sharing raw data.

**Differential Privacy**
Techniques to provide privacy guarantees when training on sensitive data.

**Explainable AI (XAI)**
Methods to make AI model decisions interpretable and understandable.

**Attention Visualization**
Techniques to visualize what parts of input the model focuses on.

**Probing**
Analyzing what linguistic information neural networks learn.

**Bias in NLP**
Systematic errors or prejudices in language models and datasets.

**Fairness**
Ensuring equitable treatment across different groups in AI systems.

**Multilingual NLP**
Processing and understanding multiple languages.

**Cross-lingual Transfer**
Using knowledge from one language to improve performance in another.

**Code-switching**
Mixing languages within a conversation or text.

**Low-resource Languages**
Languages with limited digital text and NLP resources.

**Computational Linguistics**
The scientific study of language using computational methods.

**Psycholinguistics**
The study of psychological processes involved in language use.

**Sociolinguistics**
The study of how language varies and changes in social contexts.

**Digital Humanities**
The application of computational tools to humanities research.

This comprehensive glossary provides essential definitions for understanding the field of Natural Language Processing, from fundamental concepts to cutting-edge research areas.