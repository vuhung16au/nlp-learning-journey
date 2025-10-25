# spaCy Linguistic Features Tutorial Series

A comprehensive collection of Jupyter notebooks covering spaCy's linguistic features and capabilities, following GitHub best practices for documentation and reproducibility. Includes automated setup and testing with Makefile support.

## 🚀 Quick Start

```bash
# Clone or download this repository
cd examples/spaCy-Linguistic

# Set up environment and install dependencies
make setup-env

# Test all notebooks to ensure everything works
make test-notebooks

# Start learning with the first notebook
jupyter notebook spaCy-Linguistic-Introduction-to-spaCy-Basics.ipynb
```

## 📚 Overview

This tutorial series provides hands-on experience with spaCy's powerful natural language processing capabilities. Each notebook focuses on specific aspects of linguistic analysis, from basic text processing to advanced features and real-world applications.

## Notebook Overview

### 1. Introduction to spaCy Basics (`spaCy-Linguistic-Introduction-to-spaCy-Basics.ipynb`)
**Duration: 30-45 minutes**

Learn the fundamentals of spaCy:
- Installation and setup
- Creating Doc objects
- Basic text processing
- Part-of-speech tagging
- Lemmatization

**Key Topics:**
- spaCy installation and language models
- Understanding Doc objects and tokens
- POS tagging with Universal Dependencies
- Word lemmatization and normalization
- Basic text preprocessing

### 2. Understanding Text Structure (`spaCy-Linguistic-Understanding-Text-Structure.ipynb`)
**Duration: 45-60 minutes**

Dive deeper into text analysis:
- Tokenization fundamentals
- Sentence segmentation
- Morphological analysis
- Word properties and features

**Key Topics:**
- Advanced tokenization techniques
- Handling special cases (URLs, emails, contractions)
- Sentence boundary detection
- Morphological feature extraction
- Understanding word structure

### 3. Syntactic Analysis (`spaCy-Linguistic-Syntactic-Analysis.ipynb`)
**Duration: 45-60 minutes**

Master dependency parsing:
- Dependency parsing basics
- Navigating parse trees
- Noun chunks extraction
- Dependency visualization

**Key Topics:**
- Understanding dependency relationships
- Parse tree traversal
- Noun phrase extraction
- Using displacy for visualization
- Syntactic pattern matching

### 4. Named Entity Recognition (`spaCy-Linguistic-Named-Entity-Recognition.ipynb`)
**Duration: 45-60 minutes**

Extract structured information:
- NER fundamentals
- Working with entity types
- Entity visualization
- Custom entity rules

**Key Topics:**
- Built-in entity types (PERSON, ORG, GPE, etc.)
- Entity extraction and filtering
- NER visualization with displacy
- Custom entity recognition
- Information extraction applications

### 5. Advanced Features (`spaCy-Linguistic-Advanced-Features.ipynb`)
**Duration: 60+ minutes**

Explore advanced spaCy capabilities:
- Word vectors and similarity
- Custom tokenization
- Pipeline customization
- Custom components

**Key Topics:**
- Word embeddings and similarity
- Custom tokenizer patterns
- Pipeline modification
- Creating custom components
- Performance optimization

### 6. Practical Mini-Projects (`spaCy-Linguistic-Practical-Mini-Projects.ipynb`)
**Duration: 60+ minutes**

Apply your knowledge to real projects:
- Text summarization helper
- Information extraction system
- Text preprocessing pipeline

**Key Topics:**
- Building complete NLP workflows
- Information extraction from unstructured text
- Text preprocessing best practices
- Real-world application development

## Prerequisites

- Basic Python knowledge
- Understanding of natural language processing concepts
- Jupyter Notebook environment
- Python 3.9 or higher
- Virtual environment support (recommended)

## Requirements

This tutorial series requires the following packages, all specified in `requirements.txt`:

### Core Dependencies
- **spacy** (>=3.7.0): Main NLP library
- **numpy** (>=1.21.0): Numerical computing
- **pandas** (>=1.3.0): Data manipulation
- **matplotlib** (>=3.5.0): Plotting and visualization
- **seaborn** (>=0.11.0): Statistical visualization
- **jupyter** (>=1.0.0): Jupyter notebook support
- **ipywidgets** (>=7.6.0): Interactive widgets
- **nltk** (>=3.7.0): Natural language toolkit
- **regex** (>=2022.1.18): Regular expressions

### Development Dependencies
- **pytest** (>=6.2.0): Testing framework
- **black** (>=22.0.0): Code formatting
- **flake8** (>=4.0.0): Code linting

### Optional Dependencies
- **transformers** (>=4.20.0): For transformer models
- **torch** (>=1.12.0): For PyTorch models
- **scikit-learn** (>=1.1.0): For machine learning features

## Installation

### Prerequisites
- Python 3.9 or higher
- `uv` package manager (recommended for fast installation)
- Virtual environment support
- Make (for automated setup)

### 🚀 Quick Setup with Makefile (Recommended)

The easiest way to set up the environment is using the provided Makefile:

```bash
# Set up virtual environment and install all dependencies
make setup-env

# Test all notebooks to ensure everything works
make test-notebooks

# Show all available commands
make help
```

The Makefile automatically:
- Creates a Python 3.9 virtual environment
- Installs all dependencies using `uv pip`
- Downloads the required spaCy language model (`en_core_web_sm`)
- Provides error checking and feedback

### Manual Setup (Alternative)

If you prefer manual setup:

1. **Create a Python 3.9 virtual environment:**
   ```bash
   python3.9 -m venv .venv
   
   # Activate the virtual environment
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate     # On Windows
   ```

2. **Install dependencies using uv (recommended):**
   ```bash
   # Install uv if you haven't already
   pip install uv
   
   # Install all requirements
   uv pip install -r requirements.txt
   ```

3. **Download spaCy language models:**
   ```bash
   # Download English language model (small, fast)
   python -m spacy download en_core_web_sm
   ```

### Manual Installation (without virtual environment)

```bash
# Install spaCy and core dependencies
pip install spacy numpy pandas matplotlib seaborn jupyter ipywidgets nltk regex

# Download English language model
python -m spacy download en_core_web_sm

# For advanced features, you might want the medium model
python -m spacy download en_core_web_md
```

### Development Dependencies

For development and testing, install additional packages:
```bash
pip install pytest black flake8
```

## Virtual Environment Setup

### Creating the Virtual Environment

The tutorial includes a pre-configured virtual environment setup:

```bash
# Navigate to the spaCy-Linguistic directory
cd examples/spaCy-Linguistic

# Create Python 3.9 virtual environment
/opt/homebrew/bin/python3.9 -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows

# Install dependencies using uv (fastest method)
uv pip install -r requirements.txt

# Download spaCy language models
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
```

### Activating the Virtual Environment

Before working with the notebooks, always activate the virtual environment:

```bash
# On macOS/Linux
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

### Deactivating the Virtual Environment

When you're done working:

```bash
deactivate
```

## Available Commands

The project includes a Makefile with convenient commands for development:

### Makefile Commands

```bash
# Show all available commands
make help

# Set up virtual environment and install dependencies
make setup-env

# Test all Jupyter notebooks for errors
make test-notebooks

# Clean up virtual environment and temporary files
make clean
```

### Command Details

- **`make setup-env`**: Creates Python 3.9 virtual environment, installs dependencies, and downloads spaCy models
- **`make test-notebooks`**: Executes all notebooks to verify they work correctly
- **`make clean`**: Removes virtual environment and cleans up temporary files
- **`make help`**: Shows all available commands

## Usage

### Quick Start
1. **Set up the environment**: `make setup-env`
2. **Test everything works**: `make test-notebooks`
3. **Start with Notebook 1** for beginners
4. **Progress through notebooks sequentially**
5. **Complete the practice exercises** in each notebook
6. **Apply your learning** in the final projects notebook

### Manual Usage
1. **Activate the virtual environment**: `source .venv/bin/activate`
2. **Start Jupyter**: `jupyter notebook` or `jupyter lab`
3. **Open the notebooks** in order
4. **Run all cells** to see the examples in action

## Key Learning Path

### Beginner Path (Must Cover)
- Notebook 1: Introduction to spaCy Basics
- Notebook 2: Understanding Text Structure
- Notebook 3: Syntactic Analysis
- Notebook 4: Named Entity Recognition

### Advanced Path (Nice to Have)
- Notebook 5: Advanced Features
- Notebook 6: Practical Mini-Projects

## Features Covered

### Core spaCy Features
- ✅ Tokenization and sentence segmentation
- ✅ Part-of-speech tagging
- ✅ Lemmatization and morphological analysis
- ✅ Dependency parsing
- ✅ Named entity recognition
- ✅ Text visualization

### Advanced Features
- ✅ Word vectors and similarity
- ✅ Custom tokenization rules
- ✅ Pipeline customization
- ✅ Custom components
- ✅ Performance optimization

### Practical Applications
- ✅ Text preprocessing pipelines
- ✅ Information extraction
- ✅ Text summarization
- ✅ Entity-based analysis
- ✅ Syntactic pattern matching

## Common Use Cases

1. **Text Preprocessing**: Clean and normalize text for machine learning
2. **Information Extraction**: Extract structured data from unstructured text
3. **Text Analysis**: Understand text structure and meaning
4. **Named Entity Recognition**: Identify people, places, organizations
5. **Dependency Parsing**: Understand grammatical relationships
6. **Text Visualization**: Create interactive visualizations

## Tips for Success

1. **Run all code cells**: Don't just read, execute the code
2. **Experiment**: Modify examples and see what happens
3. **Practice exercises**: Complete all exercises in each notebook
4. **Real data**: Try the techniques on your own text data
5. **Documentation**: Refer to spaCy's official documentation


## Next Steps

After completing this tutorial series:

1. **Explore other languages**: Try spaCy models for different languages
2. **Advanced topics**: Learn about custom model training
3. **Integration**: Combine spaCy with other NLP libraries
4. **Real projects**: Apply your knowledge to actual NLP projects
5. **Community**: Contribute to open-source NLP projects

## Resources

- [spaCy Documentation](https://spacy.io/)
- [spaCy Models](https://spacy.io/models)
- [Universal Dependencies](https://universaldependencies.org/)
- [Natural Language Processing with Python](https://www.nltk.org/book/)

---

**Happy Learning!** 🚀

This tutorial series will give you a solid foundation in spaCy and natural language processing. Start with the basics and gradually work your way up to advanced features and real-world applications.
