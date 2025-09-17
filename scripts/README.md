# Synthetic Data Generation Scripts

This directory contains scripts for generating synthetic text data for Natural Language Processing (NLP) tasks. The main script `synthetic-data.py` can generate realistic text data in multiple languages with various dataset sizes.

## 📁 Directory Structure

```
scripts/
├── synthetic-data.py    # Main data generation script
├── README.md           # This documentation file
└── .venv/              # Python virtual environment
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Virtual environment support

### Setup

1. **Navigate to the scripts directory:**
   ```bash
   cd scripts/
   ```

2. **Activate the virtual environment:**
   ```bash
   # On Linux/Mac
   source .venv/bin/activate
   
   # On Windows
   .venv\Scripts\activate
   ```

3. **Run the data generation script:**
   ```bash
   python synthetic-data.py
   ```

## 📊 Generated Data Structure

The script generates synthetic text data in the following structure:

```
data/
├── english/
│   ├── small.txt     # 1,000 words
│   ├── medium.txt    # 100,000 words
│   └── large.txt     # 1,000,000 words
├── japanese/
│   ├── small.txt     # 1,000 words
│   ├── medium.txt    # 100,000 words
│   └── large.txt     # 1,000,000 words
└── vietnamese/
    ├── small.txt     # 1,000 words
    ├── medium.txt    # 100,000 words
    └── large.txt     # 1,000,000 words
```

## 🛠️ Usage

### Basic Usage

Generate all datasets for all languages:
```bash
python synthetic-data.py
```

### Advanced Usage

#### Generate data for a specific language:
```bash
python synthetic-data.py --language english
python synthetic-data.py --language japanese
python synthetic-data.py --language vietnamese
```

#### Generate data for a specific language and size:
```bash
python synthetic-data.py --language english --size small
python synthetic-data.py --language japanese --size medium
python synthetic-data.py --language vietnamese --size large
```

#### Generate custom word count:
```bash
python synthetic-data.py --language english --words 5000
```

#### Specify custom output directory:
```bash
python synthetic-data.py --output /path/to/custom/directory
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--language` | Target language (english, japanese, vietnamese) | All languages |
| `--size` | Dataset size (small, medium, large) | All sizes |
| `--words` | Custom word count (requires --language) | N/A |
| `--output` | Output directory path | `../data` |
| `--help` | Show help message | N/A |

## 📝 Data Characteristics

### English Data
- **Content Types**: News articles, product reviews, social media posts
- **Vocabulary**: Common English words across various domains
- **Structure**: Realistic sentence patterns and grammatical structures
- **Topics**: Technology, healthcare, education, entertainment, travel

### Japanese Data
- **Content Types**: Daily conversations, business communications, social interactions
- **Script**: Native Japanese characters (Hiragana, Katakana, Kanji)
- **Vocabulary**: Common Japanese words and phrases
- **Structure**: Natural Japanese sentence patterns and grammar

### Vietnamese Data
- **Content Types**: Social media posts, reviews, daily conversations
- **Script**: Vietnamese with proper diacritical marks
- **Vocabulary**: Common Vietnamese words across various contexts
- **Structure**: Natural Vietnamese sentence patterns and grammar

## 🎯 Data Quality Features

### Realistic Content
- Template-based generation ensures grammatically correct sentences
- Diverse vocabulary across multiple domains and contexts
- Natural language patterns specific to each language

### Scalability
- Efficient generation for large datasets
- Consistent quality across different dataset sizes
- Memory-efficient processing for large word counts

### Customization
- Flexible word count specification
- Multiple language support
- Configurable output directories

## 🔧 Technical Details

### Algorithm
The script uses a template-based approach with the following components:

1. **Template System**: Pre-defined sentence templates for each language
2. **Vocabulary Banks**: Extensive word lists organized by categories
3. **Random Selection**: Probabilistic word and template selection
4. **Text Assembly**: Dynamic sentence construction and formatting

### Performance
- **Small datasets** (1K words): ~1-2 seconds
- **Medium datasets** (100K words): ~10-30 seconds  
- **Large datasets** (1M words): ~2-5 minutes

### File Encoding
- All generated files use UTF-8 encoding
- Proper handling of special characters and diacritical marks
- Cross-platform compatibility

## 📈 Use Cases

### NLP Research
- Text classification training data
- Language model fine-tuning
- Multilingual NLP experiments
- Benchmark dataset creation

### Educational Projects
- Learning NLP preprocessing techniques
- Testing text analysis algorithms
- Demonstrating language-specific patterns
- Creating tutorial datasets

### Development & Testing
- API testing with realistic text data
- Performance testing with large datasets
- Internationalization testing
- Text processing pipeline validation

## 🔍 Quality Assurance

### Data Validation
The script includes built-in validation:
- Word count verification
- File size reporting
- Character encoding validation
- Output directory creation

### Error Handling
- Graceful handling of file system errors
- Validation of command-line arguments
- Clear error messages and suggestions
- Automatic directory creation

## 🚨 Limitations

### Current Limitations
1. **Template Dependency**: Generated text follows predefined patterns
2. **Semantic Coherence**: Individual sentences, not coherent long-form text
3. **Domain Specificity**: Limited to general-purpose content

### Future Improvements
- Advanced context awareness for better text coherence
- Domain-specific templates (legal, medical, technical)
- Integration with neural language models for more natural text

## 🤝 Contributing

To extend the synthetic data generator:

1. **Adding New Languages**: 
   - Create template sets in the `SyntheticTextGenerator` class
   - Add vocabulary dictionaries for the new language
   - Implement a new sentence generation method

2. **Improving Templates**:
   - Add more diverse sentence structures
   - Include domain-specific templates
   - Enhance vocabulary with specialized terms

3. **Performance Optimization**:
   - Implement parallel processing for large datasets
   - Add memory-efficient streaming for very large files
   - Optimize template selection algorithms

## 📄 License

This script is part of the NLP Learning Journey repository and is licensed under the MIT License.

## 🔗 Related Resources

- [Main Repository README](../README.md)
- [NLP Learning Journey Documentation](../docs/)
- [Python Virtual Environments Guide](https://docs.python.org/3/tutorial/venv.html)
- [UTF-8 Encoding Documentation](https://docs.python.org/3/library/codecs.html#standard-encodings)

---

*Happy Data Generation! 🎉*