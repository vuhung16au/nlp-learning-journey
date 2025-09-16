# NLP Learning Journey - GitHub Copilot Instructions

**Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

This repository is a comprehensive Natural Language Processing (NLP) learning resource containing documentation, example notebooks, and code implementations. It's designed as an educational repository for learning NLP concepts from fundamentals to advanced topics.

## Working Effectively

### Bootstrap the Environment
Execute these commands in order to set up a fully functional development environment:

1. **Verify Python version** (should be Python 3.8+):
   ```bash
   python --version
   ```

2. **Install all dependencies** - NEVER CANCEL: Takes 5-8 minutes. Set timeout to 600+ seconds:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download spaCy English model** - NEVER CANCEL: Takes 1-2 minutes. Set timeout to 300+ seconds:
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Download NLTK datasets** - NEVER CANCEL: Takes 1-2 minutes. Set timeout to 300+ seconds:
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
   ```

5. **Verify installation** (should complete in under 10 seconds):
   ```bash
   python -c "
   import nltk, spacy, pandas as pd, numpy as np, sklearn
   nlp = spacy.load('en_core_web_sm')
   from nltk.tokenize import word_tokenize
   print('All libraries installed and working correctly!')
   "
   ```

### Core Libraries and Dependencies
The repository uses these essential libraries:
- **NLTK**: Natural language processing toolkit
- **spaCy**: Industrial-strength NLP library  
- **Transformers**: Hugging Face transformer models (requires internet for model downloads)
- **PyTorch & TensorFlow**: Deep learning frameworks
- **scikit-learn**: Machine learning algorithms
- **Pandas & NumPy**: Data manipulation and analysis
- **Matplotlib, Seaborn, Plotly**: Data visualization
- **Jupyter**: Interactive notebook environment

### Running Jupyter Notebooks
- **Start Jupyter Lab**:
  ```bash
  jupyter lab
  ```
- **Start Jupyter Notebook**:
  ```bash
  jupyter notebook
  ```
- **Convert notebook to script**:
  ```bash
  jupyter nbconvert --to script examples/notebook_name.ipynb
  ```

**IMPORTANT**: Some notebooks (like tokenization.ipynb) require internet access to download pre-trained models from Hugging Face. In offline environments, these cells will fail with network errors - this is expected behavior.

## Validation Scenarios

### Always Test These Scenarios After Making Changes:

1. **Basic Library Import Test** (should complete in 2-3 seconds):
   ```bash
   python -c "
   import nltk, spacy, transformers, torch, tensorflow as tf, sklearn
   import pandas as pd, numpy as np, matplotlib.pyplot as plt
   print('Core imports successful!')
   "
   ```

2. **spaCy Functionality Test** (should complete in 2-3 seconds):
   ```bash
   python -c "
   import spacy
   nlp = spacy.load('en_core_web_sm')
   doc = nlp('Hello world! This is a test.')
   print([token.text for token in doc])
   "
   ```

3. **NLTK Functionality Test** (should complete in 2-3 seconds):
   ```bash
   python -c "
   from nltk.tokenize import word_tokenize
   tokens = word_tokenize('Hello world!')
   print(tokens)
   "
   ```

4. **Data Science Libraries Test** (should complete in 2-3 seconds):
   ```bash
   python -c "
   import pandas as pd, numpy as np
   from sklearn.feature_extraction.text import TfidfVectorizer
   df = pd.DataFrame({'text': ['hello', 'world']})
   vectorizer = TfidfVectorizer()
   tfidf = vectorizer.fit_transform(df['text'])
   print('Data science libraries working!')
   "
   ```

**Manual Validation Requirement**: After any significant changes, run all four validation tests to ensure the environment remains functional.

## Repository Structure and Navigation

### Key Directories:
- **`docs/`**: Documentation and learning notes (currently contains `python-libraries.md`)
- **`examples/`**: 9 Jupyter notebooks with practical NLP examples:
  - `ner.ipynb` - Named Entity Recognition
  - `normalization.ipynb` - Text normalization
  - `pos-tagging.ipynb` - Part-of-speech tagging  
  - `sentiment-analysis.ipynb` - Sentiment analysis
  - `text-classification.ipynb` - Text classification
  - `text-generation.ipynb` - Text generation
  - `text-summarization.ipynb` - Text summarization
  - `text-translation.ipynb` - Text translation
  - `tokenization.ipynb` - Tokenization techniques

### Important Files:
- **`requirements.txt`**: All Python dependencies (30 packages)
- **`README.md`**: Comprehensive repository documentation
- **`LICENSE.md`**: MIT license

## Common Tasks and Workflows

### For Documentation Changes:
- Edit markdown files in `docs/` directory
- No build or test steps required for documentation-only changes

### For Notebook Development:
1. **Create new notebook**:
   ```bash
   jupyter lab
   # Create new notebook in examples/ directory
   ```

2. **Test notebook execution** (may take 30+ seconds for complex notebooks):
   ```bash
   jupyter nbconvert --execute examples/your_notebook.ipynb --to notebook --inplace
   ```

3. **Convert to different formats**:
   ```bash
   jupyter nbconvert examples/your_notebook.ipynb --to html
   jupyter nbconvert examples/your_notebook.ipynb --to pdf  # requires LaTeX
   ```

### For Code Development:
1. Always run the validation scenarios after changes
2. Use virtual environments for isolation:
   ```bash
   python -m venv nlp_env
   source nlp_env/bin/activate  # Linux/Mac
   # nlp_env\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

## Troubleshooting Common Issues

### Network-Related Errors:
- **Hugging Face model downloads fail**: Expected in offline environments. Use local models or skip those cells.
- **NLTK download errors**: Re-run the NLTK download command from bootstrap steps.

### Environment Issues:
- **Import errors**: Re-run the complete bootstrap sequence
- **spaCy model not found**: Run `python -m spacy download en_core_web_sm`
- **NLTK data missing**: Run the NLTK download command from bootstrap steps

### Performance Expectations:
- **Environment setup**: 5-8 minutes total
- **Individual notebook execution**: 30 seconds to 2 minutes (without network downloads)
- **Library imports**: 2-3 seconds
- **Simple NLP operations**: Under 1 second

## Development Best Practices

### Before Making Changes:
1. Always run the complete bootstrap sequence in a fresh environment
2. Execute all validation scenarios
3. Test at least one example notebook end-to-end

### After Making Changes:
1. Run all validation scenarios to ensure nothing broke
2. If adding new dependencies, update `requirements.txt`
3. If creating new notebooks, ensure they follow the existing structure and include proper documentation

### Code Quality:
- No formal linting or testing infrastructure exists
- Follow Python PEP 8 style guidelines
- Include comprehensive docstrings and comments in notebooks
- Ensure all cells in notebooks can execute successfully (except for network-dependent cells in offline environments)

## Limitations and Known Issues

1. **Internet Dependency**: Some transformers functionality requires internet access for model downloads
2. **No CI/CD**: Repository has no automated testing or continuous integration
3. **No Formal Testing**: No unit tests or automated validation beyond manual scenarios
4. **Resource Intensive**: Some deep learning operations may require significant memory/CPU

## Quick Reference Commands

```bash
# Complete setup (run in order, NEVER CANCEL any step)
pip install -r requirements.txt                    # 5-8 minutes
python -m spacy download en_core_web_sm            # 1-2 minutes  
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"  # 1-2 minutes

# Quick validation (run all to verify environment)
python -c "import nltk, spacy, transformers, torch, tensorflow, sklearn, pandas, numpy, matplotlib; print('All imports OK')"
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('spaCy OK')"
python -c "from nltk.tokenize import word_tokenize; print('NLTK OK')"
python -c "from sklearn.feature_extraction.text import TfidfVectorizer; print('sklearn OK')"

# Start development environment
jupyter lab                                        # Interactive development
```

## Documentation Standards

### Mermaid Diagrams

**ALWAYS use Mermaid diagrams instead of plaintext diagrams for visual representations.**

**Prohibited - DO NOT use plaintext diagrams:**
```
Input Sequence → [Encoder] → Context Vector → [Decoder] → Output Sequence
    (variable)                  (fixed-size)                (variable)
```

**Required - Use Mermaid with proper styling:**
```mermaid
graph TD
    A[Input Sequence] --> B[Encoder]
    B --> C[Context Vector]
    C --> D[Decoder]
    D --> E[Output Sequence]

    style A fill:#FFFFFF,stroke:#582C67,color:#333,stroke-width:2px
    style B fill:#582C67,stroke:#C60C30,color:#FFFFFF,stroke-width:2px
    style C fill:#C60C30,stroke:#582C67,color:#FFFFFF,stroke-width:2px
    style D fill:#582C67,stroke:#C60C30,color:#FFFFFF,stroke-width:2px
    style E fill:#FFFFFF,stroke:#582C67,color:#333,stroke-width:2px

    subgraph Notes
        sub1[Input Sequence - variable length]
        sub2[Context Vector - fixed-size]
        sub3[Output Sequence - variable length]
    end
```

**Mermaid Styling Guidelines:**
- Use the specified color palette: Purple (#582C67), White (#FFFFFF), Red (#C60C30)
- Apply gradients between these colors for visual variety
- Use stroke styling for emphasis (stroke-width, stroke colors)
- Include contextual notes using subgraphs when helpful
- Use descriptive node labels that clearly explain the concept
- Maintain readability with appropriate text color contrast

### LaTeX Mathematical Notation

**For inline LaTeX expressions, use single dollar signs:**
```markdown
Correct: `$h_t$: Hidden state at time step t`
Correct: The probability `$P(y_t | x_1, ..., x_t)$` represents...
Wrong: `$$h_t$$: Hidden state at time step t`
```

**For complex or standalone formulas, use double dollar signs in separate paragraphs:**
```markdown
The forward pass equations are defined as:

$$ f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) $$

$$ \alpha_t(i) = P(x_1, x_2, ..., x_t, q_t = s_i | \lambda) $$

$$ f(r) = \frac{k}{r^\alpha} $$
```

**LaTeX Best Practices:**
- Use inline math (`$...$`) for variables, short expressions, and mathematical terms within sentences
- Use block math (`$$...$$`) for complex formulas, equations, and mathematical expressions that deserve emphasis
- Always put block formulas in separate paragraphs for better readability
- Include text explanations before or after complex formulas to provide context
- Use consistent notation throughout the repository

**Example of proper mathematical documentation:**
```markdown
The LSTM cell uses three gates to control information flow. The forget gate `$f_t$` determines what information to discard:

$$ f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) $$

Where `$\sigma$` is the sigmoid function, `$W_f$` is the weight matrix, and `$b_f$` is the bias vector.
```

### Visual Documentation Requirements

1. **No Plaintext Diagrams**: Always use Mermaid for flowcharts, architecture diagrams, and process flows
2. **Color Coding**: Use consistent colors to represent similar concepts across diagrams
3. **Mathematical Precision**: Use LaTeX for all mathematical expressions and formulas
4. **Contextual Notes**: Include explanatory text around diagrams and formulas
5. **Accessibility**: Ensure diagrams and formulas have accompanying text descriptions

These standards apply to all Markdown files in the repository, including documentation in `docs/`, README files, and example notebooks.

Remember: This is an educational repository focused on learning NLP concepts. Always prioritize working examples and clear documentation over complex tooling or optimization.