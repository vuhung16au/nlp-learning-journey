# Quick Start Guide

This guide gets you from zero to running the hands-on examples in this repository. For the full overview of what NLP is and what this project covers, start with [README.md](README.md).

---

## How to Use This Project

Follow this loop:

1. **Read [README.md](README.md)** — understand the learning goals, repository layout, and core topics.
2. **Work through the notebooks in [examples/](examples/)** — run cells top to bottom, change inputs, and redo the exercises. Use the suggested order below.
3. **Open [docs/](docs/) when you need theory** — each notebook maps to reference notes; you do not need to read every doc upfront.

---

## Environment Setup

### Requirements

| Requirement | Details |
|-------------|---------|
| **Python** | 3.8 or higher (3.9+ recommended for `examples/spaCy-Linguistic/`) |
| **Programming** | Basic Python (lists, dicts, functions, classes) |
| **Optional** | Familiarity with machine learning helps for later notebooks |
| **Disk / network** | ~2–4 GB for PyTorch and Hugging Face model downloads; internet needed on first run |

### 1. Clone the repository

```bash
git clone https://github.com/vuhung16au/nlp-learning-journey.git
cd nlp-learning-journey
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download language models and NLTK data

Many notebooks expect these assets to already be present:

```bash
# spaCy English model (used in spaCy and NER/POS notebooks)
python -m spacy download en_core_web_sm

# NLTK corpora (tokenization, stemming, lemmatization)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words')"
```

### 5. (Optional) Generate sample text data

Some notebooks can use synthetic datasets under `data/`:

```bash
cd scripts
python synthetic-data.py
cd ..
```

See [scripts/README.md](scripts/README.md) for options (`--language`, `--size`, etc.).

### 6. Start Jupyter

From the repository root:

```bash
jupyter notebook
```

Or use JupyterLab:

```bash
jupyter lab
```

Then open notebooks from `examples/`.

### Troubleshooting

| Issue | What to try |
|-------|-------------|
| `ModuleNotFoundError` | Activate `.venv` and run `pip install -r requirements.txt` again |
| spaCy model missing | `python -m spacy download en_core_web_sm` |
| NLTK lookup errors | Re-run the `nltk.download(...)` command above |
| Hugging Face model download fails | Check network; first run of `HuggingFace-basic.ipynb` and transformer notebooks downloads models |
| Out of memory (GPU/CPU) | Close other apps; use smaller models or skip heavy cells marked optional |

---

## Running the Examples

### General workflow

1. Open one notebook from the order below.
2. **Run all cells in order** — most notebooks include a setup cell that installs or verifies packages.
3. Read the markdown cells, then execute code cells yourself (do not only read).
4. When a concept is unclear, jump to the linked doc, then return to the notebook.
5. Move to the next notebook only after you understand the current one.

### Notebooks with their own setup

The **spaCy Linguistic** sub-series has a dedicated Makefile:

```bash
cd examples/spaCy-Linguistic
make setup-env      # creates venv, installs deps, downloads spaCy models
make test-notebooks # optional: verify all six notebooks run
jupyter notebook
```

Details: [examples/spaCy-Linguistic/README.md](examples/spaCy-Linguistic/README.md).

---

## Recommended Notebook Order

Work through **Phase 1 → Phase 6** in order. Within each phase, follow the numbered list.

### Phase 1 — Text fundamentals

Build intuition for cleaning and splitting text before any library-specific APIs.

| # | Notebook | Focus |
|---|----------|--------|
| 1 | [examples/regex.ipynb](examples/regex.ipynb) | Pattern matching, text cleaning, extraction |
| 2 | [examples/tokenization.ipynb](examples/tokenization.ipynb) | Word, sentence, and subword tokenization |
| 3 | [examples/normalization.ipynb](examples/normalization.ipynb) | Case folding, stemming, lemmatization, stop words |

**Docs to read when stuck:** [docs/key-concepts.md](docs/key-concepts.md) (preprocessing, tokenization), [docs/chunking.md](docs/chunking.md)

---

### Phase 2 — Core NLP libraries

Learn the two main Python stacks used throughout the repo.

| # | Notebook | Focus |
|---|----------|--------|
| 4 | [examples/NLTK-basic.ipynb](examples/NLTK-basic.ipynb) | Corpora, tokenizers, basic NLP with NLTK |
| 5 | [examples/spacy-basic.ipynb](examples/spacy-basic.ipynb) | `Doc` objects, pipelines, production-style NLP |

**Docs to read when stuck:** [docs/python-libraries.md](docs/python-libraries.md), [docs/NLTK-spaCy.md](docs/NLTK-spaCy.md), [docs/nlp_prerequisites.md](docs/nlp_prerequisites.md)

---

### Phase 3 — Linguistic analysis tasks

Apply libraries to classic NLP tasks.

| # | Notebook | Focus |
|---|----------|--------|
| 6 | [examples/pos-tagging.ipynb](examples/pos-tagging.ipynb) | Part-of-speech tagging (NLTK + spaCy) |
| 7 | [examples/ner.ipynb](examples/ner.ipynb) | Named entity recognition |

**Docs to read when stuck:** [docs/part-of-speech.md](docs/part-of-speech.md), [docs/tagging.md](docs/tagging.md), [docs/NER.md](docs/NER.md), [docs/linguistic-concepts.md](docs/linguistic-concepts.md)

---

### Phase 4 — spaCy deep dive (sequential sub-series)

Complete this **six-notebook track in order** after Phase 3. It goes deeper than `spacy-basic.ipynb`.

| # | Notebook | Focus |
|---|----------|--------|
| 8 | [examples/spaCy-Linguistic/01-spaCy-Linguistic-Introduction-to-spaCy-Basics.ipynb](examples/spaCy-Linguistic/01-spaCy-Linguistic-Introduction-to-spaCy-Basics.ipynb) | Installation, `Doc`, POS, lemmatization |
| 9 | [examples/spaCy-Linguistic/02-spaCy-Linguistic-Understanding-Text-Structure.ipynb](examples/spaCy-Linguistic/02-spaCy-Linguistic-Understanding-Text-Structure.ipynb) | Tokenization, sentences, morphology |
| 10 | [examples/spaCy-Linguistic/03-spaCy-Linguistic-Syntactic-Analysis.ipynb](examples/spaCy-Linguistic/03-spaCy-Linguistic-Syntactic-Analysis.ipynb) | Dependency parsing, noun chunks |
| 11 | [examples/spaCy-Linguistic/04-spaCy-Linguistic-Named-Entity-Recognition.ipynb](examples/spaCy-Linguistic/04-spaCy-Linguistic-Named-Entity-Recognition.ipynb) | NER types, custom rules, visualization |
| 12 | [examples/spaCy-Linguistic/05-spaCy-Linguistic-Advanced-Features.ipynb](examples/spaCy-Linguistic/05-spaCy-Linguistic-Advanced-Features.ipynb) | Vectors, custom pipelines |
| 13 | [examples/spaCy-Linguistic/06-spaCy-Linguistic-Practical-Mini-Projects.ipynb](examples/spaCy-Linguistic/06-spaCy-Linguistic-Practical-Mini-Projects.ipynb) | End-to-end mini projects |

**Docs to read when stuck:** [docs/parsing.md](docs/parsing.md), [docs/CFG.md](docs/CFG.md)

---

### Phase 5 — Machine learning for text

Classical ML and neural building blocks before transformers.

| # | Notebook | Focus |
|---|----------|--------|
| 14 | [examples/perceptron.ipynb](examples/perceptron.ipynb) | Linear classifier basics (scikit-learn) |
| 15 | [examples/text-classification.ipynb](examples/text-classification.ipynb) | TF-IDF, pipelines, text classification |
| 16 | [examples/sentiment-analysis.ipynb](examples/sentiment-analysis.ipynb) | Rule-based, ML, and transformer sentiment |
| 17 | [examples/activation-functions.ipynb](examples/activation-functions.ipynb) | Activations used in neural NLP models |

**Docs to read when stuck:** [docs/bag-of-words.md](docs/bag-of-words.md), [docs/TF-IDF.md](docs/TF-IDF.md), [docs/mathematics.md](docs/mathematics.md), [docs/gradient-descent.md](docs/gradient-descent.md), [docs/loss-functions.md](docs/loss-functions.md), [docs/softmax.md](docs/softmax.md)

---

### Phase 6 — Transformers and advanced applications

Requires Phases 1–5 (especially sentiment and activation functions). Downloads large pre-trained models.

| # | Notebook | Focus |
|---|----------|--------|
| 18 | [examples/HuggingFace-basic.ipynb](examples/HuggingFace-basic.ipynb) | Pipelines, BERT, GPT-2, fine-tuning basics |
| 19 | [examples/text-generation.ipynb](examples/text-generation.ipynb) | N-grams, Markov chains, transformer generation |
| 20 | [examples/text-summarization.ipynb](examples/text-summarization.ipynb) | Extractive and abstractive summarization |
| 21 | [examples/text-translation.ipynb](examples/text-translation.ipynb) | Language detection and translation pipelines |
| 22 | [examples/date-string-converter.ipynb](examples/date-string-converter.ipynb) | Encoder–decoder seq2seq with PyTorch (advanced) |

**Docs to read when stuck:** [docs/transformer.md](docs/transformer.md), [docs/attention.md](docs/attention.md), [docs/BERT.md](docs/BERT.md), [docs/GPT.md](docs/GPT.md), [docs/LLM.md](docs/LLM.md), [docs/transfer-learning.md](docs/transfer-learning.md), [docs/encoder-decoder-architecture.md](docs/encoder-decoder-architecture.md), [docs/RNN.md](docs/RNN.md), [docs/LSTM.md](docs/LSTM.md), [docs/embedding.md](docs/embedding.md), [docs/byte-pair-encoding-BPE.md](docs/byte-pair-encoding-BPE.md)

---

## When to Read `docs/`

You do **not** need to read all documentation before coding. Use this map:

| You are working on… | Start with these docs |
|---------------------|------------------------|
| First time / orientation | [docs/nlp_prerequisites.md](docs/nlp_prerequisites.md), [docs/definitions.md](docs/definitions.md), [docs/why.md](docs/why.md) |
| Text preprocessing | [docs/key-concepts.md](docs/key-concepts.md), [docs/chunking.md](docs/chunking.md) |
| Features & classical ML | [docs/bag-of-words.md](docs/bag-of-words.md), [docs/TF-IDF.md](docs/TF-IDF.md), [docs/similarity.md](docs/similarity.md), [docs/distances.md](docs/distances.md) |
| Linguistics background | [docs/linguistic-concepts.md](docs/linguistic-concepts.md), [docs/discourse.md](docs/discourse.md), [docs/centering-theory.md](docs/centering-theory.md) |
| Neural networks & training | [docs/mathematics.md](docs/mathematics.md), [docs/backpropagation.md](docs/backpropagation.md), [docs/HMM.md](docs/HMM.md) |
| Modern LLMs | [docs/self-supervised-learning.md](docs/self-supervised-learning.md), [docs/zipf-law.md](docs/zipf-law.md) |
| Real-world motivation | [docs/use-cases.md](docs/use-cases.md), [docs/Turing-test.md](docs/Turing-test.md) |
| Repo style & notebook norms | [docs/conventions.md](docs/conventions.md) |

For a conceptual index of everything in `docs/`, see the **Core Documentation** section in [README.md](README.md).

---

## Suggested Weekly Pace (optional)

| Week | Phases | Approx. time |
|------|--------|--------------|
| 1 | Phase 1 + Phase 2 | 6–8 hours |
| 2 | Phase 3 + Phase 4 (notebooks 8–11) | 8–10 hours |
| 3 | Phase 4 (12–13) + Phase 5 | 8–10 hours |
| 4 | Phase 6 | 8–12 hours |

Adjust based on your background. Re-run notebooks with your own text samples to reinforce learning.

---

## Next Steps

- Explore additional notes in [docs/](docs/) as you encounter new terms ([docs/definitions.md](docs/definitions.md) is a good glossary).
- Generate custom practice data with [scripts/synthetic-data.py](scripts/synthetic-data.py).
- Track your progress via git history and your own notes alongside the notebooks.

Happy learning!
