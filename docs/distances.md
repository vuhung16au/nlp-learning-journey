# Distance Metrics in Natural Language Processing

Distance metrics are fundamental tools in Natural Language Processing (NLP) used to measure similarity or dissimilarity between text documents, words, sentences, or other linguistic units. These metrics enable various NLP tasks such as information retrieval, clustering, classification, and similarity search.

## Overview

Distance metrics quantify how "close" or "far apart" two objects are in a given space. In NLP, these objects can be:
- Text documents
- Word vectors/embeddings
- Sets of tokens or n-grams
- Character sequences
- Semantic representations

The choice of distance metric significantly impacts the performance of NLP algorithms and should align with the specific task requirements and data characteristics.

## String Distance Metrics

### Levenshtein Distance (Edit Distance)

**Purpose**: Measures the minimum number of single-character edits (insertions, deletions, or substitutions) required to change one string into another.

**Mathematical Definition**:
For strings `a` and `b` of lengths `m` and `n`:
```
lev(a,b) = max(m,n) if min(m,n) = 0
lev(a,b) = min(
    lev(tail(a), b) + 1,
    lev(a, tail(b)) + 1,
    lev(tail(a), tail(b)) + cost
) otherwise
```
where `cost = 0` if characters match, `1` otherwise.

**Common Use Cases**:
- Spell checking and correction
- Fuzzy string matching
- DNA sequence analysis
- Plagiarism detection
- Name matching in entity resolution

**Python Implementation**:
```python
def levenshtein_distance(s1, s2):
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

# Example usage
text1 = "kitten"
text2 = "sitting"
distance = levenshtein_distance(text1, text2)
print(f"Levenshtein distance: {distance}")  # Output: 3

# Using external library
from Levenshtein import distance as lev_distance
print(f"Using library: {lev_distance(text1, text2)}")
```

### Hamming Distance

**Purpose**: Measures the number of positions at which corresponding characters are different between two strings of equal length.

**Mathematical Definition**:
For strings `a` and `b` of equal length `n`:
$$ \text{hamming}(a,b) = \sum_{i=1}^n [a[i] \neq b[i]] $$

**Common Use Cases**:
- Error detection and correction
- Comparing binary sequences
- Genetic sequence analysis
- Feature vector comparison

**Python Implementation**:
```python
def hamming_distance(s1, s2):
    """Calculate Hamming distance between two strings of equal length."""
    if len(s1) != len(s2):
        raise ValueError("Strings must be of equal length")
    
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

# Example usage
text1 = "karolin"
text2 = "kathrin"
distance = hamming_distance(text1, text2)
print(f"Hamming distance: {distance}")  # Output: 3

# For binary strings
binary1 = "1011101"
binary2 = "1001001"
distance = hamming_distance(binary1, binary2)
print(f"Binary Hamming distance: {distance}")  # Output: 2
```

### Damerau-Levenshtein Distance

**Purpose**: Extension of Levenshtein distance that also allows transposition of adjacent characters.

**Common Use Cases**:
- Improved spell checking (handles common typing errors)
- OCR error correction
- Data deduplication

**Python Implementation**:
```python
def damerau_levenshtein_distance(s1, s2):
    """Calculate Damerau-Levenshtein distance between two strings."""
    len1, len2 = len(s1), len(s2)
    
    # Create a dictionary for character positions
    char_array = {}
    for c in s1 + s2:
        char_array[c] = 0
    
    # Create distance matrix
    max_dist = len1 + len2
    H = [[max_dist for _ in range(len2 + 2)] for _ in range(len1 + 2)]
    
    H[0][0] = max_dist
    for i in range(0, len1 + 1):
        H[i + 1][0] = max_dist
        H[i + 1][1] = i
    for j in range(0, len2 + 1):
        H[0][j + 1] = max_dist
        H[1][j + 1] = j
    
    for i in range(1, len1 + 1):
        db = 0
        for j in range(1, len2 + 1):
            k = char_array[s2[j - 1]]
            l = db
            if s1[i - 1] == s2[j - 1]:
                cost = 0
                db = j
            else:
                cost = 1
            
            H[i + 1][j + 1] = min(
                H[i][j] + cost,  # substitution
                H[i + 1][j] + 1,  # insertion
                H[i][j + 1] + 1,  # deletion
                H[k][l] + (i - k - 1) + 1 + (j - l - 1)  # transposition
            )
        
        char_array[s1[i - 1]] = i
    
    return H[len1 + 1][len2 + 1]

# Example usage
text1 = "abcdef"
text2 = "abdcef"  # 'cd' transposed to 'dc'
distance = damerau_levenshtein_distance(text1, text2)
print(f"Damerau-Levenshtein distance: {distance}")  # Output: 1
```

## Set-Based Distance Metrics

### Jaccard Distance

**Purpose**: Measures dissimilarity between finite sets, defined as one minus the Jaccard similarity coefficient.

**Mathematical Definition**:
For sets A and B:
```
Jaccard_similarity(A, B) = |A ∩ B| / |A ∪ B|
Jaccard_distance(A, B) = 1 - Jaccard_similarity(A, B) = 1 - |A ∩ B| / |A ∪ B|
```

**Common Use Cases**:
- Document similarity comparison
- Collaborative filtering
- Gene expression analysis
- Image similarity (using feature sets)
- Social network analysis
- Plagiarism detection

**Python Implementation**:
```python
def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 0.0
    
    return intersection / union

def jaccard_distance(set1, set2):
    """Calculate Jaccard distance between two sets."""
    return 1 - jaccard_similarity(set1, set2)

# Example with text documents
def text_to_word_set(text):
    """Convert text to set of words."""
    return set(text.lower().split())

doc1 = "the quick brown fox jumps over the lazy dog"
doc2 = "the lazy dog sleeps under the warm sun"

set1 = text_to_word_set(doc1)
set2 = text_to_word_set(doc2)

similarity = jaccard_similarity(set1, set2)
distance = jaccard_distance(set1, set2)

print(f"Document sets: {set1} vs {set2}")
print(f"Jaccard similarity: {similarity:.3f}")  # Output: ~0.200
print(f"Jaccard distance: {distance:.3f}")     # Output: ~0.800

# Using scikit-learn
from sklearn.metrics import jaccard_score
import numpy as np

# Binary representation example
binary1 = np.array([1, 1, 0, 1, 0, 1, 0])
binary2 = np.array([1, 0, 0, 1, 1, 1, 0])

jaccard_sim = jaccard_score(binary1, binary2)
jaccard_dist = 1 - jaccard_sim
print(f"Binary Jaccard similarity: {jaccard_sim:.3f}")
print(f"Binary Jaccard distance: {jaccard_dist:.3f}")
```

### Masi Distance

**Purpose**: A variation of Jaccard distance designed specifically for comparing sets with partial overlaps, commonly used in computational linguistics for comparing annotation sets.

**Mathematical Definition**:
For sets A and B:
```
Masi_distance(A, B) = 1 - (|A ∩ B| / max(|A|, |B|)) × δ
```
Where δ (delta) is a monotonic function that gives more weight to partial overlaps.

**Properties**:
- More sensitive to partial overlaps than Jaccard distance
- Particularly useful for evaluating inter-annotator agreement
- Values range from 0 (identical sets) to 1 (completely disjoint sets)

**Common Use Cases**:
- Inter-annotator agreement in NLP tasks
- Named entity recognition evaluation
- Part-of-speech tagging evaluation
- Semantic annotation comparison
- Information extraction evaluation

**Python Implementation**:
```python
def masi_distance(set1, set2):
    """
    Calculate MASI (Measuring Agreement on Set-valued Items) distance.
    
    The MASI distance is designed for comparing sets and is particularly
    useful in computational linguistics for inter-annotator agreement.
    """
    if len(set1) == 0 and len(set2) == 0:
        return 0.0
    
    if len(set1) == 0 or len(set2) == 0:
        return 1.0
    
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    # Calculate delta (δ) - monotonic function for partial overlap
    if len(intersection) == 0:
        delta = 0
    elif intersection == set1 or intersection == set2:
        delta = 1  # One set is subset of another
    else:
        delta = len(intersection) / len(union)  # Partial overlap
    
    # MASI coefficient
    masi_coeff = (len(intersection) / max(len(set1), len(set2))) * delta
    
    # MASI distance
    return 1 - masi_coeff

def masi_coefficient(set1, set2):
    """Calculate MASI coefficient (similarity)."""
    return 1 - masi_distance(set1, set2)

# Example with NLP annotation tasks
def compare_annotations():
    """Example of comparing NLP annotations using MASI distance."""
    
    # Named Entity Recognition annotations by two annotators
    annotator1_entities = {"John", "New York", "Apple Inc", "Monday"}
    annotator2_entities = {"John", "New York", "Apple", "Monday", "CEO"}
    
    masi_dist = masi_distance(annotator1_entities, annotator2_entities)
    masi_coeff = masi_coefficient(annotator1_entities, annotator2_entities)
    
    print("NER Annotation Comparison:")
    print(f"Annotator 1: {annotator1_entities}")
    print(f"Annotator 2: {annotator2_entities}")
    print(f"MASI distance: {masi_dist:.3f}")
    print(f"MASI coefficient: {masi_coeff:.3f}")
    
    # POS tag sets comparison
    pos_tags1 = {"NN", "VB", "DT", "JJ", "IN"}
    pos_tags2 = {"NN", "VBZ", "DT", "JJ"}  # Similar but not identical
    
    masi_dist_pos = masi_distance(pos_tags1, pos_tags2)
    jaccard_dist_pos = jaccard_distance(pos_tags1, pos_tags2)
    
    print("\nPOS Tags Comparison:")
    print(f"Set 1: {pos_tags1}")
    print(f"Set 2: {pos_tags2}")
    print(f"MASI distance: {masi_dist_pos:.3f}")
    print(f"Jaccard distance: {jaccard_dist_pos:.3f}")
    print(f"MASI is more sensitive to partial overlap")

compare_annotations()
```

## Vector-Based Distance Metrics

### Cosine Distance

**Purpose**: Measures the cosine of the angle between two vectors, commonly used for high-dimensional data like word embeddings.

**Mathematical Definition**:
For vectors A and B:
```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
cosine_distance(A, B) = 1 - cosine_similarity(A, B)
```

**Common Use Cases**:
- Word embedding similarity
- Document similarity
- Recommendation systems
- Information retrieval
- Text clustering

**Python Implementation**:
```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

def cosine_distance_manual(vec1, vec2):
    """Calculate cosine distance between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 1.0  # Maximum distance for zero vectors
    
    cosine_sim = dot_product / (norm1 * norm2)
    return 1 - cosine_sim

# Example with word embeddings
word_embeddings = {
    'king': np.array([0.1, 0.3, -0.2, 0.8, 0.4]),
    'queen': np.array([0.2, 0.4, -0.1, 0.7, 0.3]),
    'man': np.array([0.0, 0.2, -0.3, 0.5, 0.1]),
    'woman': np.array([0.1, 0.3, -0.2, 0.4, 0.2]),
    'cat': np.array([-0.1, -0.2, 0.5, -0.1, 0.6])
}

# Calculate distances
king_queen_dist = cosine_distance_manual(word_embeddings['king'], word_embeddings['queen'])
king_cat_dist = cosine_distance_manual(word_embeddings['king'], word_embeddings['cat'])

print(f"Cosine distance between 'king' and 'queen': {king_queen_dist:.3f}")
print(f"Cosine distance between 'king' and 'cat': {king_cat_dist:.3f}")

# Using scikit-learn
vectors = np.array([word_embeddings['king'], word_embeddings['queen']])
cos_sim = cosine_similarity(vectors)[0, 1]
cos_dist = cosine_distances(vectors)[0, 1]

print(f"Using sklearn - similarity: {cos_sim:.3f}, distance: {cos_dist:.3f}")
```

### Euclidean Distance

**Purpose**: Measures the straight-line distance between two points in Euclidean space.

**Mathematical Definition**:
For vectors A and B of dimension n:
$$ \text{euclidean\_distance}(A, B) = \sqrt{\sum_{i=1}^n (A[i] - B[i])^2} $$

**Common Use Cases**:
- K-means clustering
- Nearest neighbor search
- Feature space analysis
- Continuous vector comparisons

**Python Implementation**:
```python
import numpy as np
from scipy.spatial.distance import euclidean

def euclidean_distance_manual(vec1, vec2):
    """Calculate Euclidean distance between two vectors."""
    return np.sqrt(np.sum((vec1 - vec2) ** 2))

# Example usage
vec1 = np.array([1, 2, 3, 4])
vec2 = np.array([2, 3, 4, 5])

dist_manual = euclidean_distance_manual(vec1, vec2)
dist_scipy = euclidean(vec1, vec2)

print(f"Manual calculation: {dist_manual:.3f}")
print(f"SciPy calculation: {dist_scipy:.3f}")

# Text feature vectors example
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "The cat sat on the mat",
    "The dog played in the park",
    "Cats and dogs are pets"
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Calculate Euclidean distances between documents
euclidean_dist_01 = euclidean(tfidf_matrix[0].toarray()[0], tfidf_matrix[1].toarray()[0])
euclidean_dist_02 = euclidean(tfidf_matrix[0].toarray()[0], tfidf_matrix[2].toarray()[0])

print(f"Euclidean distance between doc 0 and 1: {euclidean_dist_01:.3f}")
print(f"Euclidean distance between doc 0 and 2: {euclidean_dist_02:.3f}")
```

### Manhattan Distance (L1 Distance)

**Purpose**: Measures the sum of absolute differences between coordinates, also known as taxicab distance.

**Mathematical Definition**:
For vectors A and B of dimension n:
$$ \text{manhattan\_distance}(A, B) = \sum_{i=1}^n |A[i] - B[i]| $$

**Common Use Cases**:
- Robust to outliers
- High-dimensional sparse data
- Feature selection
- Regularization in machine learning

**Python Implementation**:
```python
import numpy as np
from scipy.spatial.distance import cityblock

def manhattan_distance_manual(vec1, vec2):
    """Calculate Manhattan distance between two vectors."""
    return np.sum(np.abs(vec1 - vec2))

# Example usage
vec1 = np.array([1, 2, 3, 4])
vec2 = np.array([2, 4, 1, 6])

dist_manual = manhattan_distance_manual(vec1, vec2)
dist_scipy = cityblock(vec1, vec2)

print(f"Manual calculation: {dist_manual}")
print(f"SciPy calculation: {dist_scipy}")

# Comparison with Euclidean distance
euclidean_dist = np.sqrt(np.sum((vec1 - vec2) ** 2))
print(f"Manhattan distance: {dist_manual}")
print(f"Euclidean distance: {euclidean_dist:.3f}")
```

## Specialized NLP Distance Metrics

### Word Mover's Distance (WMD)

**Purpose**: Measures the distance between two documents by computing the minimum cumulative distance that words from one document need to travel to reach words in another document.

**Common Use Cases**:
- Document similarity with semantic understanding
- Information retrieval
- Text classification
- Plagiarism detection

**Python Implementation**:
```python
# Note: Requires gensim and word embeddings
try:
    from gensim.models import Word2Vec
    from gensim.similarities import WmdSimilarity
    import numpy as np
    
    def word_movers_distance_example():
        """Example of Word Mover's Distance calculation."""
        
        # Sample documents
        documents = [
            "Obama speaks to the media in Illinois".lower().split(),
            "The President greets the press in Chicago".lower().split(),
            "The cat sits on the mat".lower().split()
        ]
        
        # Train a simple Word2Vec model (in practice, use pre-trained embeddings)
        model = Word2Vec(documents, vector_size=50, window=5, min_count=1, workers=1)
        
        # Calculate WMD between first two documents
        distance = model.wv.wmdistance(documents[0], documents[1])
        print(f"WMD between political documents: {distance:.3f}")
        
        # Calculate WMD between first and third documents
        distance2 = model.wv.wmdistance(documents[0], documents[2])
        print(f"WMD between political and cat document: {distance2:.3f}")
        
        return model
    
    # Uncomment to run example (requires additional dependencies)
    # word_movers_distance_example()
    
except ImportError:
    print("Gensim not available. Install with: pip install gensim")
```

### Normalized Compression Distance (NCD)

**Purpose**: Uses compression algorithms to measure similarity between strings or documents.

**Mathematical Definition**:
```
NCD(x,y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))
```
Where C(x) is the compressed size of x, and C(xy) is the compressed size of concatenated x and y.

**Python Implementation**:
```python
import zlib
import bz2

def normalized_compression_distance(s1, s2, compressor=zlib):
    """Calculate Normalized Compression Distance between two strings."""
    
    # Convert strings to bytes if necessary
    if isinstance(s1, str):
        s1 = s1.encode('utf-8')
    if isinstance(s2, str):
        s2 = s2.encode('utf-8')
    
    # Compress individual strings and concatenation
    c1 = len(compressor.compress(s1))
    c2 = len(compressor.compress(s2))
    c12 = len(compressor.compress(s1 + s2))
    
    # Calculate NCD
    ncd = (c12 - min(c1, c2)) / max(c1, c2)
    return ncd

# Example usage
text1 = "The quick brown fox jumps over the lazy dog"
text2 = "A quick brown fox leaps over a lazy dog"
text3 = "Machine learning is a subset of artificial intelligence"

ncd_similar = normalized_compression_distance(text1, text2)
ncd_different = normalized_compression_distance(text1, text3)

print(f"NCD between similar texts: {ncd_similar:.3f}")
print(f"NCD between different texts: {ncd_different:.3f}")

# Using different compressors
ncd_bz2 = normalized_compression_distance(text1, text2, bz2)
print(f"NCD with bz2 compression: {ncd_bz2:.3f}")
```

## Distance Metrics Comparison

### Choosing the Right Distance Metric

The choice of distance metric depends on several factors:

1. **Data Type**:
   - Categorical data: Hamming, Jaccard
   - Continuous data: Euclidean, Manhattan, Cosine
   - Text data: Levenshtein, Jaccard, Cosine (with embeddings)

2. **Dimensionality**:
   - High-dimensional sparse data: Cosine, Manhattan
   - Low-dimensional dense data: Euclidean

3. **Semantic Considerations**:
   - Semantic similarity: Cosine (with embeddings), WMD
   - Syntactic similarity: Levenshtein, Jaccard

4. **Computational Efficiency**:
   - Fast computation needed: Hamming, Manhattan
   - Quality over speed: WMD, NCD

### Practical Comparison Example

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, manhattan_distances

def compare_distance_metrics():
    """Compare different distance metrics on the same dataset."""
    
    # Sample documents
    documents = [
        "The cat sat on the mat",
        "A cat is sitting on a mat",
        "The dog played in the park",
        "Dogs love to play outside",
        "Machine learning is fascinating"
    ]
    
    # Convert to TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # Calculate different distance metrics
    cosine_dist = cosine_distances(tfidf_matrix)
    euclidean_dist = euclidean_distances(tfidf_matrix)
    manhattan_dist = manhattan_distances(tfidf_matrix)
    
    print("Distance Metrics Comparison:")
    print("Documents:")
    for i, doc in enumerate(documents):
        print(f"  {i}: {doc}")
    print()
    
    # Compare distances between similar documents (0 and 1)
    print("Distances between similar documents (0 and 1):")
    print(f"  Cosine distance: {cosine_dist[0, 1]:.3f}")
    print(f"  Euclidean distance: {euclidean_dist[0, 1]:.3f}")
    print(f"  Manhattan distance: {manhattan_dist[0, 1]:.3f}")
    print()
    
    # Compare distances between different documents (0 and 4)
    print("Distances between different documents (0 and 4):")
    print(f"  Cosine distance: {cosine_dist[0, 4]:.3f}")
    print(f"  Euclidean distance: {euclidean_dist[0, 4]:.3f}")
    print(f"  Manhattan distance: {manhattan_dist[0, 4]:.3f}")
    print()
    
    # Jaccard distance for word sets
    def text_to_set(text):
        return set(text.lower().split())
    
    set1 = text_to_set(documents[0])
    set2 = text_to_set(documents[1])
    set3 = text_to_set(documents[4])
    
    jaccard_similar = jaccard_distance(set1, set2)
    jaccard_different = jaccard_distance(set1, set3)
    
    print("Jaccard distances:")
    print(f"  Similar documents: {jaccard_similar:.3f}")
    print(f"  Different documents: {jaccard_different:.3f}")

compare_distance_metrics()
```

## Applications in NLP Tasks

### Information Retrieval

```python
def document_retrieval_example():
    """Example of using distance metrics for document retrieval."""
    
    # Document collection
    documents = [
        "Natural language processing with deep learning",
        "Machine learning algorithms for text analysis",
        "Deep neural networks in NLP applications",
        "Statistical methods for language modeling",
        "Computer vision and image recognition"
    ]
    
    query = "deep learning for NLP"
    
    # Convert to TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    doc_vectors = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform([query])
    
    # Calculate similarities
    similarities = cosine_similarity(query_vector, doc_vectors)[0]
    
    # Rank documents by similarity
    ranked_docs = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
    
    print("Document Retrieval Results:")
    print(f"Query: {query}")
    print("\nRanked documents:")
    for rank, (doc_idx, score) in enumerate(ranked_docs[:3]):
        print(f"  {rank+1}. Score: {score:.3f} - {documents[doc_idx]}")

document_retrieval_example()
```

### Text Clustering

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def text_clustering_example():
    """Example of text clustering using different distance metrics."""
    
    # Sample texts from different categories
    texts = [
        "Machine learning algorithms",
        "Deep neural networks",
        "Artificial intelligence research",
        "Apple fruit nutrition facts",
        "Banana health benefits",
        "Orange vitamin content",
        "Football match results",
        "Basketball game highlights",
        "Soccer tournament news"
    ]
    
    # Convert to TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    # Display results
    print("Text Clustering Results:")
    for i, (text, cluster) in enumerate(zip(texts, clusters)):
        print(f"Cluster {cluster}: {text}")

text_clustering_example()
```

## Performance Considerations

### Computational Complexity

| Distance Metric | Time Complexity | Space Complexity | Best Use Case |
|----------------|-----------------|------------------|---------------|
| Hamming | O(n) | O(1) | Equal-length strings |
| Levenshtein | O(mn) | O(mn) | String similarity |
| Jaccard | O(min(n,m)) | O(n+m) | Set similarity |
| Cosine | O(n) | O(1) | High-dimensional vectors |
| Euclidean | O(n) | O(1) | Continuous features |
| Manhattan | O(n) | O(1) | Sparse data |

### Optimization Tips

```python
# For large-scale applications
import faiss  # Facebook AI Similarity Search
import numpy as np

def efficient_similarity_search():
    """Example of efficient similarity search for large datasets."""
    
    # Generate sample embeddings
    dimension = 128
    num_vectors = 10000
    vectors = np.random.random((num_vectors, dimension)).astype('float32')
    
    # Build FAISS index for fast similarity search
    index = faiss.IndexFlatIP(dimension)  # Inner Product (for normalized vectors)
    faiss.normalize_L2(vectors)  # Normalize for cosine similarity
    index.add(vectors)
    
    # Search for similar vectors
    query_vector = np.random.random((1, dimension)).astype('float32')
    faiss.normalize_L2(query_vector)
    
    k = 5  # Number of nearest neighbors
    similarities, indices = index.search(query_vector, k)
    
    print(f"Top {k} similar vectors:")
    for i, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
        print(f"  {i+1}. Index: {idx}, Similarity: {sim:.3f}")

# Uncomment to run (requires faiss installation)
# efficient_similarity_search()
```

## Conclusion

Distance metrics are essential tools in NLP that enable various applications from basic string matching to sophisticated semantic similarity tasks. The choice of metric should be guided by:

1. **Data characteristics**: Type, dimensionality, and sparsity
2. **Task requirements**: Speed vs. accuracy, semantic vs. syntactic similarity
3. **Computational constraints**: Available resources and real-time requirements

Understanding these metrics and their properties allows NLP practitioners to make informed decisions and achieve better results in their applications.

### Key Takeaways

- **String distances** (Levenshtein, Hamming) are ideal for exact text matching and error correction
- **Set-based distances** (Jaccard, MASI) work well for comparing discrete feature sets
- **Vector distances** (Cosine, Euclidean) are essential for embedding-based similarity
- **Specialized metrics** (WMD, NCD) provide advanced semantic understanding but with higher computational cost
- **Performance considerations** are crucial for large-scale applications

Choose the appropriate distance metric based on your specific use case, data characteristics, and performance requirements to achieve optimal results in your NLP tasks.