# Mathematics for Natural Language Processing

This document covers the essential mathematical concepts, formulas, and techniques needed to understand and implement Natural Language Processing systems. From basic statistics to advanced linear algebra and optimization, these mathematical foundations are crucial for modern NLP.

## Table of Contents

1. [Linear Algebra](#linear-algebra)
2. [Probability Theory](#probability-theory)
3. [Statistics](#statistics)
4. [Calculus](#calculus)
5. [Optimization](#optimization)
6. [Information Theory](#information-theory)
7. [Distance and Similarity Measures](#distance-and-similarity-measures)
8. [Matrix Decomposition](#matrix-decomposition)
9. [Neural Network Mathematics](#neural-network-mathematics)
10. [Transformer Mathematics](#transformer-mathematics)

## Linear Algebra

Linear algebra forms the foundation of modern NLP, as text is represented using vectors and matrices.

### Vectors

**Vector Representation**
- Text elements (words, sentences) are represented as vectors in high-dimensional space
- Vector: `v = [v₁, v₂, ..., vₙ]`
- Dimensionality typically ranges from 50 to 4096 in modern systems

**Vector Operations**
- **Addition**: `v + w = [v₁+w₁, v₂+w₂, ..., vₙ+wₙ]`
- **Scalar Multiplication**: `αv = [αv₁, αv₂, ..., αvₙ]`
- **Dot Product**: `v · w = Σᵢ vᵢwᵢ = v₁w₁ + v₂w₂ + ... + vₙwₙ`

**Vector Norms**
- **L1 Norm (Manhattan)**: `||v||₁ = Σᵢ |vᵢ|`
- **L2 Norm (Euclidean)**: `||v||₂ = √(Σᵢ vᵢ²)`
- **L∞ Norm (Maximum)**: `||v||∞ = maxᵢ |vᵢ|`

### Matrices

**Matrix Representation**
- Batch processing of text data
- Weight matrices in neural networks
- Matrix A with dimensions m×n: `A ∈ ℝᵐˣⁿ`

**Matrix Operations**
- **Matrix Multiplication**: `(AB)ᵢⱼ = Σₖ AᵢₖBₖⱼ`
- **Transpose**: `(Aᵀ)ᵢⱼ = Aⱼᵢ`
- **Trace**: `tr(A) = Σᵢ Aᵢᵢ` (sum of diagonal elements)

**Important Matrix Properties**
- **Orthogonal Matrix**: `AAᵀ = AᵀA = I`
- **Symmetric Matrix**: `A = Aᵀ`
- **Positive Definite**: `xᵀAx > 0` for all non-zero x

### Eigenvalues and Eigenvectors

**Definition**
- For matrix A, eigenvector v and eigenvalue λ satisfy: `Av = λv`
- Used in Principal Component Analysis (PCA) for dimensionality reduction

**Characteristic Equation**
- `det(A - λI) = 0`
- Polynomial whose roots are eigenvalues

**Applications in NLP**
- Latent Semantic Analysis (LSA)
- Spectral clustering of documents
- Word embedding analysis

## Probability Theory

Probability theory is essential for language modeling and uncertainty quantification in NLP.

### Basic Probability

**Probability Axioms**
- `0 ≤ P(A) ≤ 1` for any event A
- `P(Ω) = 1` (probability of sample space)
- `P(A ∪ B) = P(A) + P(B)` if A and B are disjoint

**Conditional Probability**
- `P(A|B) = P(A ∩ B) / P(B)` when `P(B) > 0`
- Foundation of language modeling: `P(word|context)`

**Bayes' Theorem**
- `P(A|B) = P(B|A)P(A) / P(B)`
- Used in classification: `P(class|text) = P(text|class)P(class) / P(text)`

### Random Variables

**Discrete Random Variables**
- Probability Mass Function (PMF): `P(X = x) = p(x)`
- Expected Value: `E[X] = Σₓ x·p(x)`
- Variance: `Var(X) = E[X²] - (E[X])²`

**Continuous Random Variables**
- Probability Density Function (PDF): `f(x)`
- Expected Value: `E[X] = ∫ x·f(x)dx`
- Cumulative Distribution Function: `F(x) = P(X ≤ x)`

### Important Distributions

**Multinomial Distribution**
- Used for word counts in documents
- PMF: `P(x₁,...,xₖ) = n! / (x₁!...xₖ!) · p₁^x₁...pₖ^xₖ`
- Where `Σᵢ xᵢ = n` and `Σᵢ pᵢ = 1`

**Gaussian (Normal) Distribution**
- Used for word embeddings and neural network weights
- PDF: `f(x) = 1/(σ√(2π)) · exp(-(x-μ)²/(2σ²))`
- Multivariate: `f(x) = 1/√((2π)ᵏ|Σ|) · exp(-½(x-μ)ᵀΣ⁻¹(x-μ))`

**Categorical Distribution**
- Used for word prediction in language models
- PMF: `P(X = k) = pₖ` where k ∈ {1,2,...,K}

## Statistics

Statistical methods are crucial for analyzing text data and evaluating model performance.

### Descriptive Statistics

**Measures of Central Tendency**
- **Mean**: `μ = (1/n)Σᵢ xᵢ`
- **Median**: Middle value when data is sorted
- **Mode**: Most frequently occurring value

**Measures of Dispersion**
- **Variance**: `σ² = (1/n)Σᵢ (xᵢ - μ)²`
- **Standard Deviation**: `σ = √σ²`
- **Coefficient of Variation**: `CV = σ/μ`

### Correlation and Covariance

**Covariance**
- `Cov(X,Y) = E[(X - E[X])(Y - E[Y])]`
- Measures linear relationship between variables

**Pearson Correlation Coefficient**
- `ρ = Cov(X,Y) / (σₓσᵧ)`
- Range: [-1, 1], where 0 means no linear correlation

**Spearman Rank Correlation**
- Non-parametric measure based on rankings
- Robust to outliers and non-linear relationships

### Hypothesis Testing

**Statistical Significance**
- p-value: Probability of observing data given null hypothesis
- Common significance level: α = 0.05

**t-test**
- Compare means between groups
- t-statistic: `t = (x̄ - μ₀) / (s/√n)`

**Chi-square Test**
- Test independence of categorical variables
- χ² statistic: `χ² = Σᵢⱼ (Oᵢⱼ - Eᵢⱼ)² / Eᵢⱼ`

## Calculus

Calculus is essential for optimization in machine learning and neural networks.

### Derivatives

**Single Variable Calculus**
- Derivative: `f'(x) = lim_{h→0} [f(x+h) - f(x)] / h`
- Chain Rule: `(f(g(x)))' = f'(g(x)) · g'(x)`

**Common Derivatives for NLP**
- `d/dx(log x) = 1/x` (used in log-likelihood)
- `d/dx(eˣ) = eˣ` (used in softmax)
- `d/dx(sigmoid(x)) = sigmoid(x)(1 - sigmoid(x))`

### Partial Derivatives

**Multivariable Functions**
- Partial derivative: `∂f/∂x = lim_{h→0} [f(x+h,y) - f(x,y)] / h`
- Gradient: `∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]`

**Hessian Matrix**
- Second-order partial derivatives: `H(f)ᵢⱼ = ∂²f/∂xᵢ∂xⱼ`
- Used in second-order optimization methods

### Backpropagation

**Chain Rule for Neural Networks**
- For composition `f(g(x))`: `df/dx = df/dg · dg/dx`
- Extended to deep networks: `∂L/∂w₁ = ∂L/∂y · ∂y/∂z · ∂z/∂w₁`

**Gradient Flow**
- Forward pass: Compute outputs layer by layer
- Backward pass: Compute gradients from output to input

## Optimization

Optimization algorithms train neural networks by minimizing loss functions.

### Gradient Descent

**Basic Gradient Descent**
- Update rule: `θₜ₊₁ = θₜ - α∇f(θₜ)`
- α is the learning rate

**Stochastic Gradient Descent (SGD)**
- Use single sample or mini-batch
- Update: `θₜ₊₁ = θₜ - α∇f(θₜ; xᵢ, yᵢ)`

**SGD with Momentum**
- `vₜ₊₁ = βvₜ + α∇f(θₜ)`
- `θₜ₊₁ = θₜ - vₜ₊₁`
- β is momentum coefficient (typically 0.9)

### Advanced Optimizers

**Adam (Adaptive Moment Estimation)**
- Combines momentum and adaptive learning rates
- `mₜ = β₁mₜ₋₁ + (1-β₁)∇f(θₜ)`
- `vₜ = β₂vₜ₋₁ + (1-β₂)(∇f(θₜ))²`
- `θₜ₊₁ = θₜ - α·m̂ₜ/(√v̂ₜ + ε)`

**AdaGrad**
- Adapts learning rate based on historical gradients
- `θₜ₊₁ = θₜ - α/(√Gₜ + ε) · ∇f(θₜ)`
- Where `Gₜ = Σᵢ₌₁ᵗ (∇f(θᵢ))²`

### Regularization

**L1 Regularization (Lasso)**
- Add penalty: `R(θ) = λ||θ||₁ = λΣᵢ|θᵢ|`
- Promotes sparsity in parameters

**L2 Regularization (Ridge)**
- Add penalty: `R(θ) = λ||θ||₂² = λΣᵢθᵢ²`
- Prevents overfitting by penalizing large weights

**Elastic Net**
- Combines L1 and L2: `R(θ) = λ₁||θ||₁ + λ₂||θ||₂²`

## Information Theory

Information theory provides mathematical tools for measuring and analyzing information content.

### Entropy

**Shannon Entropy**
- `H(X) = -Σₓ P(x) log₂ P(x)`
- Measures uncertainty or information content
- Units: bits (base 2), nats (base e)

**Cross-Entropy**
- `H(p,q) = -Σₓ p(x) log q(x)`
- Used as loss function in classification
- Measures difference between distributions

**Kullback-Leibler (KL) Divergence**
- `D_KL(p||q) = Σₓ p(x) log(p(x)/q(x))`
- Measures how one distribution differs from another
- Always non-negative: `D_KL(p||q) ≥ 0`

### Mutual Information

**Definition**
- `I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)`
- Measures shared information between variables
- Used in feature selection and dependency analysis

**Pointwise Mutual Information (PMI)**
- `PMI(x,y) = log(P(x,y)/(P(x)P(y)))`
- Used in word association and collocation discovery

### Perplexity

**Language Model Evaluation**
- `PP(W) = 2^{H(W)} = 2^{-1/N Σᵢ log₂ P(wᵢ)}`
- Lower perplexity indicates better model
- Intrinsic evaluation metric for language models

## Distance and Similarity Measures

Various metrics quantify similarity and distance between text representations.

### Vector Distances

**Euclidean Distance**
- `d(x,y) = ||x-y||₂ = √(Σᵢ(xᵢ-yᵢ)²)`
- Sensitive to magnitude and dimensionality

**Manhattan Distance**
- `d(x,y) = ||x-y||₁ = Σᵢ|xᵢ-yᵢ|`
- Less sensitive to outliers than Euclidean

**Chebyshev Distance**
- `d(x,y) = ||x-y||∞ = maxᵢ|xᵢ-yᵢ|`
- Maximum difference along any dimension

### Similarity Measures

**Cosine Similarity**
- `cos(θ) = (x·y)/(||x||₂||y||₂)`
- Range: [-1, 1], measures angle between vectors
- Widely used in NLP for semantic similarity

**Jaccard Similarity**
- `J(A,B) = |A∩B|/|A∪B|`
- For sets of words or n-grams
- Range: [0, 1]

**Dice Coefficient**
- `D(A,B) = 2|A∩B|/(|A|+|B|)`
- Similar to Jaccard but gives more weight to intersections

### Edit Distances

**Levenshtein Distance**
- Minimum edits (insertions, deletions, substitutions) to transform one string to another
- Dynamic programming solution: O(mn) time complexity

**Hamming Distance**
- Number of positions where symbols differ
- Only for equal-length strings
- `d(x,y) = Σᵢ I(xᵢ ≠ yᵢ)`

## Matrix Decomposition

Matrix decomposition techniques are fundamental for dimensionality reduction and latent representation learning.

### Singular Value Decomposition (SVD)

**Definition**
- Any matrix A can be decomposed as: `A = UΣVᵀ`
- U: left singular vectors (m×m orthogonal matrix)
- Σ: diagonal matrix of singular values
- V: right singular vectors (n×n orthogonal matrix)

**Applications in NLP**
- Latent Semantic Analysis (LSA)
- Principal Component Analysis (PCA)
- Low-rank approximations of word co-occurrence matrices

**Truncated SVD**
- Keep only k largest singular values: `A ≈ UₖΣₖVₖᵀ`
- Reduces dimensionality while preserving most information

### Principal Component Analysis (PCA)

**Mathematical Foundation**
- Find orthogonal directions of maximum variance
- Eigendecomposition of covariance matrix: `C = QΛQᵀ`
- Q: eigenvectors (principal components)
- Λ: eigenvalues (explained variance)

**Dimensionality Reduction**
- Project data onto top k principal components
- `Y = XQₖ` where Qₖ contains k eigenvectors
- Preserves k/n proportion of total variance

### Non-negative Matrix Factorization (NMF)

**Constraint**
- Decompose A ≈ WH where W,H ≥ 0
- Useful when non-negativity has semantic meaning

**Topic Modeling Application**
- Documents × Words matrix → Topics
- W: Documents × Topics (document-topic weights)
- H: Topics × Words (topic-word weights)

## Neural Network Mathematics

Understanding the mathematics behind neural networks is crucial for modern NLP.

### Activation Functions

**Sigmoid**
- `σ(x) = 1/(1 + e^{-x})`
- Derivative: `σ'(x) = σ(x)(1 - σ(x))`
- Range: (0, 1), but suffers from vanishing gradients

**Hyperbolic Tangent (tanh)**
- `tanh(x) = (e^x - e^{-x})/(e^x + e^{-x})`
- Derivative: `tanh'(x) = 1 - tanh²(x)`
- Range: (-1, 1), zero-centered

**ReLU (Rectified Linear Unit)**
- `ReLU(x) = max(0, x)`
- Derivative: `ReLU'(x) = 1 if x > 0, else 0`
- Computationally efficient, addresses vanishing gradients

**GELU (Gaussian Error Linear Unit)**
- `GELU(x) = x · Φ(x)` where Φ is CDF of standard normal
- Approximation: `GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))`
- Used in modern transformers

### Softmax Function

**Definition**
- `softmax(xᵢ) = e^{xᵢ}/Σⱼe^{xⱼ}`
- Converts logits to probability distribution
- Output sums to 1: `Σᵢ softmax(xᵢ) = 1`

**Numerical Stability**
- Subtract maximum to prevent overflow:
- `softmax(xᵢ) = e^{xᵢ-max(x)}/Σⱼe^{xⱼ-max(x)}`

**Temperature Scaling**
- `softmax(xᵢ/T) = e^{xᵢ/T}/Σⱼe^{xⱼ/T}`
- T > 1: smoother distribution
- T < 1: sharper distribution

### Loss Functions

**Cross-Entropy Loss**
- Binary: `L = -[y log(p) + (1-y) log(1-p)]`
- Categorical: `L = -Σᵢ yᵢ log(pᵢ)`
- Used for classification tasks

**Mean Squared Error (MSE)**
- `MSE = (1/n)Σᵢ(yᵢ - ŷᵢ)²`
- Used for regression tasks
- L2 loss function

**Huber Loss**
- Combines MSE and MAE for robustness
- `L_δ(y,ŷ) = ½(y-ŷ)² if |y-ŷ| ≤ δ, else δ|y-ŷ| - ½δ²`

## Transformer Mathematics

The transformer architecture relies on sophisticated mathematical operations.

### Attention Mechanism

**Scaled Dot-Product Attention**
- `Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V`
- Q: query matrix (n×dₖ)
- K: key matrix (m×dₖ)  
- V: value matrix (m×dᵥ)
- Scale factor √dₖ prevents vanishing gradients

**Multi-Head Attention**
- `MultiHead(Q,K,V) = Concat(head₁,...,headₕ)W^O`
- `headᵢ = Attention(QW^Q_i, KW^K_i, VW^V_i)`
- Allows attending to different representation subspaces

### Position Encoding

**Sinusoidal Position Encoding**
- `PE(pos,2i) = sin(pos/10000^{2i/d_model})`
- `PE(pos,2i+1) = cos(pos/10000^{2i/d_model})`
- Provides positional information without learned parameters

**Learned Position Embeddings**
- Trainable embeddings for each position
- `PE(pos) = embedding_lookup(position_table, pos)`

### Layer Normalization

**Formula**
- `LayerNorm(x) = γ ⊙ (x-μ)/σ + β`
- μ: mean across features
- σ: standard deviation across features
- γ, β: learned scale and shift parameters

**Benefits**
- Stabilizes training in deep networks
- Reduces internal covariate shift
- Enables higher learning rates

### Feed-Forward Networks

**Structure**
- `FFN(x) = max(0, xW₁ + b₁)W₂ + b₂`
- Two linear transformations with ReLU activation
- Typical dimensions: d_model → 4×d_model → d_model

**GELU Alternative**
- `FFN(x) = GELU(xW₁ + b₁)W₂ + b₂`
- Used in BERT and other modern models

This mathematical foundation provides the necessary tools for understanding and implementing sophisticated NLP systems, from basic text processing to state-of-the-art transformer models.