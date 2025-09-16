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
- Vector: $$ v = [v_1, v_2, ..., v_n] $$
- Dimensionality typically ranges from 50 to 4096 in modern systems

**Vector Operations**
- **Addition**: $$ v + w = [v_1+w_1, v_2+w_2, ..., v_n+w_n] $$
- **Scalar Multiplication**: $$ \alpha v = [\alpha v_1, \alpha v_2, ..., \alpha v_n] $$
- **Dot Product**: $$ v \cdot w = \sum_i v_i w_i = v_1 w_1 + v_2 w_2 + ... + v_n w_n $$

**Vector Norms**
- **L1 Norm (Manhattan)**: $$ \|v\|_1 = \sum_i |v_i| $$
- **L2 Norm (Euclidean)**: $$ \|v\|_2 = \sqrt{\sum_i v_i^2} $$
- **L∞ Norm (Maximum)**: $$ \|v\|_\infty = \max_i |v_i| $$

### Matrices

**Matrix Representation**
- Batch processing of text data
- Weight matrices in neural networks
- Matrix A with dimensions m×n: $$ A \in \mathbb{R}^{m \times n} $$

**Matrix Operations**
- **Matrix Multiplication**: $$ (AB)_{ij} = \sum_k A_{ik} B_{kj} $$
- **Transpose**: $$ (A^T)_{ij} = A_{ji} $$
- **Trace**: $\text{tr}(A) = \sum_i A_{ii}$ (sum of diagonal elements)

**Important Matrix Properties**
- **Orthogonal Matrix**: $$ AA^T = A^T A = I $$
- **Symmetric Matrix**: $$ A = A^T $$
- **Positive Definite**: $x^T A x > 0$ for all non-zero x

### Eigenvalues and Eigenvectors

**Definition**
- For matrix A, eigenvector v and eigenvalue λ satisfy: $$ Av = \lambda v $$
- Used in Principal Component Analysis (PCA) for dimensionality reduction

**Characteristic Equation**
- $$ \det(A - \lambda I) = 0 $$
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
- $P(A|B) = \frac{P(A \cap B)}{P(B)}$ when $P(B) > 0$
- Foundation of language modeling: $P(\text{word}|\text{context})$

**Bayes' Theorem**
- $$ P(A|B) = \frac{P(B|A)P(A)}{P(B)} $$
- Used in classification: $P(\text{class}|\text{text}) = \frac{P(\text{text}|\text{class})P(\text{class})}{P(\text{text})}$

### Random Variables

**Discrete Random Variables**
- Probability Mass Function (PMF): $$ P(X = x) = p(x) $$
- Expected Value: $$ E[X] = \sum_x x \cdot p(x) $$
- Variance: $$ \text{Var}(X) = E[X^2] - (E[X])^2 $$

**Continuous Random Variables**
- Probability Density Function (PDF): $f(x)$
- Expected Value: $$ E[X] = \int x \cdot f(x) dx $$
- Cumulative Distribution Function: $$ F(x) = P(X \leq x) $$

### Important Distributions

**Multinomial Distribution**
- Used for word counts in documents
- PMF: $$ P(x_1,...,x_k) = \frac{n!}{x_1! \cdots x_k!} \cdot p_1^{x_1} \cdots p_k^{x_k} $$
- Where $\sum_i x_i = n$ and $\sum_i p_i = 1$

**Gaussian (Normal) Distribution**
- Used for word embeddings and neural network weights
- PDF: $$ f(x) = \frac{1}{\sigma\sqrt{2\pi}} \cdot \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right) $$
- Multivariate: $$ f(x) = \frac{1}{\sqrt{(2\pi)^k|\Sigma|}} \cdot \exp\left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right) $$

**Categorical Distribution**
- Used for word prediction in language models
- PMF: $P(X = k) = p_k$ where $k \in \{1,2,...,K\}$

## Statistics

Statistical methods are crucial for analyzing text data and evaluating model performance.

### Descriptive Statistics

**Measures of Central Tendency**
- **Mean**: $$ \mu = \frac{1}{n}\sum_i x_i $$
- **Median**: Middle value when data is sorted
- **Mode**: Most frequently occurring value

**Measures of Dispersion**
- **Variance**: $$ \sigma^2 = \frac{1}{n}\sum_i (x_i - \mu)^2 $$
- **Standard Deviation**: $$ \sigma = \sqrt{\sigma^2} $$
- **Coefficient of Variation**: $$ CV = \frac{\sigma}{\mu} $$

### Correlation and Covariance

**Covariance**
- $$ \text{Cov}(X,Y) = E[(X - E[X])(Y - E[Y])] $$
- Measures linear relationship between variables

**Pearson Correlation Coefficient**
- $$ \rho = \frac{\text{Cov}(X,Y)}{\sigma_x \sigma_y} $$
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
- t-statistic: $$ t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}} $$

**Chi-square Test**
- Test independence of categorical variables
- χ² statistic: $$ \chi^2 = \sum_{ij} \frac{(O_{ij} - E_{ij})^2}{E_{ij}} $$

## Calculus

Calculus is essential for optimization in machine learning and neural networks.

### Derivatives

**Single Variable Calculus**
- Derivative: $$ f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h} $$
- Chain Rule: $$ (f(g(x)))' = f'(g(x)) \cdot g'(x) $$

**Common Derivatives for NLP**
- $\frac{d}{dx}(\log x) = \frac{1}{x}$ (used in log-likelihood)
- $\frac{d}{dx}(e^x) = e^x$ (used in softmax)
- $$ \frac{d}{dx}(\text{sigmoid}(x)) = \text{sigmoid}(x)(1 - \text{sigmoid}(x)) $$

### Partial Derivatives

**Multivariable Functions**
- Partial derivative: $$ \frac{\partial f}{\partial x} = \lim_{h \to 0} \frac{f(x+h,y) - f(x,y)}{h} $$
- Gradient: $$ \nabla f = \left[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, ..., \frac{\partial f}{\partial x_n}\right] $$

**Hessian Matrix**
- Second-order partial derivatives: $$ H(f)_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j} $$
- Used in second-order optimization methods

### Backpropagation

**Chain Rule for Neural Networks**
- For composition $f(g(x))$: $$ \frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx} $$
- Extended to deep networks: $$ \frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial z} \cdot \frac{\partial z}{\partial w_1} $$

**Gradient Flow**
- Forward pass: Compute outputs layer by layer
- Backward pass: Compute gradients from output to input

## Optimization

Optimization algorithms train neural networks by minimizing loss functions.

### Gradient Descent

**Basic Gradient Descent**
- Update rule: $$ \theta_{t+1} = \theta_t - \alpha\nabla f(\theta_t) $$
- $\alpha$ is the learning rate

**Stochastic Gradient Descent (SGD)**
- Use single sample or mini-batch
- Update: $$ \theta_{t+1} = \theta_t - \alpha\nabla f(\theta_t; x_i, y_i) $$

**SGD with Momentum**
- $$ v_{t+1} = \beta v_t + \alpha\nabla f(\theta_t) $$
- $$ \theta_{t+1} = \theta_t - v_{t+1} $$
- $\beta$ is momentum coefficient (typically 0.9)

### Advanced Optimizers

**Adam (Adaptive Moment Estimation)**
- Combines momentum and adaptive learning rates
- $$ m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla f(\theta_t) $$
- $$ v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla f(\theta_t))^2 $$
- $$ \theta_{t+1} = \theta_t - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} $$

**AdaGrad**
- Adapts learning rate based on historical gradients
- $$ \theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{G_t + \epsilon}} \cdot \nabla f(\theta_t) $$
- Where $$ G_t = \sum_{i=1}^t (\nabla f(\theta_i))^2 $$

### Regularization

**L1 Regularization (Lasso)**
- Add penalty: $$ R(\theta) = \lambda\|\theta\|_1 = \lambda\sum_i|\theta_i| $$
- Promotes sparsity in parameters

**L2 Regularization (Ridge)**
- Add penalty: $$ R(\theta) = \lambda\|\theta\|_2^2 = \lambda\sum_i\theta_i^2 $$
- Prevents overfitting by penalizing large weights

**Elastic Net**
- Combines L1 and L2: $$ R(\theta) = \lambda_1\|\theta\|_1 + \lambda_2\|\theta\|_2^2 $$

## Information Theory

Information theory provides mathematical tools for measuring and analyzing information content.

### Entropy

**Shannon Entropy**
- $$ H(X) = -\sum_x P(x) \log_2 P(x) $$
- Measures uncertainty or information content
- Units: bits (base 2), nats (base e)

**Cross-Entropy**
- $$ H(p,q) = -\sum_x p(x) \log q(x) $$
- Used as loss function in classification
- Measures difference between distributions

**Kullback-Leibler (KL) Divergence**
- $$ D_{KL}(p\|q) = \sum_x p(x) \log\frac{p(x)}{q(x)} $$
- Measures how one distribution differs from another
- Always non-negative: $D_{KL}(p\|q) \geq 0$

### Mutual Information

**Definition**
- $$ I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) $$
- Measures shared information between variables
- Used in feature selection and dependency analysis

**Pointwise Mutual Information (PMI)**
- $$ \text{PMI}(x,y) = \log\frac{P(x,y)}{P(x)P(y)} $$
- Used in word association and collocation discovery

### Perplexity

**Language Model Evaluation**
- $$ PP(W) = 2^{H(W)} = 2^{-\frac{1}{N} \sum_i \log_2 P(w_i)} $$
- Lower perplexity indicates better model
- Intrinsic evaluation metric for language models

## Distance and Similarity Measures

Various metrics quantify similarity and distance between text representations.

### Vector Distances

**Euclidean Distance**
- $$ d(x,y) = \|x-y\|_2 = \sqrt{\sum_i(x_i-y_i)^2} $$
- Sensitive to magnitude and dimensionality

**Manhattan Distance**
- $$ d(x,y) = \|x-y\|_1 = \sum_i|x_i-y_i| $$
- Less sensitive to outliers than Euclidean

**Chebyshev Distance**
- $$ d(x,y) = \|x-y\|_\infty = \max_i|x_i-y_i| $$
- Maximum difference along any dimension

### Similarity Measures

**Cosine Similarity**
- $$ \cos(\theta) = \frac{x \cdot y}{\|x\|_2\|y\|_2} $$
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
- $$ d(x,y) = \sum_i I(x_i \neq y_i) $$

## Matrix Decomposition

Matrix decomposition techniques are fundamental for dimensionality reduction and latent representation learning.

### Singular Value Decomposition (SVD)

**Definition**
- Any matrix A can be decomposed as: $$ A = U\Sigma V^T $$
- U: left singular vectors (m×m orthogonal matrix)
- $\Sigma$: diagonal matrix of singular values
- V: right singular vectors (n×n orthogonal matrix)

**Applications in NLP**
- Latent Semantic Analysis (LSA)
- Principal Component Analysis (PCA)
- Low-rank approximations of word co-occurrence matrices

**Truncated SVD**
- Keep only k largest singular values: $$ A \approx U_k\Sigma_k V_k^T $$
- Reduces dimensionality while preserving most information

### Principal Component Analysis (PCA)

**Mathematical Foundation**
- Find orthogonal directions of maximum variance
- Eigendecomposition of covariance matrix: $$ C = Q\Lambda Q^T $$
- Q: eigenvectors (principal components)
- $\Lambda$: eigenvalues (explained variance)

**Dimensionality Reduction**
- Project data onto top k principal components
- $Y = XQ_k$ where $Q_k$ contains k eigenvectors
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
- $$ \sigma(x) = \frac{1}{1 + e^{-x}} $$
- Derivative: $$ \sigma'(x) = \sigma(x)(1 - \sigma(x)) $$
- Range: (0, 1), but suffers from vanishing gradients

**Hyperbolic Tangent (tanh)**
- $$ \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} $$
- Derivative: $$ \tanh'(x) = 1 - \tanh^2(x) $$
- Range: (-1, 1), zero-centered

**ReLU (Rectified Linear Unit)**
- $$ \text{ReLU}(x) = \max(0, x) $$
- Derivative: $$ \text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{otherwise} \end{cases} $$
- Computationally efficient, addresses vanishing gradients

**GELU (Gaussian Error Linear Unit)**
- $\text{GELU}(x) = x \cdot \Phi(x)$ where $\Phi$ is CDF of standard normal
- Approximation: $$ \text{GELU}(x) \approx 0.5x\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right)\right) $$
- Used in modern transformers

### Softmax Function

**Definition**
- $$ \text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}} $$
- Converts logits to probability distribution
- Output sums to 1: $$ \sum_i \text{softmax}(x_i) = 1 $$

**Numerical Stability**
- Subtract maximum to prevent overflow:
- $$ \text{softmax}(x_i) = \frac{e^{x_i-\max(x)}}{\sum_j e^{x_j-\max(x)}} $$

**Temperature Scaling**
- $$ \text{softmax}(x_i/T) = \frac{e^{x_i/T}}{\sum_j e^{x_j/T}} $$
- T > 1: smoother distribution
- T < 1: sharper distribution

### Loss Functions

**Cross-Entropy Loss**
- Binary: $$ L = -[y \log(p) + (1-y) \log(1-p)] $$
- Categorical: $$ L = -\sum_i y_i \log(p_i) $$
- Used for classification tasks

**Mean Squared Error (MSE)**
- $$ \text{MSE} = \frac{1}{n}\sum_i(y_i - \hat{y}_i)^2 $$
- Used for regression tasks
- L2 loss function

**Huber Loss**
- Combines MSE and MAE for robustness
- $$ L_\delta(y,\hat{y}) = \begin{cases} 
\frac{1}{2}(y-\hat{y})^2 & \text{if } |y-\hat{y}| \leq \delta \\
\delta|y-\hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise}
\end{cases} $$

## Transformer Mathematics

The transformer architecture relies on sophisticated mathematical operations.

### Attention Mechanism

**Scaled Dot-Product Attention**
- $$ \text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$
- Q: query matrix (n×$d_k$)
- K: key matrix (m×$d_k$)  
- V: value matrix (m×$d_v$)
- Scale factor $\sqrt{d_k}$ prevents vanishing gradients

**Multi-Head Attention**
- $$ \text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,...,\text{head}_h)W^O $$
- $$ \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) $$
- Allows attending to different representation subspaces

### Position Encoding

**Sinusoidal Position Encoding**
- $$ PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right) $$
- $$ PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right) $$
- Provides positional information without learned parameters

**Learned Position Embeddings**
- Trainable embeddings for each position
- $$ PE(pos) = \text{embedding\_lookup}(\text{position\_table}, pos) $$

### Layer Normalization

**Formula**
- $$ \text{LayerNorm}(x) = \gamma \odot \frac{x-\mu}{\sigma} + \beta $$
- $\mu$: mean across features
- $\sigma$: standard deviation across features
- $\gamma$, $\beta$: learned scale and shift parameters

**Benefits**
- Stabilizes training in deep networks
- Reduces internal covariate shift
- Enables higher learning rates

### Feed-Forward Networks

**Structure**
- $$ \text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2 $$
- Two linear transformations with ReLU activation
- Typical dimensions: $d_{\text{model}} \to 4 \times d_{\text{model}} \to d_{\text{model}}$

**GELU Alternative**
- $$ \text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2 $$
- Used in BERT and other modern models

This mathematical foundation provides the necessary tools for understanding and implementing sophisticated NLP systems, from basic text processing to state-of-the-art transformer models.