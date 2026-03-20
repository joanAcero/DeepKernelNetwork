# Deep Kernel Networks: Neural Architectures with SVM Foundations

**Joan Acero-Pousa** — Master in Innovation and Research in Informatics (Machine Learning track)  
Universitat Politècnica de Catalunya — School of Computer Science  
Supervisor: Lluís A. Belanche-Muñoz

---

## 1. Background: The Multilayer SVM

### 1.1 Motivation

Support Vector Machines are among the most theoretically well-founded classifiers in the machine learning literature. Their solid grounding in statistical learning theory, the use of the kernel trick to operate in high-dimensional reproducing kernel Hilbert spaces (RKHS), and their built-in regularisation through margin maximisation have made them a reference method for decades. However, the standard kernel SVM suffers from a fundamental scalability problem: forming and solving the kernel matrix requires $O(n^2)$ space and $O(n^3)$ time, making exact training infeasible for large datasets.

A natural response is to exploit kernel approximation. Random Fourier Features (RFF), introduced by Rahimi and Recht (2007), approximate a shift-invariant kernel $k(x, x') \approx \hat{\phi}(x)^\top \hat{\phi}(x')$ via a randomised feature map:

$$\hat{\phi}(x) = \sqrt{\frac{2}{D}} \left[\cos(\omega_1^\top x + b_1),\ \ldots,\ \cos(\omega_D^\top x + b_D)\right]^\top \in \mathbb{R}^D$$

where $\omega_j \sim p(\omega)$ are frequencies drawn from the Fourier transform of the kernel and $b_j \sim \mathcal{U}[0, 2\pi]$ are random phases. For the Gaussian RBF kernel with bandwidth $\gamma$, $p(\omega) = \mathcal{N}(0, 2\gamma I)$. Replacing the kernel matrix with RFF features reduces the learning problem to training a linear model on $\hat{\phi}(X)$, which can be solved in $O(nD)$ time.

Stacking RFF layers with linear SVMs at each level is a natural extension — if a single kernel layer can discover non-linear structure, successive layers may capture increasingly complex patterns. This is the motivation for the **Multilayer SVM (ML-SVM)** proposed in Acero-Pousa and Belanche-Muñoz (ESANN 2025).

### 1.2 Architecture

The ML-SVM processes data through a cascade of RFF transformations and linear SVM classifiers. For $k = 1, \ldots, L$ hidden layers, the computation is:

$$h^{(k)} = \hat{\phi}^{(k)}\!\left(\mathrm{SVM}^{(k-1)}\right) = \sqrt{\frac{2}{D}} \left[\cos\!\left(\omega_1^\top \mathrm{SVM}^{(k-1)} + b_1\right),\ \ldots\right]^\top$$

$$\mathrm{SVM}^{(k)} = w_k^\top h^{(k)} + b_k$$

where $\mathrm{SVM}^{(0)} = X$ is the raw input. Each SVM is a Linear SVM solved as a convex QP, and the RFF frequencies $\{\omega_j\}$ are sampled once and frozen. The final prediction is the output of the last SVM.

### 1.3 Training

The ML-SVM is trained **greedily**: each layer is optimised independently given the output of the previous one, with no backpropagation through the network. The bandwidth $\gamma$ for each layer's RFF map is estimated from the data using the `sigest` heuristic (Caputo et al. 2002 / kernlab), which selects $\gamma$ based on the distribution of pairwise distances in the input.

### 1.4 Structural Limitations

Despite competitive results on several benchmarks, the ML-SVM has two structural pathologies that limit its representational power:

**P1 — No learnable parameters.** The RFF frequencies $\{\omega_j\}$ are drawn once from a fixed prior and never updated. The network approximates a single fixed kernel and cannot adapt its internal representation to the geometry of the task. No matter how many layers are added, the basis functions remain static.

**P2 — Dimensional bottleneck between layers.** The output of $\mathrm{SVM}^{(k)}$ has dimension $P$ (the number of classes) or 1 (in binary problems). Layer $k+1$ therefore applies its RFF map to a $P$-dimensional vector, discarding all the geometric structure accumulated in the $D$-dimensional RFF space of layer $k$. The dimensional profile is:

$$d \xrightarrow{\text{RFF}} D \xrightarrow{\text{SVM}} P \xrightarrow{\text{RFF}} D \xrightarrow{\text{SVM}} P \xrightarrow{} \cdots$$

Because $P \ll D$ in all practical settings, each hidden layer is starved of the representation built by the previous one. Adding depth provides marginal benefit at best.

---

## 2. Deep Kernel Networks

The **Deep Kernel Network (DKN)** is the architecture proposed in this thesis. It addresses both limitations of the ML-SVM simultaneously: it introduces learnable inter-layer weight matrices that resolve P1, and it redesigns the inter-layer communication to eliminate the bottleneck of P2. Two training regimes are proposed, yielding two model variants: **DKN-AGOP** and **DKN-Alignment**.

### 2.1 Shared Inference Graph

Both DKN variants share the same inference-time computation graph. Let $h^{(0)} = x \in \mathbb{R}^d$. For each layer $k = 1, \ldots, L$:

$$z^{(k)} = W^{(k)} h^{(k-1)}, \qquad W^{(k)} \in \mathbb{R}^{r \times d_{k-1}} \tag{1}$$

$$h^{(k)} = \hat{\phi}^{(k)}\!\left(z^{(k)}\right) \in \mathbb{R}^D \tag{2}$$

The final prediction is made by a linear LS-SVM trained on $h^{(L)}$:

$$\hat{y} = w_{L+1}^\top h^{(L)} + b_{L+1} \tag{3}$$

The dimensional profile is now:

$$d \xrightarrow{W^{(1)}} r \xrightarrow{\text{RFF}} D \xrightarrow{W^{(2)}} r \xrightarrow{\text{RFF}} D \xrightarrow{} \cdots \xrightarrow{} D \xrightarrow{\text{SVM}} \hat{y}$$

where $r$ is the rank hyperparameter — a free design choice independent of $P$. This eliminates P2: each layer receives a full $D$-dimensional representation from the previous one, and the inter-layer dimensionality $r$ is chosen by the practitioner, not forced by the number of classes.

The learnable weight matrices $\{W^{(k)}\}$ also resolve P1: unlike the frozen random projections of the ML-SVM, the DKN learns task-adaptive projections that orient the RFF map toward the directions most relevant for the classification task.

### 2.2 Relationship to Mehrkanoon (2018)

Mehrkanoon and Suykens (2018) propose a hybrid network with the same inference graph as equations (1)–(3). Their model learns $W^{(k)}$ by global backpropagation through the full network, optimising a cross-entropy loss end-to-end. Although this solves P1 and P2, the SVM structure — convexity, margin maximisation, per-layer closed-form solution — disappears entirely. The DKN differs from Mehrkanoon's model exclusively in **how $W^{(k)}$ is learned**: each DKN training method replaces global backpropagation with a layer-wise, locally supervised procedure that preserves the convex structure of the SVM sub-problems.

---

## 3. DKN-AGOP

### 3.1 Theoretical Grounding

DKN-AGOP learns the weight matrices $\{W^{(k)}\}$ using the **Average Gradient Outer Product (AGOP)**, a tool from the theory of Recursive Feature Machines (Radhakrishnan et al. 2022) justified as a greedy gradient approximation by Gan and Poggio (2024).

The AGOP of a predictor $f$ at layer $k$ is defined as:

$$M^{(k)} = \frac{1}{n} \sum_{i=1}^{n} \nabla_{h^{(k-1)}} f^{(k)}(x_i)\, \nabla_{h^{(k-1)}} f^{(k)}(x_i)^\top \in \mathbb{R}^{d_{k-1} \times d_{k-1}} \tag{4}$$

$M^{(k)}$ is a positive semi-definite matrix whose eigenvectors are directions in the input space of layer $k$, ordered by how much the predictor's output varies along them. The top eigenvectors identify the subspace of $h^{(k-1)}$ most informative for predicting $y$.

Gan and Poggio (2024) prove that for a predictor of the form $f(x) = h(Wx)$ initialised at $W = 0$, gradient descent on $W$ produces $W^\top W \propto M$ at each step, where $M$ is the AGOP. The AGOP is therefore a closed-form, backpropagation-free substitute for one step of gradient descent on the layer weights.

### 3.2 Training Algorithm

The DKN-AGOP training procedure is fully greedy and backpropagation-free. SVMs appear only as training oracles; they are discarded after use and do not participate in inference.

**Algorithm 1: DKN-AGOP Training**

---

**Input:** Data $(X, y)$, depth $L$, rank $r$, RFF dimension $D$, regularisation $C$  
**Output:** Weight matrices $\{W^{(k)}\}_{k=1}^L$, final SVM weights $w_{L+1}$

1. $h \leftarrow X$
2. **for** $k = 1$ **to** $L$ **do**
3. $\quad$ Compute $WH = h \cdot (W^{(k)})^\top$
4. $\quad$ Fit $\hat{\phi}^{(k)}$ as an RFF map on $WH$ (bandwidth $\gamma$ estimated by `sigest`)
5. $\quad$ $\Phi \leftarrow \hat{\phi}^{(k)}(WH)$
6. $\quad$ Solve oracle LS-SVM: $w_k \leftarrow \arg\min_w \frac{1}{2}\|w\|^2 + C\sum_i \ell(w^\top \Phi_i, y_i)$
7. $\quad$ Compute AGOP: $M^{(k)} \leftarrow \frac{1}{n} \sum_i J_i^\top J_i$ where $J_i = \nabla_{h} f^{(k)}(x_i) \in \mathbb{R}^{P \times d_{k-1}}$
8. $\quad$ $W^{(k+1)} \leftarrow$ top-$r$ eigenvectors of $M^{(k)}$ (rows)
9. $\quad$ Discard oracle $w_k$
10. $\quad$ $h \leftarrow \hat{\phi}^{(k)}(h \cdot (W^{(k)})^\top)$ — propagate representation forward
11. **end for**
12. Fit final LS-SVM $w_{L+1}$ on $(h, y)$ — kept for inference

---

### 3.3 AGOP Gradient Computation

For the oracle LS-SVM with RFF features $\hat{\phi}(z) = \sqrt{2/D}\,\cos(z\Omega^\top + b)$, where $\Omega \in \mathbb{R}^{D \times r}$ are the RFF frequencies and $b \in \mathbb{R}^D$ the phases, the Jacobian of class $p$ with respect to the block input $h$ is:

$$\frac{\partial f_p}{\partial h_i} = \left(-\sqrt{\frac{2}{D}}\,\sin(WH_i\,\Omega^\top + b) \odot w_p\right)\,\Omega\, W \in \mathbb{R}^{1 \times d_{k-1}}$$

The full AGOP accumulates over all $n$ samples and all $P$ classes:

$$M^{(k)} = \frac{1}{n} \sum_{i=1}^{n} \sum_{p=1}^{P} \left(\frac{\partial f_p}{\partial h_i}\right)^\top \left(\frac{\partial f_p}{\partial h_i}\right)$$

Note that for binary classification, $\mathrm{sklearn}$'s `RidgeClassifier` returns a 1D coefficient vector rather than the $(1, D)$ matrix documented in the API; this must be normalised with `np.atleast_2d` before the AGOP computation.

### 3.4 The Rank Parameter

The rank $r$ controls how many AGOP eigenvectors are retained in $W^{(k+1)}$. Geometrically, $r$ is the dimensionality of the subspace the next layer's RFF operates in. Too large an $r$ retains near-zero eigenvalue directions that are noise; too small an $r$ discards genuinely discriminative directions. The optimal $r$ reflects the **intrinsic discriminative dimensionality** of the problem at that layer: for a dataset where only a few directions are informative (e.g. MADELON, with 20 informative features among 500), a small $r$ suffices; for a dataset with distributed signal (e.g. ionosphere), a larger $r$ is required. Empirically, $r$ is selected by nested cross-validation.

Crucially, $W^{(k)} \in \mathbb{R}^{r \times d_{k-1}}$ has exactly $r$ rows — the previous implementation padded with zeros to a larger $d_k > r$, which silently caused the RFF to operate in an effectively $r$-dimensional space disguised as $d_k$-dimensional, wasting $(d_k - r)/d_k$ of the RFF capacity and compounding noise across layers. This bug has been corrected: rank is the sole dimension-control parameter.

### 3.5 Key Properties

| Property | Value |
|---|---|
| Training objective per layer | Convex (LS-SVM / Ridge regression) |
| Backpropagation required | No |
| SVMs appear at inference | Final layer only |
| Inter-layer bottleneck | None ($r$ is a free hyperparameter) |
| Feature adaptation | Yes (via AGOP-guided $W$) |
| Theoretical grounding for $W$ update | Gan & Poggio (2024): one step of gradient descent on $W$ |

---

## 4. DKN-Alignment

### 4.1 Theoretical Grounding

DKN-Alignment learns $W^{(k)}$ by maximising the **kernel-target alignment** of the RFF features produced by that layer. Kernel-target alignment, introduced by Cristianini et al. (2001) and extended by Cortes et al. (2012), measures how well a kernel matrix $K$ matches the ideal kernel defined by the labels:

$$\hat{A}(K, y) = \frac{\langle K, yy^\top \rangle_F}{\|K\|_F \cdot \|yy^\top\|_F}$$

Cortes et al. (2012) prove that maximising alignment minimises an upper bound on the SVM generalisation error, providing a direct theoretical link between the alignment objective and classification performance — without requiring a QP to be solved at each step.

For RFF features $\Phi = \hat{\phi}(Wh) \in \mathbb{R}^{n \times D}$, the empirical kernel matrix is $K \approx \Phi\Phi^\top$ and the alignment becomes:

$$\mathcal{A}(W) = \frac{\|\Phi^\top Y\|_F^2}{\|\Phi\Phi^\top\|_F} \tag{5}$$

where $Y \in \mathbb{R}^{n \times P}$ is the one-hot label matrix. DKN-Alignment maximises $\mathcal{A}(W)$ — equivalently minimises $-\mathcal{A}(W)$ — at each layer by gradient descent through $W$.

### 4.2 Training Algorithm

**Algorithm 2: DKN-Alignment Training**

---

**Input:** Data $(X, y)$, depth $L$, RFF dimension $D$, regularisation $C$, learning rate $\eta$, epochs $T$  
**Output:** Frozen blocks $\{(W^{(k)}, \Omega^{(k)}, b^{(k)})\}_{k=1}^L$, final SVM weights $w_{L+1}$

1. $h \leftarrow X$; $Y \leftarrow$ one-hot$(y)$
2. **for** $k = 1$ **to** $L$ **do**
3. $\quad$ Initialise $W^{(k)}$ as identity-like; sample frozen $\Omega^{(k)}, b^{(k)}$ (bandwidth by `sigest`)
4. $\quad$ **for** $t = 1$ **to** $T$ **do** (Adam optimiser)
5. $\quad\quad$ $\Phi_B \leftarrow \hat{\phi}^{(k)}(W^{(k)} h_B)$ on minibatch $B$
6. $\quad\quad$ $\mathcal{L} \leftarrow -\mathcal{A}(W^{(k)}) = -\|\Phi_B^\top Y_B\|_F^2\, /\, \|\Phi_B \Phi_B^\top\|_F$
7. $\quad\quad$ $W^{(k)} \leftarrow W^{(k)} - \eta \nabla_{W^{(k)}} \mathcal{L}$
8. $\quad$ **end for**
9. $\quad$ Freeze $W^{(k)}$; $h \leftarrow \hat{\phi}^{(k)}(W^{(k)} h)$ — propagate forward
10. **end for**
11. Fit final LS-SVM $w_{L+1}$ on $(h, y)$ — kept for inference

---

### 4.3 Distinction from DKN-AGOP

The two DKN variants share the same inference graph but differ in how $W^{(k)}$ is learned:

| Aspect | DKN-AGOP | DKN-Alignment |
|---|---|---|
| How $W^{(k)}$ is learned | Top eigenvectors of AGOP | Gradient descent on alignment loss |
| Sub-problem at each layer | Convex (LS-SVM, closed form) | Non-convex (Adam on $\mathcal{A}$) |
| Backpropagation | No | Yes, within a single layer |
| RFF frozen during $W$ update | Yes (re-fitted per AGOP step) | Yes (frozen at block construction) |
| Theoretical grounding | Gan & Poggio (2024) | Cortes et al. (2012) |
| Global backprop through network | No | No |

The critical distinction is that DKN-Alignment does use gradient descent, but only **locally within each layer** — gradients do not flow across layer boundaries. The architecture is still trained greedily; each block is frozen before the next is constructed. This preserves the layer-wise interpretability and avoids the vanishing gradient problems of end-to-end training, while allowing the objective to be differentiable and expressive.

### 4.4 Key Properties

| Property | Value |
|---|---|
| Training objective per layer | Non-convex (alignment maximisation) |
| Backpropagation required | Within-layer only (no global backprop) |
| SVMs appear at inference | Final layer only |
| Inter-layer bottleneck | None |
| Feature adaptation | Yes (via alignment-guided $W$) |
| Theoretical grounding for objective | Cortes et al. (2012) |

---

## 5. Comparison of All Three Models

| | ML-SVM | DKN-AGOP | DKN-Alignment |
|---|---|---|---|
| **Learnable inter-layer weights** | No | Yes (AGOP) | Yes (alignment) |
| **Inter-layer bottleneck** | Yes ($P$ dims) | No ($r$ dims, free) | No ($d_k$ dims, free) |
| **Training convexity** | Per-layer convex | Per-layer convex | Per-layer non-convex |
| **Backpropagation** | No | No | Within-layer only |
| **SVM at inference** | All layers | Final layer only | Final layer only |
| **Depth utility** | Marginal (bottleneck) | Meaningful | Meaningful |
| **Bandwidth selection** | `sigest` | `sigest` per step | `sigest` at construction |

---

## 6. Theoretical Connections

The DKN architecture sits at the intersection of three lines of theoretical work:

**Recursive Feature Machines (Radhakrishnan et al. 2022).** RFMs show that iterating between kernel regression and AGOP updates converges to a kernel machine that learns the same features as a deep network trained by backpropagation, matching deep network performance on tabular benchmarks. DKN-AGOP applies this insight layer-by-layer, using the SVM oracle at each layer as the kernel regressor.

**AGOP as greedy gradient descent (Gan and Poggio 2024).** For a predictor $f(x) = h(Wx)$ initialised at $W = 0$, gradient descent on $W$ produces $W^\top W \propto M$ at each step, where $M$ is the AGOP. This justifies using the AGOP eigenvectors as the update for $W^{(k+1)}$: it recovers the Gram matrix that gradient descent would produce, without backpropagating through the network.

**Kernel-target alignment (Cortes et al. 2012).** Maximising alignment between the empirical kernel and the ideal label kernel directly minimises a bound on the SVM generalisation error. This gives DKN-Alignment a principled training objective that is independent of — and does not require — solving an inner QP at each step.

---

## References

Acero-Pousa, J. and Belanche-Muñoz, L. A. (2025). A new approach to multilayer SVMs. In *Proceedings of ESANN 2025*, pages 467–472.

Cortes, C., Mohri, M., and Rostamizadeh, A. (2012). Algorithms for learning kernels based on centered alignment. *Journal of Machine Learning Research*, 13:795–828.

Cristianini, N., Shawe-Taylor, J., Elisseeff, A., and Kandola, J. (2001). On kernel-target alignment. In *Advances in Neural Information Processing Systems*, volume 14.

Gan, Y. and Poggio, T. (2024). For HyperBFs AGOP is a greedy approximation to gradient descent. CBMM Memo No. 148.

Mehrkanoon, S. and Suykens, J. A. K. (2018). Deep hybrid neural-kernel networks using random Fourier features. *Neurocomputing*, 298:46–54.

Rahimi, A. and Recht, B. (2007). Random features for large-scale kernel machines. In *Advances in Neural Information Processing Systems*, volume 20.

Radhakrishnan, A., Beaglehole, D., Pandit, P., and Belkin, M. (2022). Mechanism of feature learning in deep fully connected networks and kernel machines that recursively learn features. *arXiv preprint arXiv:2212.13881*.
