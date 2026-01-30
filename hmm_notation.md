# Updated Notation on the HMM for the project of robot self-localization:

## 2. Theoretical Framework: The Knowledge Base

This section is the shared reference for the team’s math and must match the implementation interfaces. We model robot self-localization as a discrete Hidden Markov Model (HMM) over a grid with obstacles.

We define the HMM parameters as:

$\theta = (\mathcal{S}, A, B, \pi)$

where all quantities are defined over **free cells only**.

### 2.1 The Hidden Markov Model Structure

We distinguish two stochastic processes:

1. **Hidden state process (**$x_t$**):**
    
    ($x_t \in \mathcal{S}$) is the robot’s true cell at time (t).
    
    The state space is:
    
    $\mathcal{S} = {s_1, s_2, \dots, s_K}$
    
    where each $s_i$ corresponds to a **free** grid cell. Thus:
    
    $K = N^2 -$ #obstacles
    
    The dynamics satisfy the first-order Markov property:
    
    $P(x_t \mid x_{0:t-1}) = P(x_t \mid x_{t-1})$
    
2. **Observation process (**$y_t$**):**
    
    Each observation is a 4-bit binary reading from the sensors:
    
    $y_t = (y_t^N, y_t^E, y_t^S, y_t^W) \in {0,1}^4$
    
    corresponding to obstacle presence at distance 1 in the four cardinal directions.
    
    There are:
    
    $M = 2^4 = 16$
    
    possible observation symbols.
    
    The conditional independence assumption is:
    
    $P(y_t \mid x_t, x_{t-1}, y_{t-1}) = P(y_t \mid x_t)$
    

**Joint distribution (length (T)):**

$P(x_{1:T}, y_{1:T}) = P(x_1)\prod_{t=2}^{T} P(x_t \mid x_{t-1})\prod_{t=1}^{T} P(y_t \mid x_t)$

**Initial distribution (**$\pi$**):**

For global localization:

$\pi_i = P(x_1 = s_i) = \frac{1}{K}$

unless a non-uniform prior is explicitly introduced for experiments.

---

### 2.2 Derivation of the Grid Transition Matrix $A$

The transition matrix describes the random walk motion model:

$a_{ij} = P(x_t = s_j \mid x_{t-1} = s_i)$

so $A$ is a ($K \times K$) matrix.

**Valid move set:**

From cell $s_i$, the robot may:

- **stay** in place, or
- move to any of its **8 neighbors** (Moore neighborhood),
    
    as long as the target cell is free and inside the grid.
    

Let:

$\mathcal{N}(s_i) = { \text{valid next states from } s_i}$

including $s_i$ itself.

**Uniform random walk over valid moves:**

$$
⁍
$$

**Key implications:**

- $A$ is **row-stochastic**:
    
    $$
    ⁍
    $$
    
- The number of valid moves ($|\mathcal{N}(s_i)|$) depends on obstacles and borders.
    
    Open areas have up to 9 options (8 neighbors + stay), while corners/corridors have fewer.
    

---

### 2.3 Derivation of the Sensor Likelihood Matrix (B)

The observation (emission) model is:

$b_i(k) = P(y_t = o_k \mid x_t = s_i)$

where:

- $o_k$ is the $k$-th possible 4-bit observation,
- $k \in {1, \dots, M}$,
- $M = 16$.

Thus $B$ has shape $K \times 16$.

**True local signature from the map:**

For each state $s_i$, define the deterministic 4-bit pattern:

$z(s_i) = (z^N, z^E, z^S, z^W)$

where $z^d = 1$ if there is an obstacle (or boundary) one cell away in direction $d$, else $0$.

**Sensor error model:**

Each bit is independently flipped with probability $p_e$:

$P(y^d = z^d) = 1 - p_e,\quad$

$P(y^d \neq z^d) = p_e$

**Likelihood of a full 4-bit observation:**

$$
P(y \mid x_t = s_i)
= \prod_{d \in {N,E,S,W}}
\begin{cases}
1-p_e, & y^d = z^d(s_i)\\
p_e, & y^d \neq z^d(s_i)
\end{cases}
$$

Equivalently using Hamming distance (H(\cdot,\cdot)):

$$
⁍
$$

**Implementation note:**

Precompute $z(s_i)$ for all free states, then fill each row of $B$ for all 16 possible observations. This is efficient and avoids recomputing sensor geometry at each time step.

---

### 2.4 Filtering Objective (for implementation consistency)

The self-localization task is to compute the belief:

$$
p_t(i) = P(x_t = s_i \mid y_{1:t}, \text{map})
$$

Using the standard HMM filter:

**Prediction:**

$$
\hat{p}t = A^\top p{t-1}
$$

**Correction:**

$$
p_t(i) \propto b_i(y_t)\hat{p}_t(i)
$$

followed by normalization.

This defines the exact computation your `get_state_probabilities()` function should implement.