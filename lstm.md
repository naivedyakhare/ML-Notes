# LSTM (Long Short-Term Memory) — Interview Guide

---

## 1. What is LSTM?

LSTM is a type of **Recurrent Neural Network (RNN)** designed to handle **sequential data** and overcome the **vanishing gradient problem**.

* Introduced by Hochreiter & Schmidhuber (1997)
* Processes data **step-by-step (time series)**
* Maintains long-term dependencies using memory

---

## 2. Core Idea

> LSTM selectively **remembers or forgets information** using gates.

Unlike vanilla RNN:

* Can retain information over long sequences
* Avoids vanishing gradients

---

## 3. Architecture Overview

Each LSTM cell contains:

* Cell state (C_t) → long-term memory
* Hidden state (h_t) → short-term output
* Gates:

  1. Forget gate
  2. Input gate
  3. Output gate

---

## 4. Key Components

### Inputs at time step (t):

* Current input → (x_t)
* Previous hidden state → (h_{t-1})
* Previous cell state → (C_{t-1})

---

## 5. Gates and Equations

### (1) Forget Gate

Decides what to remove from memory:

```id="lstm1"
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
```

* Output between 0 and 1
* 0 → forget completely
* 1 → keep fully

---

### (2) Input Gate

Decides what new information to store:

```id="lstm2"
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
```

Candidate values:

```id="lstm3"
\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)
```

---

### (3) Cell State Update

```id="lstm4"
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
```

* Combines old memory and new information

---

### (4) Output Gate

```id="lstm5"
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
```

Final hidden state:

```id="lstm6"
h_t = o_t \odot \tanh(C_t)
```

---

## 6. Intuition

* **Forget gate** → removes irrelevant info
* **Input gate** → adds new relevant info
* **Cell state** → carries memory across time
* **Output gate** → decides what to expose

---

## 7. Why LSTM works

* Cell state provides a **direct path for gradients**
* Gates control information flow
* Helps preserve long-term dependencies

---

## 8. Training

* Uses **Backpropagation Through Time (BPTT)**
* Loss computed across sequence
* Optimized using gradient descent (Adam, SGD)

---

## 9. Types of LSTM Architectures

### 1. Many-to-One

* Input: sequence
* Output: single value
* Example: sentiment analysis

---

### 2. One-to-Many

* Input: single value
* Output: sequence
* Example: text generation

---

### 3. Many-to-Many

* Input: sequence
* Output: sequence
* Example: translation

---

## 10. Applications

* Time series forecasting
* Speech recognition
* Language modeling
* Machine translation (pre-transformer era)
* Sequence classification

---

## 11. Advantages

* Handles long-term dependencies
* Reduces vanishing gradient problem
* Works well for sequential data

---

## 12. Limitations

* Slow (sequential computation)
* Hard to parallelize
* Less effective than Transformers for large-scale NLP
* Memory intensive

---

## 13. LSTM vs RNN vs Transformer

| Feature           | RNN  | LSTM   | Transformer |
| ----------------- | ---- | ------ | ----------- |
| Long-term memory  | ❌    | ✅      | ✅           |
| Parallelization   | ❌    | ❌      | ✅           |
| Speed             | Slow | Slower | Fast        |
| Performance (NLP) | Low  | Medium | High        |

---

## 14. Practical Usage (PyTorch)

```python id="lstm7"
import torch.nn as nn

lstm = nn.LSTM(input_size=100, hidden_size=256, num_layers=2)
```

---

## 15. Interview One-Liner

> LSTM is a gated RNN architecture that maintains long-term dependencies using a cell state and three gates (input, forget, output), trained via backpropagation through time.

---

## 16. Key Takeaways

* Sequential model with memory
* Uses gates to control information flow
* Solves vanishing gradient problem
* Predecessor to Transformer-based models

---
