# BERT (Bidirectional Encoder Representations from Transformers) — Interview Guide

---

## 1. What is BERT?

BERT is a **Transformer-based encoder model** designed for **Natural Language Understanding (NLU)** tasks.

* Introduced by Google in 2018
* Uses **bidirectional context (left + right simultaneously)**
* Pretrained on large unlabeled corpora
* Fine-tuned for downstream tasks

---

## 2. Core Idea

> BERT understands a word using its **entire surrounding context**, not just left-to-right.

Example:
“bank” → river bank vs financial bank (context decides meaning)

---

## 3. Architecture

* Based on **Transformer Encoder**
* No decoder
* Stacked layers:

  * BERT-base → 12 layers, 768 hidden size, 12 heads
  * BERT-large → 24 layers, 1024 hidden size, 16 heads

### Each layer contains:

1. Multi-head self-attention
2. Feedforward network (FFN)
3. Residual connections
4. Layer normalization

---

## 4. Input Representation

Each token embedding is the sum of:

```
Embedding = Token + Position + Segment
```

### Special Tokens

* `[CLS]` → classification token (start of sequence)
* `[SEP]` → separates sentences
* `[PAD]` → padding
* `[MASK]` → used during training

---

## 5. Self-Attention (Core Math)

Each token is projected into:

* Query (Q)
* Key (K)
* Value (V)

```
Q = XWq,  K = XWk,  V = XWv
```

### Attention formula:

```
Attention(Q, K, V) = softmax((QKᵀ) / √d_k) V
```

### Intuition:

* Q → what I am looking for
* K → what I offer
* V → actual information

---

## 6. Pretraining Objectives

### (1) Masked Language Modeling (MLM)

* Mask ~15% tokens
* Predict missing tokens

Example:

```
"I love [MASK]" → "NLP"
```

---

### (2) Next Sentence Prediction (NSP)

* Input:

```
[CLS] Sentence A [SEP] Sentence B [SEP]
```

* Output:

  * IsNext / NotNext

---

## 7. Fine-Tuning

Add a task-specific head and train on labeled data.

### Common tasks:

### 1. Text Classification

* Use `[CLS]` embedding
* Output → class label

### 2. Named Entity Recognition (NER)

* Token-level classification

### 3. Question Answering (QA)

* Predict:

  * start index
  * end index

### 4. Sentence Pair Tasks

* Paraphrase detection
* Natural Language Inference

---

## 8. Question Answering (Important)

Input:

```
[CLS] Question [SEP] Context [SEP]
```

Output:

* start_logits → probability per token
* end_logits → probability per token

Final answer:

* Extract span from context using argmax

---

## 9. What BERT Returns

Depends on model variant:

* `BertModel` → embeddings
* `BertForClassification` → logits
* `BertForQA` → start & end logits

---

## 10. Encoder vs Decoder

| Model | Type            | Capability    |
| ----- | --------------- | ------------- |
| BERT  | Encoder         | Understanding |
| GPT   | Decoder         | Generation    |
| T5    | Encoder-Decoder | Both          |

BERT:

* ❌ Cannot generate text
* ✅ Strong for understanding

---

## 11. Cross-Attention (BERT + Decoder)

If used in encoder-decoder setup:

* Encoder runs once
* Decoder generates step-by-step

### Cross-attention:

* Q → decoder
* K, V → encoder

Shapes:

```
Q: (B, T, H)
K,V: (B, S, H)
Scores: (B, h, T, S)
Output: (B, T, H)
```

---

## 12. Training Flow

1. Input text → tokenized
2. Convert to embeddings
3. Pass through Transformer layers
4. Compute MLM + NSP loss
5. Backpropagation (Adam optimizer)

---

## 13. Advantages

* Deep bidirectional understanding
* Works well with limited labeled data
* Strong baseline for NLU tasks

---

## 14. Limitations

* Cannot generate text
* Computationally heavy
* NSP not very useful (removed in RoBERTa)

---

## 15. Variants

* RoBERTa → better training, no NSP
* ALBERT → parameter sharing
* DistilBERT → smaller, faster
* BERT-large → bigger version

---

## 16. Practical Usage (Hugging Face)

```python
from transformers import BertForQuestionAnswering

model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
```

---

## 17. Interview One-Liner

> BERT is a bidirectional Transformer encoder trained using masked language modeling and next sentence prediction, used to generate contextual embeddings for downstream language understanding tasks via fine-tuning.

---

## 18. Key Takeaways

* Encoder-only architecture
* Uses self-attention (Q, K, V)
* Pretrained on large corpora
* Fine-tuned for specific tasks
* Outputs embeddings/logits, not text

---
