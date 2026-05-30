# 🏷️ Part-of-Speech Tagging using Hidden Markov Models (HMMs)

A comprehensive implementation of Hidden Markov Models for Part-of-Speech tagging from scratch, featuring the Viterbi algorithm and comparative analysis against state-of-the-art baseline models.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![NLTK](https://img.shields.io/badge/NLTK-Brown%20Corpus-green.svg)](https://www.nltk.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-orange.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)

---

## 📋 Assignment Overview

| **Aspect** | **Details** |
|------------|-----------|
| **Course** | Natural Language Processing (S2-25_AIMLCZG530) |
| **Institution** | BITS Pilani |
| **Instructor** | Ajay Naik (Course LF) |
| **Task** | Implement HMM-based POS tagger with Viterbi decoding + baseline comparison |
| **Dataset** | NLTK Brown Corpus (Training: `news` \| Testing: `fiction`) |
| **Total Marks** | 11 Marks (including +1 for BITS Lab) |

### 🎯 Core Objective
Develop a probabilistic sequence labeling model from first principles, implement dynamic programming inference, and critically evaluate performance against industrial-strength baselines.

---

## 📚 Learning Objectives

By completing this assignment, you will master:

1. **Probabilistic Framework** 
   - Formulate and implement HMM transition (A) and emission (B) matrices
   - Use Maximum Likelihood Estimation (MLE) for parameter learning

2. **Viterbi Algorithm** 
   - Design and implement dynamic programming-based sequence decoding
   - Achieve optimal POS tag prediction for any word sequence

3. **Real-world NLP Challenges** 
   - Address out-of-vocabulary (OOV) words through smoothing techniques
   - Implement back-off models for sparse data handling

4. **Model Evaluation** 
   - Perform rigorous sequence labeling assessment with standard metrics
   - Generate confusion matrices and conduct error analysis
   - Compare against baseline models critically

---

## 🏗️ Project Structure & Marks Breakdown

### **Part 1: Data Preprocessing & Exploratory Data Analysis (EDA)** `[2 Marks]`

Extract transition and emission states from a tagged corpus through systematic preprocessing.

#### Sub-tasks:
- ✅ **Data Loading** 
  - Load NLTK Brown corpus from designated categories
  - Structure data for processing

- 📊 **Exploratory Data Analysis** 
  - Visualize POS tag distributions (bar chart required)
  - Analyze sentence length statistics
  - Compute vocabulary coverage metrics
  - Identify frequency-based tag relationships

- 🔤 **Vocabulary Handling & Unknown Words** 
  - Define OOV strategy and thresholds
  - Create `<UNK>` token representation
  - Analyze impact of vocabulary size

- 📝 **Sentence Padding** 
  - Implement padding/masking for uniform processing
  - Add start/end markers (`<START>`, `<END>`)

- 🎯 **Data Splitting** 
  - Create 80-20 train-test splits
  - Ensure stratified sampling by tag distribution

**📌 Deliverable:** 
- Bar chart of POS tag distribution
- Statistical summary DataFrame

---

### **Part 2: Model Architecture & Parameter Estimation** `[4 Marks]`

Build a **bigram HMM** from scratch using explicit probability matrix calculations.

#### **2a) Probability Matrix Estimation**

**Transition Probability Matrix (A):**
```
P(tag_i | tag_j) = Count(tag_j → tag_i) / Count(tag_j)

Example:
P(NN | DET) = # of times (DET→NN) appears / # of times DET appears
```

**Emission Probability Matrix (B):**
```
P(word | tag) = Count(word, tag) / Count(tag)

Example:
P("book" | NN) = # of times "book" tagged as NN / # of times NN appears
```

**Implementation Steps:**
1. Extract all (prev_tag → curr_tag) transitions from training data
2. Extract all (word, tag) pairs from training data
3. Count occurrences and normalize to probabilities
4. Create dense/sparse matrices for efficient lookup

#### **2b) Out-of-Vocabulary (OOV) Handling** ⚠️

Implement **one or more** of the following:

| Strategy | Description | Trade-off |
|----------|-------------|-----------|
| **Laplace Smoothing** | Add-one smoothing to all counts | Uniform probability increase |
| **Unknown Token** | Replace rare words with `<UNK>` | Information loss for rare words |
| **Back-off Model** | Use character n-grams or POS tags | Computationally expensive |
| **Probability Redistribution** | Reserve portion of probability mass | Complex calibration |

#### **2c) Viterbi Algorithm Implementation** 🎯

**Algorithm Overview:**
```
Function Viterbi(observation_sequence):
    Initialize DP table [# tags × sequence length]
    
    For each time step t:
        For each possible current tag:
            Find max probability over all previous tags
            Store backpointer for path recovery
    
    Backtrack from final state to recover optimal path
    
    Return: Sequence of most likely tags
```

**Key Requirements:**
- ❌ **Avoid:** High-level wrappers (nltk.tag.hmm, sklearn.hmm)
- ✅ **Encouraged:** Vectorized NumPy implementations
- ⚡ **Optimization:** Use log-probabilities to avoid underflow

**Pseudocode:**
```python
# Viterbi decoding
viterbi[0, :] = initial_probs * B[:, word[0]]  # Initialize
backpointer = []

for t in range(1, len(words)):
    for tag_i in all_tags:
        transitions = viterbi[t-1, :] * A[:, tag_i]
        viterbi[t, tag_i] = max(transitions) * B[tag_i, word[t]]
        backpointer[t, tag_i] = argmax(transitions)

# Backtrack to recover optimal path
optimal_tags = reconstruct_path(backpointer)
```

**📌 Deliverable:**
- Heatmap/DataFrame snapshot of transition matrix (5×5 sample)
- Implementation code with detailed comments
- Performance analysis (time/space complexity)

---

### **Part 3: Inference & Evaluation** `[4 Marks]`

#### **3a) Sequence Inference** (Predict POS tags for 4 sentences)

Run your trained HMM + Viterbi decoder on:
- ✅ **2 sentences from TRAINING dataset**
- ✅ **2 sentences from TEST dataset**

**Output Format:**
```
Sentence 1: "The quick brown fox jumps over the lazy dog"
Predicted:   [DET ADJ  ADJ   NN  VBZ  ADP   DET ADJ  NN ]
Expected:    [DET ADJ  ADJ   NN  VBZ  IN    DET ADJ  NN ]
Confidence:  [0.95 0.92 0.88 0.99 0.87 0.73 0.98 0.91 0.96]
```

#### **3b) Error Analysis** (Explain 3 misclassifications)

Identify and document:
- The problematic sentence
- Incorrect prediction vs. expected tag
- Root cause analysis:
  - 🔤 **Lexical Ambiguity:** Word has multiple valid POS tags
  - 📊 **Sparse Transitions:** Rare tag bigrams in training data
  - 🎯 **Contextual Confusion:** Model lacks long-range dependencies
  - 📉 **Low Emission Probability:** Word-tag pair underrepresented

**Example Analysis:**
```
Misclassification #1:
Sentence: "They can fish for dinner"
Issue: "can" predicted as NN (noun), actual: MD (modal)
Reason: "can" as NN (container) more frequent in training data
        Model lacks context to distinguish "can" (modal) from "can" (noun)
```

#### **3c) Complete Test Set Evaluation** (20% test set)

| Metric | Method |
|--------|--------|
| **Token Accuracy** | (Correct Predictions) / (Total Tokens) |
| **Confusion Matrix** | For top 5 most frequent POS tags |
| **Precision/Recall** | Per-tag performance breakdown |
| **Macro F1 Score** | Unweighted average across tags |

**📌 Deliverable:**
- Overall accuracy score (expected: 80-90% for HMM)
- Confusion matrix heatmap (5×5 grid of top tags)
- Per-tag precision/recall table
- Comparison with random baseline (25-30% for 5 tags)

---

### **Part 4: Baseline Comparison** `[1 Mark]`

#### **Baseline Model Options:**

Choose one and evaluate:
- 🔗 **SpaCy** - Production-grade, transformer-based
- 📚 **Stanza** - Stanford NLP, LSTM-based
- 🎓 **NLTK PerceptronTagger** - Structured prediction
- 🤗 **Hugging Face** - Pre-trained transformers (BERT, etc.)

#### **Comparative Analysis:**

Answer these questions:
1. **What is the baseline accuracy?** (typically 95-97% for modern models)
2. **Why does it outperform your HMM?** 
   - Higher model capacity (more parameters)
   - Richer feature representations
   - Pre-training on massive corpora
3. **What architectural improvements account for the gap?**
   - RNNs/LSTMs for long-range dependencies
   - Attention mechanisms for context
   - Transformer-based contextual embeddings
4. **Insights:** What did you learn about HMM limitations?

**📌 Deliverable:**
- Baseline accuracy + confusion matrix
- Side-by-side comparison table
- Written analysis (2-3 paragraphs)

---

## 📊 Evaluation Metrics Deep Dive

### **Token-Level Accuracy**
```
Accuracy = (Correct Tokens) / (Total Tokens)
```

### **Confusion Matrix Interpretation**
```
       Predicted
         NN  VB  JJ  DET  IN
Actual
NN     [120  5   2   1    2]
VB     [3    95  1   0    1]
JJ     [2    0   87  0    0]
DET    [0    0   0   110  0]
IN     [5    1   0   2    92]

Row: True label
Col: Predicted label
Diagonal: Correct predictions
Off-diagonal: Errors
```

### **Per-Tag Metrics**
```
Precision (NN) = TP(NN) / (TP(NN) + FP(NN))  [How many predicted NNs were correct?]
Recall (NN)    = TP(NN) / (TP(NN) + FN(NN))  [How many actual NNs did we find?]
F1 (NN)        = 2 × (Precision × Recall) / (Precision + Recall)
```

---

## 📦 Deliverables Checklist

### **Jupyter Notebook Structure**

- [ ] **Header Section**
  - Team member names and BITS IDs prominently displayed
  - Course and assignment title
  - Date of submission

- [ ] **Code Execution Quality**
  - Kernel → Restart & Run All must pass without errors
  - No red error messages in output
  - All cells execute sequentially

- [ ] **Part 1: EDA (2 Marks)**
  - [ ] Data loading code and dataset statistics
  - [ ] POS tag distribution bar chart (visualization)
  - [ ] Vocabulary analysis (top 20 words)
  - [ ] Sentence length statistics
  - [ ] Train-test split confirmation

- [ ] **Part 2: Model (4 Marks)**
  - [ ] Transition matrix computation code
  - [ ] Emission matrix computation code
  - [ ] Heatmap visualization of transition matrix sample
  - [ ] OOV handling implementation explanation
  - [ ] Viterbi algorithm implementation (with comments)
  - [ ] Time/space complexity analysis

- [ ] **Part 3: Evaluation (4 Marks)**
  - [ ] POS predictions for 4 sentences (2 train, 2 test)
  - [ ] 3 misclassification examples with analysis
  - [ ] Full test set evaluation results
  - [ ] Final accuracy score
  - [ ] Confusion matrix heatmap for top 5 tags
  - [ ] Per-tag precision/recall metrics

- [ ] **Part 4: Baseline (1 Mark)**
  - [ ] Baseline model implementation/loading
  - [ ] Baseline predictions on same test set
  - [ ] Comparative accuracy analysis
  - [ ] Discussion of why baseline outperforms HMM

- [ ] **Bonus: BITS Lab (1 Mark)**
  - [ ] Screenshots showing BITS CSIS Labs environment
  - [ ] Evidence of running the notebook in lab environment
  - [ ] Lab session ID or timestamp

### **Visualization Requirements** (All must be present and clear)

```
Required Output Cells:
✓ POS Tag Distribution Bar Chart
✓ Transition Matrix Heatmap (5×5 sample)
✓ Confusion Matrix Heatmap (5×5)
✓ Example predictions with tags
✓ Error case analysis with explanations
✓ Baseline comparison table
✓ BITS Lab screenshots
```

---

## 🛠️ Tech Stack & Libraries

| **Component** | **Technology** | **Purpose** |
|---------------|---------------|-----------|
| **Language** | Python 3.8+ | Core implementation |
| **Data Processing** | Pandas, NumPy | Matrix operations, data handling |
| **NLP Corpus** | NLTK | Brown corpus access |
| **Visualization** | Matplotlib, Seaborn | Charts and heatmaps |
| **Baseline Models** | SpaCy / Stanza | Pre-trained comparison |
| **Environment** | Jupyter Notebook | Development & documentation |
| **Computing** | BITS CSIS Labs | Recommended (bonus mark) |

**Installation:**
```bash
pip install nltk numpy pandas matplotlib seaborn spacy
python -m spacy download en_core_web_sm
```

---

## 📖 Key Concepts Reference

### **Hidden Markov Models (HMM)**

**Components:**
- **State Space (S):** Set of all POS tags {NN, VB, JJ, DET, IN, ...}
- **Observation Space (O):** Entire vocabulary of words
- **Transition Probability:** P(tag_t | tag_{t-1}) - probability of current tag given previous
- **Emission Probability:** P(word_t | tag_t) - probability of word given its tag
- **Initial Probability:** P(tag_1) - probability of starting with a tag

**Assumptions:**
- **Markov Property:** Future depends only on present, not past
- **Bigram Model:** Only previous tag influences current tag
- **Output Independence:** Observations depend only on current state

### **Viterbi Algorithm**

**Why Dynamic Programming?**
```
Naive approach: Evaluate all possible tag sequences
Complexity: O(|tags|^sequence_length) - EXPONENTIAL ❌

Viterbi DP approach:
Complexity: O(|tags|^2 × sequence_length) - POLYNOMIAL ✅
```

**Key Insight:**
```
Instead of recomputing probabilities, store the maximum at each step:
viterbi[t][i] = max_over_j(viterbi[t-1][j] × P(tag_i | tag_j)) × P(word_t | tag_i)
```

### **Evaluation Metrics**

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **Accuracy** | Correct / Total | Overall correctness |
| **Precision** | TP / (TP + FP) | Quality of predictions |
| **Recall** | TP / (TP + FN) | Coverage of actual cases |
| **F1 Score** | 2PR/(P+R) | Balanced metric |

---

## 🔗 Important Notes & Constraints

### ⚠️ **Critical Requirements**

1. **No External HMM Libraries**
   - ❌ Cannot use `nltk.tag.hmm` or `sklearn.HiddenMarkovModel`
   - ❌ Cannot use pre-built Viterbi decoders
   - ✅ Must implement from scratch or use NumPy

2. **OOV Handling is Mandatory**
   - Must implement at least one smoothing/back-off strategy
   - Document your approach clearly

3. **Dataset Specificity**
   - Training: NLTK Brown corpus `news` category only
   - Testing: NLTK Brown corpus `fiction` category only
   - No mixing or different data sources

### 📞 **For Assignment Queries**
- **Course Lead:** Ajay Naik
- **Channel:** Official course communication platform
- **Response Time:** Expected within 48 hours

### 💡 **Bonus Opportunity**
- **+1 Mark:** Submit evidence of running the complete notebook in BITS CSIS Labs
- Capture screenshots showing lab environment and execution

---

## 🚀 Getting Started - Step-by-Step Workflow

```python
# 1. DATA LOADING & EDA (Part 1)
from nltk.corpus import brown
train_sentences = brown.tagged_sents(categories='news')[:3000]  # Sample
test_sentences = brown.tagged_sents(categories='fiction')[:1000]

# 2. PREPROCESSING
# - Tokenize and extract (word, tag) pairs
# - Create vocabulary with OOV handling
# - Build train-test sets

# 3. PROBABILITY MATRICES (Part 2)
# - Compute transition matrix A
# - Compute emission matrix B
# - Handle smoothing for unseen events

# 4. VITERBI IMPLEMENTATION (Part 2)
# def viterbi(words, A, B, tags, initial_probs):
#     # Implement DP algorithm
#     return optimal_tags

# 5. INFERENCE (Part 3)
# - Run Viterbi on 4 test sentences
# - Collect predictions

# 6. EVALUATION (Part 3)
# - Compute accuracy on full test set
# - Generate confusion matrix
# - Identify error cases

# 7. BASELINE COMPARISON (Part 4)
# - Load pre-trained model (SpaCy/Stanza)
# - Run on same test set
# - Compare and analyze

# 8. VISUALIZATION & DOCUMENTATION
# - Create all required charts
# - Write analysis sections
# - Clean notebook structure
```

---

## 📊 Expected Outcomes & Performance Benchmarks

### **HMM Baseline Performance**
- **Token Accuracy:** 80-90% (depends on OOV handling quality)
- **Top 5 Tags:** NN, VB, JJ, DET, IN (cover ~70% of tokens)
- **Error Types:** 
  - Lexical ambiguity (can/could/might): 20-30% of errors
  - Rare transitions: 15-20% of errors
  - OOV words: 20-25% of errors

### **vs. Modern Baselines**
- **SpaCy/Stanza:** 95-97% accuracy
- **BERT-based:** 97-98% accuracy
- **Gap Analysis:** 5-17% margin due to architectural differences

### **Key Learnings**
✨ You'll understand:
- Why contextual information matters in NLP
- Limitations of independence assumptions
- Trade-offs between model complexity and interpretability
- Importance of data quality and coverage

---

## 📚 References & Further Reading

### **Foundational Papers**
1. Rabiner, L. R. (1989). "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition"
2. Viterbi, A. (1967). "Error bounds for convolutional codes and an asymptotically optimal decoding algorithm"
3. Church, K. W. (1988). "A stochastic parts program and noun phrase parser for unrestricted text"

### **Textbooks**
- Jurafsky, D., & Martin, J. H. (2021). "Speech and Language Processing" (3rd ed.)
- Manning, C. D., & Schütze, H. (1999). "Foundations of Statistical NLP"

### **Implementations Reference**
- NLTK Documentation: https://www.nltk.org/
- SpaCy POS Tagging: https://spacy.io/usage/linguistic-features#pos-tagging
- Stanza: https://stanfordnlp.github.io/stanza/

---

**✨ Good luck! Make your implementation elegant, well-documented, and insightful! ✨**

*For the best experience, run this notebook in BITS CSIS Labs. Happy learning! 🎓*
