# 📝 Part-of-Speech Tagging using Hidden Markov Models (HMMs)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![NLTK](https://img.shields.io/badge/NLTK-Brown%20Corpus-green.svg)](https://www.nltk.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-orange.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)

> **Course:** Natural Language Processing S2-25_AIMLCZG530  
> **Instructor:** Ajay Naik  
> **Task:** Implement HMM from scratch for POS tagging with Viterbi decoding

---

## 📋 Table of Contents

- [Assignment Overview](#-assignment-overview)
- [Learning Objectives](#-learning-objectives)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Implementation Details](#-implementation-details)
  - [Part 1: Data Preprocessing & EDA](#part-1-data-preprocessing--eda-2-marks)
  - [Part 2: Model Architecture & Parameter Estimation](#part-2-model-architecture--parameter-estimation-4-marks)
  - [Part 3: Inference and Evaluation](#part-3-inference-and-evaluation-4-marks)
  - [Part 4: Baseline Comparison](#part-d-baseline-comparison-1-mark)
- [Evaluation Metrics](#-evaluation-metrics)
- [Deliverables](#-deliverables)
- [Team Members](#-team-members)
- [BITS Lab Usage](#-bits-lab-usage)
- [References](#-references)

---

## 🎯 Assignment Overview

| Aspect | Details |
|--------|---------|
| **Course** | Natural Language Processing S2-25_AIMLCZG530 |
| **Task** | Implement HMM from scratch for POS tagging with Viterbi decoding |
| **Dataset** | NLTK Brown Corpus |
| **Training Category** | `news` |
| **Testing Category** | `fiction` |
| **Bonus Mark** | BITS Lab usage (1 Mark) |

---

## 📚 Learning Objectives

By completing this assignment, students will demonstrate the ability to:

1. ✅ Formulate and implement probabilistic HMMs (Transition & Emission matrices)
2. ✅ Design Viterbi algorithm for dynamic programming-based sequence labeling
3. ✅ Handle unknown words using smoothing/back-off techniques
4. ✅ Evaluate models using standard sequence labeling metrics

---

## 📊 Dataset

### Brown Corpus (NLTK)

```python
from nltk.corpus import brown

# Training: news category
train_sentences = brown.tagged_sents(categories='news')

# Testing: fiction category  
test_sentences = brown.tagged_sents(categories='fiction')
