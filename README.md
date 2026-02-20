# Symbolic-RNN  
Implementation of *Parsing the Language of Expression: Enhancing Symbolic Regression with Domain-Aware Symbolic Priors* (arXiv:2503.09592v1)

This repository aims to develop a symbolic regression framework that:

- extracts domain-aware symbol priors from scientific literature,
- incorporates those priors into a tree-structured RNN controller,
- and enhances symbolic regression via reinforcement learning guided by priors.

The reference paper introduces a hierarchical tree representation of mathematical expressions, systematic extraction of symbol priors across scientific domains (physics / biology / chemistry / engineering), and integration of these priors through KL regularization and constraints into an RNN policy for generating symbolic expressions.

---

## 📌 Current Status

- 🚧 **Work in progress** — Core ML modules under development  
- 🐍 **Scripts in progress** to extract symbolic priors from arXiv math expressions  
- 🛠 No CI / workflows configured yet

---


## 📈 Roadmap

### 🟢 Short Term — Q1 2026
**Foundations + Prior Extraction**
- Finalize scripts to scrape and parse arXiv papers for symbolic expressions
- Build tree representation utilities for expression structures✅
- Produce normalized symbol frequency distributions (horizontal & vertical priors)
- Prototype basic data pipelines for prior corpora

### 🟡 Medium Term — Q2 2026
**ML Framework & Integration**
- Implement tree-structured RNN controller modules
- Integrate priors as soft (KL) & hard constraints into generation policy
- Start training pipelines (reinforcement learning + policy gradients)
- Establish basic evaluation benchmarks vs random / baseline symbolic regression

### 🔵 Long Term —   Q2–Q3 2026
**Optimization, Tools & Workflows**
- Optimize training, scheduling of KL influence & symbolic block incorporation
- Add CI / GitHub Actions to automate tests & experiments
- Expand operator dictionary with domain-specific expression blocks
- Publish pretrained models and evaluation reports

---

## 🧠 Goals

- Enable domain-aware symbolic regression that converges faster and discovers more interpretable models
- Bridge symbol prior extraction with neural symbolic methods
- Provide reusable corpora of scientific expression statistics

---

## 📚 Reference

- **Parsing the Language of Expression: Enhancing Symbolic Regression with Domain-Aware Symbolic Priors**, Sikai Huang et al., arXiv:2503.09592v1. :contentReference[oaicite:3]{index=3}

