# Symbolic-RNN  
Implementation of *Parsing the Language of Expression: Enhancing Symbolic Regression with Domain-Aware Symbolic Priors* (arXiv:2503.09592v1)

This repository aims to develop a symbolic regression framework that:

- extracts domain-aware symbol priors from scientific literature,
- incorporates those priors into a tree-structured RNN controller,
- and enhances symbolic regression via reinforcement learning guided by priors.

The reference paper introduces a hierarchical tree representation of mathematical expressions, systematic extraction of symbol priors across scientific domains (physics / biology / chemistry / engineering), and integration of these priors through KL regularization and constraints into an RNN policy for generating symbolic expressions.


## 📚 Reference

- **Parsing the Language of Expression: Enhancing Symbolic Regression with Domain-Aware Symbolic Priors**, Sikai Huang et al., arXiv:2503.09592v1.

