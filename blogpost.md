---
title: 
feature_text: |
  ## An Overview of Backpack Language Models
  Motivation and annotated code
feature_image: "../assets/banner.png"
excerpt: "Alembic is a starting point for [Jekyll](https://jekyllrb.com/) projects. Rather than starting from scratch, this boilerplate is designed to get the ball rolling immediately. Install it, configure it, tweak it, push it."
---

In this blog post, we detail the **Backpack** architecture, a drop-in replacement for the Transformer that provides new tools for **interpretability-through-control** while still enabling strong language models.
- We'll discuss the design motivation for Backpacks and what distinguishes them from Transformers, RNNs, and most other sequence architectures.
- We'll demonstrate examples of how we've used the Backpack structure to implement precise interventions.
- Then, we'll walk through the code, discussing details and showing how easy it is to implement.

### Monolithic functions and control

Neural sequence modeling usually proceeds by computing non-linear featurizations of each prefix in a single monolithic function, and modeling distributions over sequences through the product of distributions over "next token" given "context".

### The Backpack Language Model

<img src="../assets/backpack-process.gif" >

### Emergent structure in sense vectors

<img src="../assets/senses.png" >

### Examples of Control

<img src="../assets/gender.png" >

## Code Walkthrough

### Sense Network

### Contextualization Network

### The Backpack


## Citation


```
@InProceedings{hewitt2023backpack,
  author =      "Hewitt, John and Thickstun, John and Manning, Christopher D. and Liang, Percy",
  title =       "Backpack Language Models",
  booktitle =   "Proceedings of the Association for Computational Linguistics",
  year =        "2023",
  publisher =   "Association for Computational Linguistics",
  location =    "Toronto, Canada",
}
```
