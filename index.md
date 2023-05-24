---
title: 
feature_text: |
  # Backpack Models
  Neural sequence models with an interface for interpretability
feature_image: "assets/banner.png"
excerpt: "Alembic is a starting point for [Jekyll](https://jekyllrb.com/) projects. Rather than starting from scratch, this boilerplate is designed to get the ball rolling immediately. Install it, configure it, tweak it, push it."
---

A **Backpack** is a drop-in replacement for a Transformer that provides new tools for **interpretability-through-control** while still enabling strong language models.
Backpacks decompose the predictive meaning of words into components non-contextually, and aggregate them by a weighted sum, allowing for precise, predictable interventions.

<img src="assets/backpack-process.gif" >

A Backpack model is a neural network that operates on sequences of symbols. It (1) learns a set of _sense vectors_ of meaning for each symbol, and (2) in context, weights and sums each sense vector of that context to represent each word of the sequence.

<img src="assets/senses.png" >

When training Backpack Language Models, we find that sense vectors specialize to represent fine-grained aspects of predictive utility for each word.
Intuitively, sense vectors **non-contextually specify the span of ways in which the word might be useful in context,** and the context decides what member of that span to take.

Because each sense vector represents a non-contextual log-probability vector, we say they have a **transparent** semantics.
The transparent connection between symbol meaning and model prediction enables new directions in interpretability and control.
This simplicity is enabled by the use of existing, opaque neural architectures (like the Transformer) _only in the role of generating weights for the sum_.
For one example of control in our ACL paper, we identify a source of gender bias in stereotypically gendered career nouns as being partially derived from a single sense vector, and "turn down" the weights on that sense to reduce bias:

<img src="assets/gender.png" >

The name "Backpack" is inspired by the fact that a backpack is like a bag---but more orderly. Like a bag-of-words, a Backpack representation is a sum of non-contextual senses, but a Backpack is more orderly, because the weights in this sum depend on the ordered sequence.

<!--{% include button.html text="Fork it" icon="github" link="https://github.com/daviddarnes/alembic" color="#0366d6" %}   {% include button.html text="Demo" link="#" %}  {% include button.html text="ACL Paper" link="#" %}-->

#### Demo a Backpack language model

- Visualize sense vectors [here](https://huggingface.co/spaces/lora-x/Backpack).
- Generate from and control a Backpack language model [here](#).

#### Train or finetune your own Backpacks
- Download and use our up-to-170M parameter models on [HuggingFace](#).
- To get the gist, use our simple implementation on Andrej Karpathy's nanoGPT; [nanoBackpackGPT](#).
- Like JAX and TPUs? We have an implementation in JAX via Stanford's [Levanter library](#).
- Want to reproduce our ACL paper? Our original implementation is in FlashAttention, [here](#).

#### Citation

We introduced Backpack Models, and trained and evaluated Backpack Language Models in our ACL paper, _Backpack Language Models_.
If you find the ideas or models here useful, please cite:


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

#### Acknowledgements

This work was supported by the following organizations.
All errors are our own, and all opinions do not necessarily reflect the organizations.

<img src="assets/stanfordnlp-logo.jpg" width="100px" >
<img src="assets/crfm-rgb.png" width="120px" >
<img src="assets/sail-logo.png" width="120px" >

Demos were developed by [Lora Xie](#). Considerable feedback and advice on paper drafts given by the Steven Cao, Xiang Lisa Li, and the rest of the Stanford NLP Group community.
John Hewitt was supported by the National Science Foundation Graduate Research Fellowship under grand number DGE-1656518.
