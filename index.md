---
title: 
feature_text: |
  ## Backpack Models
  Neural sequence models with a transparent control mechanism
feature_image: "assets/banner.png"
excerpt: "Alembic is a starting point for [Jekyll](https://jekyllrb.com/) projects. Rather than starting from scratch, this boilerplate is designed to get the ball rolling immediately. Install it, configure it, tweak it, push it."
---

A **Backpack** is a drop-in replacement for a Transformer that provides new tools for **interpretability-through-control** while still enabling strong language models.

<img src="assets/backpack-process.gif" >

A Backpack model is a neural network that operates on sequences of symbols. It (1) learns a set of _senses_ of meaning for each symbol, and (2) in context, weights and sums each sense of that context to represent each word of the sequence.

When training Backpack Language Models, we find that 

The transparent connection between symbol meaning and model prediction enables new directions in interpretability and control of neural models.
This simplicity is enabled by the use of existing, opaque neural architectures (like the Transformer) _only in the role of generating weights for the sum_.

The name "Backpack" is inspired by the fact that a backpack is like a bag---but more orderly. Like a bag-of-words, a Backpack representation is a sum of non-contextual senses, but a Backpack is more orderly, because the weights in this sum depend on the ordered sequence.

{% include button.html text="Fork it" icon="github" link="https://github.com/daviddarnes/alembic" color="#0366d6" %}   {% include button.html text="Demo" link="#" %}  {% include button.html text="ACL Paper" link="#" %}

## Demo a Backpack language model

- Check out our [visualization of sense vectors](https://huggingface.co/spaces/lora-x/Backpack)
- Generate from and control a Backpack language model [here](#).

## Train your own Backpack
- To get started quickly, use our simple implementation on Andrej Karpathy's nanoGPT; [nanoBackpackGPT](#)
- Like JAX and TPUs? We have an implementation in JAX via Stanford's [Levanter library](#).
- Want to reproduce our ACL paper? Our original implementation is in FlashAttention, [here](#).



## Customising

When using Alembic as a theme means you can take advantage of the file overriding method. This allows you to overwrite any file in this theme with your own custom file, by matching the file name and path. The most common example of this would be if you want to add your own styles or change the core style settings.

To add your own styles copy the [`styles.scss`](https://github.com/daviddarnes/alembic/blob/master/assets/styles.scss) into your own project with the same file path (`assets/styles.scss`). From there you can add your own styles, you can even optionally ignore the theme styles by removing the `@import "alembic";` line.

If you're looking to set your own colours and fonts you can overwrite them by matching the variable names from the [`_settings.scss`](https://github.com/daviddarnes/alembic/blob/master/_sass/_settings.scss) file in your own `styles.scss`, make sure to state them before the `@import "alembic";` line so they take effect. The settings are a mixture of custom variables and settings from [Sassline](https://medium.com/@jakegiltsoff/sassline-v2-0-e424b2881e7e) - follow the link to find out how to configure the typographic settings.
