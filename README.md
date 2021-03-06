# Ai-refresher

This repository is a bunch of links with key information to review the basics of Artificial Intelligence and Data Science. Links marked as Cheat sheets do not have any in deep review however present the key ideas and are a good starting point for deciding which topics to review in depth.

[All of AI notes, website, videos and more](https://aman.ai/)

## Basic equations and concepts

### Error

**Precision**

<img src="https://render.githubusercontent.com/render/math?math=\Large \frac{tp}{tp%2Bfp}">

**Recall**

<img src="https://render.githubusercontent.com/render/math?math=\Large \frac{tp}{tp%2Bfn}">

**F1_score**

<img src="https://render.githubusercontent.com/render/math?math=\Large \frac{2\cdot tp}{2\cdot tp%2Bfp%2Bfn}">

### Statistics 
The null hypothesis involves the absence of a difference or the absence of an association. For classification: Positive is the alternative hypothesis, Negative is the Null hypothesis.

**Type I error**

Rejection of a true null hypothesis as the result of a test procedure. False positive.

**Type II error**

Failure to reject a false null hypothesis. False negative.


# Cheat Sheets
## Artificial Intelligence
Covers lots of topics including Ai and ML

[AI Cheat sheet](https://stanford.edu/~shervine/teaching/cs-221/)

## Probability and Statistics

[Cheat sheet](https://stanford.edu/~shervine/teaching/cme-106/)

## Machine Learning
The next includes supervised and unsupervised learning, deep learning, tips and tricks, probability and statistics, and linear algebra and calculus refresher

[ML Cheat sheet](https://stanford.edu/~shervine/teaching/cs-229/)

## Deep Learning

Convolutional and recurrent neural networks explained and some tips and tricks.

[DL Cheat sheet](https://stanford.edu/~shervine/teaching/cs-230/)

### Activation functions
- ReLU
It has some biological basis. Ignores negative values

- Leaky ReLu
Similar to ReLU but it uses negative values, although with a smaller weight than positive values.

- ELU
Same as ReLU for positive values. For negative values it uses an exponential. It has been shown to perform better than other common activation functions and even beat ReLU.

### Loss functions
- Wasserstein Loss
Used for GANs because it helps with mode collapse in vanishing gradients. Similar to BCE but it is not limited to 0 - 1 values and can vary to -infinity to +infinity. To use W loss, the critic or discriminator needs to be 1-Lipschitz continuous. A way to enforce this is by clipping the weight values but that could limit learning. Instead, it is used a regularization term in the loss.

### Batch normalization
- Bath normalization Weights are normalized to zero mean and a defined standard deviation
- [Spectral normalization (miyato et al)](https://arxiv.org/pdf/1802.05957.pdf) Weights are divided by the spectral norm

## Data Science

Cover tools for data retrieval (SQL), manipulation (python,R)  and visualization (R.python)

[Cheat sheet](https://www.mit.edu/~amidi/teaching/data-science-tools/)


