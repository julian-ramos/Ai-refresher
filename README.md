# Ai-refresher + related stuff

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
#### ReLU
It has some biological basis. Ignores negative values

#### Leaky ReLu
Similar to ReLU but it uses negative values, although with a smaller weight than positive values.

#### ELU
Same as ReLU for positive values. For negative values it uses an exponential. It has been shown to perform better than other common activation functions and even beat ReLU.

### Loss functions

#### Binary cross-entropy
This loss uses as error measure cross-entropy which is defined as:

<img src="https://render.githubusercontent.com/render/math?math=\Large CE= \sum_{x\in X} p(x) \cdot log(q(x))">

Where p(x) is the probability of the label (usually one or zero) and q(x) is the normalized (0,1) output from the neural network.
For the binary classification problem, the output of the network is a sigmoid and so the cross-entropy loss becomes:

<img src="https://render.githubusercontent.com/render/math?math=\Large CE= -t_i \cdot log(f(s_i) -(1-t_i) \cdot log(1-f(s_i) ">

Where

<img src="https://render.githubusercontent.com/render/math?math=\Large f(s_i)=\frac{1}{1%2Be^{-s_i}} ">


#### Categorical cross-entropy

The loss using cross entropy for multiple classes and using a softmax output becomes:

<img src="https://render.githubusercontent.com/render/math?math=\Large CE= log(\frac{ e^{s_i} }{\sum_{j\in J} e^{s_j}} )">

Where i is the index for the class that is positive. The complete equation uses a vector that one hot encode the class, but since all elements of the vector are zero except i, the above is the resulting equation.


#### Wasserstein Loss
Used for GANs because it helps with mode collapse in vanishing gradients. Similar to BCE but it is not limited to 0 - 1 values and can vary to -infinity to +infinity. To use W loss, the critic or discriminator needs to be 1-Lipschitz continuous. A way to enforce this is by clipping the weight values but that could limit learning. Instead, it is used a regularization term in the loss.

### Types of Layers (from tf.keras.layers)

#### Dense
- *How:* Typical hidden layer where all inputs all connected to all nodes. 
- *Why:* Used in places where combination of features are of interest. 
- *Where:* They are typically used near the output layers after CNN or RNN layers
- *When:* Used in most datasets.


#### CNN
- *How:* (Usually) Multiple convolution filters are applied to an input. The convolution can be in 1D, 2D or 3D. 
- *Why:* A convolution filter has associated weights which can discover
- *Where:* They are typically used near the output layers after CNN or RNN layers
- *When:* Used in most datasets.


#### Dropout
- *How:* Makes output of a layer zero in a random fashion. 
- *Why:* To prevent overfitting by forcing the network to not rely an all features all the time. 
- *Where:* After any dense layer, CNN or RNN. 
- *When:* It helps in all situatiotions but it is most useful with small datasets where chances of overfitting are high.
- 
#### Batch normalization
- *How:* Batch normalization Weights are normalized to zero mean and a defined standard deviation.
- *Why:* Helps with learning rate and decreases dependence on initialization values. It also has a regularization effect.
- *Where:* Commonly used after a CNN layer and before a non-linearity layer.
- *When:* It can be used anytime **except** after a dropout layer. In such case the dropout has an effect on the batch normalization statistics that can introduce noise. Also, it is not adequate when the values are highligh non-gaussian.
- [Spectral normalization (miyato et al)](https://arxiv.org/pdf/1802.05957.pdf) Weights are divided by the spectral norm

#### Pooling

- *How:* Applies an specified function (e.g, max, min, avg) to a patch of the input. A 2x2 patch with a stride of 2, reduces the input to by half, a 3x3 batch with stride of 3 reduces the input by a third. 
- *Why (general):* Reduces complexity.
- *Why (average pooling):* Smooths out the data, in an image it will help ignore sharp features of the image. 
- *Why (Max pooling):* Keeps the max only and helps detect sharp features.
- *Where:* After a convolution layer
- *When:* It helps ignoring or highlighting some components of the input.

## Data Science

Cover tools for data retrieval (SQL), manipulation (python,R)  and visualization (R.python)

[Cheat sheet](https://www.mit.edu/~amidi/teaching/data-science-tools/)

## Signal procession and ML
[Signal processing course in Jupyter](https://dartbrains.org/content/Instructors.html)
[Machine Learning for signal processing CMU](http://www.cs.cmu.edu/~11755/lectures/lectures.html)
[Machine Learning for Signal Processing CMU 2020](https://teamia.io/)


