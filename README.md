# Ai-refresher + related stuff

This repository is a bunch of links with key information to review the basics of Artificial Intelligence and Data Science. Links marked as Cheat sheets do not have any in deep review however present the key ideas and are a good starting point for deciding which topics to review in depth.

[All of AI notes, website, videos and more](https://aman.ai/)

## Basic equations and concepts

### Error

**Precision**
<body>

$$ \Large \frac{tp}{tp+fp} $$

**Recall**

$$\Large \frac{tp}{tp+fn} $$

**F1_score**

$$ \Large 2 \cdot \frac{\cdot precission \cdot recall}{(precision+recall)} $$

$$ \Large \frac{2 \cdot tp}{2\cdot tp+fp+fn}$$

### Linear algebra
- Rank: Number of dimensions spanned by the matrix columns which is the same as the number of lineraly independent columns. This number is the same also for the row vectors.

- Full rank: The rank is the same as the number of columns or rows (the lesser of the two). It can be also called full column rank or full row rank.

### Statistics 
The null hypothesis involves the absence of a difference or the absence of an association. For classification: Predicting a positive is the alternative hypothesis, predicting a negative is the Null hypothesis.

**Type I error**



Mistaken rejection of the null hypothesis when it holds true. To explain how it works in machine learning, lets assume we have a negative sample as input to a classifier and the classifier predicts the input is a positive. By predicting the input is a positive, the classifier is rejecting th hypothesis that the sample is negative, in other words is rejecting the null hypothesis. Since the hypothesis was true, this is a type I error. In summary a type I error means getting a false positive.

**Type II error**

Mistaken failure to reject the null hypothesis when it is false. Now lets assume a classifier gets as input a positive sample and predicts that this is a negative sample. The predictiong of a negative sample means the classifier is accepting the null hypothesis or not to rejecting it. However, the null hypothesis is false, the input is a positive then this is a type II error. In other words a type II error is a false negative.


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
For a great summary of different DL architectures and sample code check [link](https://github.com/rasbt/deeplearning-models).
For a visualization of the genealogy of DL architectures check [link](https://github.com/hunkim/deep_architecture_genealogy).

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

$$ \Large CE= \sum_{x\in X} p(x) \cdot log(q(x)) $$

Where p(x) is the probability of the label (usually one or zero) and q(x) is the normalized (0,1) output from the neural network.
For the binary classification problem, the output of the network is a sigmoid and so the cross-entropy loss becomes:

$$ \Large CE= -t_i \cdot log(f(s_i) -(1-t_i) \cdot log(1-f(s_i))$$

Where

$$ \Large f(s_i)=\frac{1}{1%2Be^{-s_i}} $$


#### Categorical cross-entropy

The loss using cross entropy for multiple classes and using a softmax output becomes:

$$\Large CE= log(\frac{ e^{s_i} }{\sum_{j\in J} e^{s_j}} ) $$

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

## Signal processing and ML
[Signal processing course in Jupyter](https://dartbrains.org/content/Instructors.html)

[Machine Learning for signal processing CMU](http://www.cs.cmu.edu/~11755/lectures/lectures.html)

[Machine Learning for Signal Processing CMU 2020](https://teamia.io/)


