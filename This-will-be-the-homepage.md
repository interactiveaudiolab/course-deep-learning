<a name="top"></a>

# DEEP LEARNING: Northwestern University CS 396/496 FALL 20

|[**Top**](#top)|  [**Calendar**](#calendar)| [**Links**](#links)| [**Slides**](#slides)|  [**Readings**](#readings)|

#### Loctation
ONLINE ON ZOOM (meeing info available on [Canvas](https://www.it.northwestern.edu/education/login.html))
#### Class Day/Time
 Tuesday/Thursday, 4:20pm - 5:40pm Central Time
#### Office Hours
 Mondays 4:00pm - 6:00pm Central Time
#### Instructor
[Bryan Pardo](http://bryanpardo.com)

## Course Description 
We will study deep learning architectures: perceptrons, multi layer perceptrons, convolutional networks, recurrent neural networks (LSTMs, GRUs), Attention networks and the combination of reinforcement learning with deep learning. Other covered topics include regularization, loss functions and gradient descent. Learning will be in the practical context of implementing networks using these architectures in a modern programming environment: Pytorch. In addition to weekly lab and reading assignments, the final capstone to the course will be a final project where students may apply deep learning to a problem domain of their choice.

## Course textbook
The primary text is the [Deep Learning book](http://www.deeplearningbook.org/). This reading will be supplemented by reading key papers in the field.

## Course Policies 
#### Questions outside of class
Please use [CampusWire](https://campuswire.com) for class-related questions.

#### Grading Policy
You will be graded on a 100 point scale (e.g. 93 to 100 = A, 90-92 = A-, 87-89 = B+, 83-86 = B, 80-82 = B-...and so on). 

Homework and reading assignments are solo assignments and must be original work.  

Final projects are group assignments and all members of a group will share a grade for all parts of the assignment.

#### Submitting assignments
Assignments must be submitted on the due date by the time specified on Canvas. If you are worried you can't finish on time, upload a safety submission an hour early with what you have. I will grade the most recent item submitted before the deadline. Late submissions will not be graded.

#### Class participation for up to 10 points of extra credit.
Students can earn up to 10 points (A full letter grade) of extra credit with class participation. 

*Participation during lecture* There are 20 lectures this term. You will be asked to select 2 lectures for which you will be *on-call*. In your on-call lectures, I will feel free to call on you and will expect that you've done the relevant reading prior to lecture and will be able to engage in meaningful interaction on the lecture topic. Each on-call day will be worth up to 3 points.  

*CampusWire reputation* We will track student [CampusWire](https://campuswire.com) reputation scores. Those in the top 25% earn 4 points, top 50% earn 3 points, top 75% earn 2 points, Bottom 25% earn 1 point.  

No additional extra credit beyond the 10% for class participation will be provided. No requests for extra-extra credit will be considered.

<a name="calendar"></a>
## Course Calendar
[**Back to top**](#top)

| Week|Day and Date| Topic (tentative)                        |Deliverable         | Points|
|----:|------------|------------------------------------------|--------------------|------:|
|1 | Thu Sep 17   | Course basics                            | | |
|2 | Tue Sep 22   | The perceptron                           | Readings             | 9  |
|2 | Thu Sep 24   | Basic Pytorch, Combining perceptrons     | | |
|3 | Tue Sep 29   | Basics of optimization                   | Readings             | 9  |
|3 | Thu Oct 1    | Backpropagation of loss through networks | | |
|4 | Tue Oct 6    | Using TensorBoard and Lightning          | Homework 1           | 10 |
|4 | Thu Oct 8    | Convolutional Filters & Pooling          | | |
|5 | Tue Oct 13   | Convolutional Filters & Pooling          | Readings             | 9 |
|5 | Thu Oct 15   | Regularization, more loss functions      | | |
|6 | Tue Oct 20   | Adversarial Attacks                      | Homework 2           | 10|
|6 | Thu Oct 22   | The final project                        | | |     
|7 | Tue Oct 27   | Recurrent networks, LSTM & GRUs          | | |
|7 | Thu Oct 29   | Recurrent networks, LSTM & GRUs          | Project proposal     | 5 |
|8 | Tue Nov 3    | Example recurrent net: source separation | | |
|8 | Thu Nov 5    | Language models                          | Project Readings     | 9 |
|9 | Tue Nov 10   | Attention networks                       | | |
|9 | Thu Nov 12   | Transformers                             | Detailed Project Plan| 5 |
|10 | Tue Nov 17  | BERT and GPT                             | | |
|10 | Thu Nov 19  | Deep reinforcement learning              | Project Meeting      | 5 |
|11 | Tue Nov 24  | Deep reinforcement learning              | | |
|11 | Thu Nov 26  | THANKSKIVING                             | | |
|12 | Tue Dec 1   | Individual project meetings              | Readings             | 9 |
|12 | Thu Dec 3   | Individual project meetings              | | |
|13 | Tue Dec 8   | no class, finals week                    | | |
|13 | Thu Dec 10  | FINAL PROJECTS DUE                       | Final website & paper| 20 |

<a name="links"></a>
## Links   
[**Back to top**](#top)

### Helpful Programming Packages

[Anaconda](https://www.anaconda.com) is the most popular python distro for machine learning.

[Pytorch](http://pytorch.org/) Facebook's popular deep learning package. My lab uses this.

[Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) makes Pytorch easier to use.

[Tensorboard](https://www.tensorflow.org/tensorboard) is what my lab uses to visualize how experiments are going. 

[Tensorflow](https://www.tensorflow.org/) is Google's most popular python DNN package

[Keras](https://keras.io/) A nice programming API that works with Tensorflow

[JAX](https://github.com/google/jax) Is an alpha package from Gogle that allows differentiation of numpy and also an optimizing compiler for working on tensor processing units

[Trax](https://github.com/google/trax) Is Google Brain's DNN package. It focuses on transformers and is implemented on top of [JAX](https://github.com/google/jax)

[MXNET](https://mxnet.apache.org/versions/1.6/get_started?) is Apache's open source DL package.

### Helpful Books on Deep Learning

[Deep Learning](http://www.deeplearningbook.org/)  is THE book on Deep Learning. One of the authors won the Turing prize due to his work on deep learning.

[Dive Into Deep Learning](http://d2l.ai/index.html) provides example code and instruction for how to write DL models in Pytorch, Tensorflow and MXNet.

### Computing Resources

[Google's Colab](https://colab.research.google.com/notebooks/intro.ipynb) offers free GPU time and a nice environment for running Jupyter notebook-style projects.
For $10 per month, you also get priority access to GPUs and TPUs.

[Amazon's SageMaker](https://aws.amazon.com/sagemaker/pricing/) offers hundres of free hours for newbies.

[The CS Department Wilkinson Lab](http://it.eecs.northwestern.edu/info/2015/11/03/info-labs.html) just got 22 new machines that each have a graphics card suitable for deep learning, and should be remote-accessable and running Linux with all the python packages needed for deep learning.

### Some Datasets

[Kaggle](https://www.kaggle.com/datasets) Has many useful datasets.

[Zenodo](https://zenodo.org) also has many useful datasets.

[The Imagenet Data](http://imagenet.stanford.edu/): An image database organized according to the WordNet hierarchy in which each node of the hierarchy is depicted by hundreds and thousands of images. Very widely-used.

[The CIFAR Datasets](https://www.cs.toronto.edu/~kriz/cifar.html): The CIFAR-10 and CIFAR-100 are labeled subsets of the 80 million tiny images dataset. Very widely-used.

[The LibriSpeech Data Set](http://www.openslr.org/12): A corpus of approximately 1000 hours of 16kHz read English speech. 

[The WikiText Data Set](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/): A collection of over 100 million tokens extracted from the set of verified Good and Featured articles on Wikipedia.



<a name="slides"></a>
## Lecture Slides and Notebooks
[**Back to top**](#top)

### Lectures


[adversarial examples](deeplearning/DL_adversarial_examples.pdf)

[attention networks](deeplearning/DL_attention_networks.pdf)

[convolutional nets](deeplearning/DL_convolutional_nets.pdf)

[deep reinforcement learning](deeplearning/DL_deep_reinforcement_learning.pdf)

[gradient descent and backpropagation of error](deeplearning/DL_gradient_descent.pdf)

[multi-layer perceptrons](deeplearning/DL_multilayer_perceptrons.pdf)

[perceptrons](deeplearning/DL_perceptrons.pdf)

[recurrent nets](deeplearning/DL_recurrent_nets.pdf)

[regularization](deeplearning/DL_regularization.pdf)

[transformers](deeplearning/DL_transformers.pdf)

### Jupyter Notebooks
[Perceptrons and basic Pytorch ](deeplearning/DL_perceptrons.ipynb)

[Lightning and Tensorboard](deeplearning/DL_lightning_and_tensorboard.ipynb)

<a name="readings"></a>
## Course Reading
[**Back to top**](#top)

#### The History
1. [The Organization of Behavior](http://s-f-walker.org.uk/pubsebooks/pdfs/The_Organization_of_Behavior-Donald_O._Hebb.pdf): Hebb's 1949 book that provides a general framework for relating behavior to synaptic organization through the dynamics of neural networks. 

1. [The Perceptron](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.335.3398&rep=rep1&type=pdf): This is the 1st neural networks paper, published in 1958. The algorithm won't be obvious, but the thinking is interesting and the conclusions are worth reading.

1. [The Perceptron: A perceiving and recognizing automoton](https://blogs.umass.edu/brain-wars/files/2016/03/rosenblatt-1957.pdf): This one is an earlier paper by Rosenblatt that is, perhaps, even more historical than the 1958 paper and a bit easer for an engineer to follow, I think.

#### The basics (1st reading topic)

1. [* Chapter 4 of Machine Learning ](http://www.cs.northwestern.edu/~pardo/courses/eecs349/readings/chapter4-ml.pdf): This is Tom Mitchell's book. Historical overview + explanation of backprop of error. It's a good starting point for actually understanding deep nets. **START HERE. IT'S WORTH 2 READINGS**

1. [Chapter 6 of Deep Learning](http://www.deeplearningbook.org/): Modern intro on deep nets. To me, this is harder to follow than Chapter 4 of Machine Learning, though. Certainly, it's longer. 

####  Optimization (2nd reading topic) 

1. [This reading is **NOT** worth points, but...](https://najeebkhan.github.io/blog/VecCal.html)...if you don't know what a gradient, Jacobian or Hessian is, you should read this before you read Chapter 4 of the Deep Learning book.

1. [Chapter 4 of the Deep Learning Book](http://www.deeplearningbook.org/): This covers basics of gradient-based optimization. **Start here**

1. [Chapter 8 of the Deep Learning Book](http://www.deeplearningbook.org/): This covers optimization. **This should come 2nd**

1. [Why Momentum Really Works](http://distill.pub/2017/momentum/): Reading this will help you understand the popular ADAM optimizer better.

1. [On the Difficulties of Training Recurrent Networks](http://proceedings.mlr.press/v28/pascanu13.pdf): A 2013 paper that explains vanishing and exploding gradients

1. [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167). This is the most common approaches to normalization.

1. [AutoClip: Adaptive Gradient Clipping for Source Separation Networks](https://arxiv.org/abs/2007.14469) is a recent paper out of Pardo's lab that helps deal with unruly gradients. There's also [a video](https://www.youtube.com/watch?v=Rc0AN_PzyE0&feature=youtu.be) for this one.


#### Convolutional Networks (3rd reading topic)
1. [Generalization and Network Design Strategies](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.476.479&rep=rep1&type=pdf): The original 1989 paper where LeCun describes Convolutional networks. **Start here.**

1. [Chapter 9 of Deep Learning: Convolutional Networks](http://www.deeplearningbook.org/). 

####  Regularization and overfitting (4th reading topic)
1. [Chapter 7 of the Deep Learning Book](http://www.deeplearningbook.org/): Covers regularization. 

1. [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf): Explains a widely-used regularizer 

1. [Understanding deep learning requires rethinking generalization](https://arxiv.org/abs/1611.03530): Thinks about the question "why aren't deep nets overfitting even more than they seem to be"?

1. [The Implicit Bias of Gradient Descent on Separable Data](http://www.jmlr.org/papers/volume19/18-188/18-188.pdf) : A study of bias that is actually based on the algorithm, rather than the dataset.


#### Experimental Design
1. [The Extent and Consequences of P-Hacking in Science](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1002106)

#### Visualizing and understanding network representations
1. [Visualizing and Understanding Convolutional Networks](https://arxiv.org/pdf/1311.2901v3.pdf): How do you see what the net is thinking? Here's one way.

1. [Local Interpretable Model-Agnostic Explanations (LIME): An Introduction](https://www.oreilly.com/content/introduction-to-local-interpretable-model-agnostic-explanations-lime/) A technique to explain the predictions of any machine learning classifier.

#### Popular Architectures for Convolutional Networks
If you already understand what convolutional networks are, then here are some populare architectures you can find out about. 

1. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385): The 2016 paper that introduces the popular ResNet architecture that can get 100 layers deep

1. [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556): The 2015 paper introducing the popular VGG architecture

1. [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842):The 2015 paper describing the Inception network architecture.

#### Recurrent Networks
1. [Chapter 10 of Deep Learning](http://www.deeplearningbook.org/): A decent starting point

1. [The Recurrent Neural Networks Tutorial](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/): Another good starting point

1. [* Extensions of recurrent neural network language model](https://ieeexplore.ieee.org/abstract/document/5947611?casa_token=VaRzW-PbtiEAAAAA:BAXcc2Tb4HL-e2TrTSdao50lxoYMaSkGA0o0iZKC8ojYP-wPHfnWCjlOfj6-coIID8PrBqBE): This covers the RNN language model discussed in class.

1. [Backpropagation through time: what it does and how to do it](https://ieeexplore.ieee.org/abstract/document/58337?casa_token=61YezqH4E60AAAAA:Sp19xOIx2R3xt8XnnCy8Cb8vqNt6LLwamZmIr2G6iAAk4PrOYVgqdQyyQKQzXwcgm9bTo6px) 

1. [Long Term Short Term Memory](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory): The original 1997 paper introducing the LSTM

1. [Understanding LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/): A simple (maybe too simple?) walk-through of LSTMs

1. [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](https://arxiv.org/pdf/1412.3555.pdf): Compares a simplified LSTM (the GRU) to the original LSTM and also simple RNN units.


#### Attention networks (read these before looking at Transformers)

1. [Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) ** This is a good starting point on attention models. **

1. [Sequence to Sequence Learning with Neural Networks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf): This is the paper that the link above was trying to explain.

1. [* Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/pdf/1406.1078.pdf): This paper introduces encoder-decoder networks for translation. Attention models were first built on this framework. Covered in class.

1. [* Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf): This paper introduces additive attention to an encoder-decoder. Covered in class.

1. [* Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/pdf/1508.04025.pdf): Introduced multiplicative attention. Covered in class. 

1. [Massive Exploration of Neural Machine Translation Architectures](https://arxiv.org/pdf/1703.03906.pdf): A 2017 paper that settles the questions about which architecture is best for doing translation....except that the Transformer model came out that same year and upended everything. Still, a good overview of the pre-transformer state-of-the-art.

1. [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](http://www.jmlr.org/proceedings/papers/v37/xuc15.pdf): Attention started with text, but is now applied to images. Here's an example.

1. [Listen, Attend and Spell](https://arxiv.org/pdf/1508.01211.pdf): Attention is also applied to speech, as per this example.

1. [A Tutorial in TensorFlow](https://github.com/tensorflow/nmt): Ths walks you how to use Tensorflow 1.X to build a neural  machine translation network with attention.

#### Transformer networks (Don't read until you understand attention models)
1. [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/): A good walkthrough that helps a lot with understanding transformers  ** I'd start with this one to learn about transformers.**

1. [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html): An annotated walk-through of the "Attention is All You Need" paper, complete with detailed python implementation of a transformer.

1. [Attention is All You Need](https://arxiv.org/abs/1706.03762): The paper that introduced transformers, which are a popular and more complicated kind of attention network. 

1. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf): A widely-used language model based on Transformer encoder blocks.

1. [The Illustrated GPT-2](http://jalammar.github.io/illustrated-gpt2/): A good overview of GPT-2 and its relation to Transformer decoder blocks.


#### Adversarial examples 
1. [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) : This paper got the ball rolling by pointing out how to make images that look good but are consistently misclassified by trained deepnets.

1. [Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable Images](https://arxiv.org/pdf/1412.1897.pdf): This paper shows just how screwy you can make an image and still have it misclsasified by a "well trained, highly accurate" image recognition deep net.

#### Reinforcement Learning
1. [Reinforcement Learning: An Introduction, Chapters 3 and 6](http://www.incompleteideas.net/book/RLbook2020.pdf): This gives you the basics of what reinforcement learning (RL) is about.

1. [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602): A key paper that showed how reinforcement learning can be used with deep nets.

1. [Mastering the game of Go with deep neural networks and tree search](http://airesearch.com/wp-content/uploads/2016/01/deepmind-mastering-go.pdf): A famous paper that showed how RL + Deepnets = the best Go player in existence at the time.

1. [A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play](https://science.sciencemag.org/content/362/6419/1140/): This is the AlphaZero paper. AlphaZero is the best go player...and a great chess player.



