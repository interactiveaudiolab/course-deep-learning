 <a name="top"></a>

# DEEP LEARNING: Columbia University MSDS Fall 2026

|[**Top**](#top)|  [**Calendar**](#calendar)| [**Links**](#links)| [**Readings**](#readings)|

#### Class Day/Time
To be determined

#### Loctation
To be determined

#### Instructors
Professor: [Bryan Pardo](http://bryanpardo.com)

Teaching Assistants: To be determined

#### Office hours

To be determined

## Course Description 
This is a first course in Deep Learning. We will study deep learning architectures: perceptrons, multi-layer perceptrons, convolutional networks, U-Nets, recurrent neural networks (LSTMs, GRUs), and attention networks (e.g. transformers). Training and loss regimes will be covered, including traditional supervised loss, adversarial loss, autoencoders,  diffusion, and the combination of reinforcement learning with deep learning. Other covered topics include regularization, and first-order gradient descent approaches (e.g. momentum, stochastic gradient descent). 

Learning will be in the practical context of implementing networks using these architectures in a modern programming environment: Pytorch. Homework consists of a mixture of programming assignments, review of research papers, running experiments with deep learning architectures, and theoretical questions about deep learning.

Students completing this course should be able to reason about deep network architectures, build a deep network from scratch in Python, modify existing deep networks, train networks, and evaluate their performance. Students completing the course should also be able to understand current research in deep networks. 

## Course Prerequisites
This course presumes prior knowledge of machine learning equivalent to having taken an introductiory course in ML. It also assumes the ability to code in Python.

## Course textbook
The primary text is the [Deep Learning book](http://www.deeplearningbook.org/). This reading will be supplemented by reading key papers in the field.

## Course Policies 
#### Questions outside of class
Please use [CampusWire](https://campuswire.com) for class-related questions.

#### Submitting assignments
Assignments must be submitted on the due date by the time specified. If you are worried you can't finish on time, upload a safety submission an hour early with what you have. I will grade the most recent item submitted before the deadline. Late submissions will not be graded.

#### Grading Policy
You will be graded on a 100 point scale (e.g. 93 to 100 = A, 90-92 = A-, 87-89 = B+, 83-86 = B, 80-82 = B-...and so on). 

Homework and reading assignments are solo assignments and must be your own original work. Use of large language models for answer generation is not allowed, unless (of course) you are buidling your own language model.

#### Extra Credit
There is no extra credit....BUT...there are 5 homework assignments and we will only use the grades of the top 4 to calculate your final score in the class.

<a name="calendar"></a>
## Course Calendar
[**Back to top**](#top)

| Week|Day and Date| Topic (tentative)                        |Due today           | Points|
|----:|------------|------------------------------------------|--------------------|------:|
|1 | Tue Sep 8     | [Perceptrons](slides/DL_perceptrons.pdf)                                               | | |
|1 | -             | [Notebook 1: perceptrons](notebooks/notebook_1_perceptron.ipynb)                       | | |
|1 | Thu Sep 10    | [Gradient descent](slides/DL_gradient_descent.pdf)                                     | | | 
|2 | Tue Sep 15    | [Backpropagation of error](slides/DL_multilayer_perceptrons.pdf)                       | | |
|2 | -             | [Notebook 2: MLP in Pytorch](notebooks/notebook_2_nn.ipynb)                            | | |
|2 | Thu Sep 17    | [Multi-layer perceptrons](slides/DL_multilayer_perceptrons.pdf)                        | | |
|3 | Tue Sep 22    | [Convolutional nets](slides/DL_convolutional_nets.pdf)                                 | Homework 1 | 15 |
|3 | -             | [Notebook 3: Image Classification](notebooks/notebook_3_image_classification.ipynb)    | | |
|3 | Thu Sep 24    | [regularization](slides/DL_regularization.pdf)                                         | | |
|4 | Tue Sep 29    | [Data augmentation & generalization](slides/DL_regularization.pdf)                     | | |
|4 | -             | [Notebook 4: CNNs & Logging](notebooks/notebook_4_augmentation_logging.ipynb)          | | |
|4 | Thu Oct 1     | [Adversarial examples](slides/DL_adversarial_examples.pdf)                             | | |
|4 | -             | [Notebook 5: adversarial examples](notebooks/notebook_5_adversarial_examples.ipynb)    | | |
|5 | Tue Oct 6     | [Generative adversarial networks (GANS)](slides/DL_GANs.pdf)                           | Homework 2 | 15 |
|5 | -             | [Notebook 6: GANs](notebooks/notebook_6_gan.ipynb)                                     | | |
|5 | Thu Oct 8     | [Unsupervised methods](slides/DL_unsupervised_methods.pdf)                             | | |
|5 | -             | [Notebook 7: autoencoders](notebooks/notebook_7_autoencoder.ipynb)                     | | |
|6 | Tue Oct 13    | [Diffusion models](GM_Score-Diffusion_Models.pdf)                                      | | |
|6 | Tue Oct 15    | [Diffusion models](GM_Score-Diffusion_Models.pdf)                                      | | |
|6 | -             | Diffusion notebook                                                                     | | |
|7 | Tue Oct 20    | Midterm Review                                                                         | | |
|7 | Thu Oct 22    | Midterm                                                                                | Midterm   | 20 |
|8 | Tue Oct 27    | [recurrent nets](slides/DL_recurrent_nets.pdf)                                         | | |
|8 | Thu Oct 29    | [LSTMs](slides/DL_recurrent_nets.pdf)                                                  | | |
|8 | -             | [Notebook 8: RNNs](notebooks/notebook_8_rnn.ipynb)                                     | | |
|9 | Tue Nov 3     | [Attention networks](slides/DL_attention_networks.pdf)                                 | Homework 3  | 15 |
|9 | Thu Nov 5     | [Embeddings](slides/GM_3_embeddings.pdf)                                               | | | 
|10 | Tue Nov 10   | [Transformers](slides/DL_transformers.pdf)                                             | | |
|10 | Thu Nov 12   | [Variational Autoencoders](slides/)                                                    | | |
|11 | Tue Nov 17   | [Tokenization of Image and Audio](slides/GM_multimodal_GenAI_with_transformers.pdf)    | | |
|11 | Thu Nov 19   | [Language Modeling of Image & Audio](slides/GM_multimodal_GenAI_with_transformers.pdf) | Homework 4 | 15 |
|12 | Tue Nov 24   | Copying, Copyright, and Generative Modeling                                            | | |
|12 | Thu Nov 26   | Thanksgiving Holiday: No class                                                         | | |
|13 | Tue Dec 1    | [Deep RL](slides/DL_deep_reinforcement_learning.pdf)                                   | | |
|13 | Thu Dec 3    | [Learning Policy Gradients (RL)](slides/GM_deep_RL_2_policy_gradients.pdf)             | | |
|14 | Tue Dec 8    | [RL from Human Feedback](slides/GM_deep_RL_3_human_feedback.pdf)                       | Homework 5 | 15 |
|14 | Thu Dec 10   | Final exam review                                                                      | | |
|?? | Dec 17-23    | FINAL EXAM                                                                             | Final exam | 20 |


<a name="links"></a>
## Links   
[**Back to top**](#top)

### Helpful Programming Packages

[Anaconda](https://www.anaconda.com) is the most popular python distro for machine learning.

[Pytorch](http://pytorch.org/) Facebook's popular deep learning package. My lab uses this.

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
To be determined

<a name="readings"></a>
## Course Reading
[**Back to top**](#top)


### Book Chapter Readings

1. [Chapter 4 of Machine Learning ](readings/chapter4-ml.pdf): **READ THIS FIRST** This is Tom Mitchell's book. Historical overview + explanation of backprop of error. It's a good starting point for actually understanding deep nets. Read the whole chapter. 

1. [What are Gradients, Jacobians, and Hessians?](https://najeebkhan.github.io/blog/VecCal.html): This isn't a book chapter, but if you don't know what a gradient, Jacobian or Hessian is, you should read this before you read Chapter 4 of the Deep Learning book.

1. [Chapter 4 of the Deep Learning Book](http://www.deeplearningbook.org/): This covers basics of gradient-based optimization. Read through Section 4.3.

1. [Chapter 6 of Deep Learning](http://www.deeplearningbook.org/): This covers the basics from a more modern perspective. To my mind, if you've read Tom Mitchell, it is mostly useful for covering different kinds of activation functions. Read through Section 6.4

1. [Chapter 7 of the Deep Learning Book](http://www.deeplearningbook.org/): Covers regularization. The minimal useful read is sections 7.1 and 7.4...but this assumes you'll read the papers some of the other sections are based on. Those papers are in the additional readings. If you don't read those, then I'd add 7.9, 7.12, 7.13.

1. [Chapter 8 of the Deep Learning Book](http://www.deeplearningbook.org/): This covers optimization. Read through Section 8.5. Beyond that, it is stuff outside the scope of the class.

1. [Chapter 9 of Deep Learning](http://www.deeplearningbook.org/): Convolutional networks. Read 9.1 through 9.4 and 9.10

1. [Understanding LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/): A simple (maybe too simple?) walk-through of LSTMs. Good to read before trying the book chapter on this topic.

1. [Chapter 10 of Deep Learning](http://www.deeplearningbook.org/): RNNs and LSTMS

1. [Reinforcement Learning: An Introduction, Chapters 3 and 6](http://www.incompleteideas.net/book/RLbook2020.pdf): This gives you the basics of what reinforcement learning (RL) is about.

### Additional Readings

1. [Generalization and Network Design Strategies](https://masters.donntu.ru/2012/fknt/umiarov/library/lecun.pdf): The original 1989 paper where LeCun describes Convolutional networks. 

1. [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167v3.pdf)

1. [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) : This paper got the ball rolling by pointing out how to make images that look good but are consistently misclassified by trained deepnets.

1. [Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661v1.pdf): The paper that introduced GANs. 

1. [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf): Explains a widely-used regularizer 

1. [DCGAN: Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434): This is an end-to-end model. Many papers build on this. The homework uses the discriminator approach from this paper

1. [Long Term Short Term Memory](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory): The original 1997 paper introducing the LSTM

1. [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602): A key paper that showed how reinforcement learning can be used with deep nets. This is discussed in class.

1. [Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/): This is the blog we base part of Homework 4 on. 

1. [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/): A good walkthrough that helps a lot with understanding transformers 

1. [Attention is All You Need](https://arxiv.org/abs/1706.03762): The paper that introduced transformers, which are a popular and more complicated kind of attention network. 

1. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf): A widely-used language model based on Transformer encoder blocks.

1. [The Illustrated GPT-2](http://jalammar.github.io/illustrated-gpt2/): Not a paper, but a good overview of GPT-2 and its relation to Transformer decoder blocks.


 |[**Top**](#top)|  [**Calendar**](#calendar)| [**Links**](#links)| [**Readings**](#readings)|

