 <a name="top"></a>

# DEEP LEARNING: Northwestern University CS 396/496 Spring 2024

|[**Top**](#top)|  [**Calendar**](#calendar)| [**Links**](#links)| [**Readings**](#readings)|

#### Class Day/Time
Tuesdays and Thursdays, 9:30am - 10:50am Central Time

#### Loctation
Tech Lecture Room 5


#### Instructors
Professor: [Bryan Pardo](http://bryanpardo.com)

TAs:  Hugo Flores Garcia, Weijan Li

Peer Mentors: Conor Kotwasinski, Cameron Churchwell, Nathan Pruyne, Finn Wintz, Ben Ferreira

#### Office hours
Monday: Weijan Li 3-5pm on [Weijan's zoom link](https://northwestern.zoom.us/j/97477504815), Conor Kotwasinski 5-6pm in Mudd 3532

Tuesday:  Hugo Flores Garcia 1-2pm in Mudd 3532, Cameron Churchwell 1-2pm on  [Cameron's zoom link](https://northwestern.zoom.us/j/3883785036?pwd=S0pxYWgzUGZ2d1p4ZEZGMnl0SG80dz09), Bryan Pardo 3-5pm in Mudd 3115

Wednesday:  Cameron Churchwell 9-10am in Mudd 3532, Ben Ferreira 1-3pm on [Ben's zoom link](https://northwestern.zoom.us/j/95885283343), Conor Kotwasinsky 3 - 4pm in Mudd 3108, Finn Wintz 4-5pm on [Finn's zoom link](https://northwestern.zoom.us/j/2092202714)

Thursday: Finn Wintz 4-5pm on [Finn's zoom link](https://northwestern.zoom.us/j/2092202714),  Nathan Pruyne 6pm - 8pm in Mudd 3532, Hugo Flores Garcia 1-2pm in Mudd 3532



## Course Description 
This is a first course in Deep Learning. We will study deep learning architectures: perceptrons, multi-layer perceptrons, convolutional networks, recurrent neural networks (LSTMs, GRUs), attention networks, transformers, autoencoders, and the combination of reinforcement learning with deep learning. Other covered topics include regularization, loss functions and gradient descent. 

Learning will be in the practical context of implementing networks using these architectures in a modern programming environment: Pytorch. Homework consists of a mixture of programming assignments, review of research papers, running experiments with deep learning architectures, and theoretical questions about deep learning.

Students completing this course should be able to reason about deep network architectures, build a deep network from scratch in Python, modify existing deep networks, train networks, and evaluate their performance. Students completing the course should also be able to understand current research in deep networks. 

## Course Prerequisites
This course presumes prior knowledge of machine learning equivalent to having taken CS 349 Machine Learning.  

## Course textbook
The primary text is the [Deep Learning book](http://www.deeplearningbook.org/). This reading will be supplemented by reading key papers in the field.

## Course Policies 
#### Questions outside of class
Please use [CampusWire](https://campuswire.com) for class-related questions.

#### Submitting assignments
Assignments must be submitted on the due date by the time specified on Canvas. If you are worried you can't finish on time, upload a safety submission an hour early with what you have. I will grade the most recent item submitted before the deadline. Late submissions will not be graded.

#### Grading Policy
You will be graded on a 100 point scale (e.g. 93 to 100 = A, 90-92 = A-, 87-89 = B+, 83-86 = B, 80-82 = B-...and so on). 

Homework and reading assignments are solo assignments and must be original work.  

#### Extra Credit
You can earn up to 8 points of extra credit in the final reading example


<a name="calendar"></a>
## Course Calendar
[**Back to top**](#top)

| Week|Day and Date| Topic (tentative)                        |Due today           | Points|
|----:|------------|------------------------------------------|--------------------|------:|
|1 | Tue Mar 26    | No class: Northwestern runs Monday classes on Tuesday                                  | | |
|1 | Thu Mar 28    | [Perceptrons](slides/DL_perceptrons.pdf)                                               | | |
|1 | -             | [Notebook 1: perceptrons](notebooks/notebook_1_perceptron.ipynb)                       | | |
|2 | Tue Apr 02    | [Gradient descent](slides/DL_gradient_descent.pdf)                                     | Reading 1 | 8 | 
|2 | Thu Apr 04    | [Backpropagation of error](slides/DL_multilayer_perceptrons.pdf)                       | | |
|2 | -             | [Notebook 2: MLP in Pytorch](notebooks/notebook_2_nn.ipynb)                            | | |
|3 | Tue Apr 9     | [Multi-layer perceptrons](slides/DL_multilayer_perceptrons.pdf)                        | Homework 1 | 15 |
|3 | Thu Apr 11    | [Convolutional nets](slides/DL_convolutional_nets.pdf)                                 | | |
|3 | -             | [Notebook 3: Image Classification](notebooks/notebook_3_image_classification.ipynb)    | | |
|4 | Tue Apr 16    | [regularization](slides/DL_regularization.pdf)                                         | Reading 2 | 8 |
|4 | Thu Apr 18    | [Data augmentation & generalization](slides/DL_regularization.pdf)                     | | |
|4 | -             | [Notebook 4: CNNs & Logging](notebooks/notebook_4_augmentation_logging.ipynb)          | | |
|5 | Tue Apr 23    | [Visual adversarial examples](slides/DL_adversarial_examples.pdf)                      | | |
|5 | Thu Apr 25    | [Auditory adversarial examples](slides/DL_audio_adversarial.pdf)                       | Homework 2| 15 |
|5 | -             | [Notebook 5: adversarial examples](notebooks/notebook_5_adversarial_examples.ipynb)    | | |
|6 | Tue Apr 30    | [Generative adversarial networks (GANS)](slides/DL_GANs.pdf)                           | | |
|6 | Thu May 02    | [More GANS](slides/DL_GANs.pdf)                                                        | Reading 3 | 8 |
|6 | -             | [Notebook 6: GANs](notebooks/notebook_6_gan.ipynb)                                     | | |
|7 | Tue May 07    | [Unsupervised methods](slides/DL_unsupervised_methods.pdf)                             | | |
|7 | Thu May 09    | [recurrent nets](slides/DL_recurrent_nets.pdf)                                         | Homework 3 |15 |
|7 | -             | [Notebook 7: autoencoders](notebooks/notebook_7_autoencoder.ipynb)                     | | |
|8 | Tue May 14    | [LSTMs](slides/DL_recurrent_nets.pdf)                                                  | | |
|8 | Thu May 16    | [Deep RL](slides/DL_deep_reinforcement_learning.pdf)                                   | Reading 4 | 8 |
|8 | -             | [Notebook 8: RNNs](notebooks/notebook_8_rnn.ipynb)                                     | | |
|9 | Tue May 21    | [Reinforcement learning (RL)](slides/DL_deep_reinforcement_learning.pdf)               | | |
|9 | Thu May 23    | [Pong with Reinforcement learning (RL)](slides/GM_deep_RL_2_policy_gradients.pdf)      | Reading 5 | 8 |
|9 | -             | [Attention networks](slides/DL_attention_networks.pdf)                                 | | |
|10| Tue May 28    | [Transformers](slides/DL_transformers.pdf)                                             | | |
|10| Thu May 30    | Current research in DL                                                                 | Homework 4 | 15 | 
|10| -             |                                                                                        | | |
|11| Tue Jun 04    | No final exam, just extra credit reading                                               | Extra Credit Reading 6 | 8 |




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

[Google's Colab](https://colab.research.google.com/notebooks/intro.ipynb) offers free GPU time and a nice environment for running Jupyter notebook-style projects.
For $10 per month, you also get priority access to GPUs and TPUs.

[Amazon's SageMaker](https://aws.amazon.com/sagemaker/pricing/) offers hundres of free hours for newbies.

[The CS Department Wilkinson Lab](http://it.eecs.northwestern.edu/info/2015/11/03/info-labs.html) just got 22 new machines that each have a graphics card suitable for deep learning, and should be remote-accessable and running Linux with all the python packages needed for deep learning.


<a name="readings"></a>
## Course Reading
[**Back to top**](#top)

#### The History
1. [The Organization of Behavior](https://pure.mpg.de/pubman/item/item_2346268_3/component/file_2346267/Hebb_1949_The_Organization_of_Behavior.pdf): Hebb's 1949 book that provides a general framework for relating behavior to synaptic organization through the dynamics of neural networks. 

1. [The Perceptron](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.335.3398&rep=rep1&type=pdf): This is the 1st neural networks paper, published in 1958. The algorithm won't be obvious, but the thinking is interesting and the conclusions are worth reading.

1. [The Perceptron: A perceiving and recognizing automoton](https://bpb-us-e2.wpmucdn.com/websites.umass.edu/dist/a/27637/files/2016/03/rosenblatt-1957.pdf): This one is an earlier paper by Rosenblatt that is, perhaps, even more historical than the 1958 paper and a bit easer for an engineer to follow, I think.

#### The basics (1st reading topic)

1. [* Chapter 4 of Machine Learning ](readings/chapter4-ml.pdf): This is Tom Mitchell's book. Historical overview + explanation of backprop of error. It's a good starting point for actually understanding deep nets. **START HERE. IT'S WORTH 2 READINGS. WHAT THAT MEANS IS...GIVE ME 2 PAGES OF REACTIONS FOR THIS READING AND GET CREDIT FOR 2 READINGS**

1. [Chapter 6 of Deep Learning](http://www.deeplearningbook.org/): Modern intro on deep nets. To me, this is harder to follow than Chapter 4 of Machine Learning, though. Certainly, it's longer. 

####  Optimization (2nd reading topic) 

1. [This reading is **NOT** worth points, but...](https://najeebkhan.github.io/blog/VecCal.html)...if you don't know what a gradient, Jacobian or Hessian is, you should read this before you read Chapter 4 of the Deep Learning book.

1. [Chapter 4 of the Deep Learning Book](http://www.deeplearningbook.org/): This covers basics of gradient-based optimization. **Start here for optimization**

1. [Chapter 8 of the Deep Learning Book](http://www.deeplearningbook.org/): This covers optimization. **This should come 2nd in your optimization reading**

1. [Why Momentum Really Works](http://distill.pub/2017/momentum/): Reading this will help you understand the popular ADAM optimizer better.

1. [On the Difficulties of Training Recurrent Networks](http://proceedings.mlr.press/v28/pascanu13.pdf): A 2013 paper that explains vanishing and exploding gradients

1. [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167). This is the most common approaches to normalization.

1. [AutoClip: Adaptive Gradient Clipping for Source Separation Networks](https://arxiv.org/abs/2007.14469) is a recent paper out of Pardo's lab that helps deal with unruly gradients. There's also [a video](https://www.youtube.com/watch?v=Rc0AN_PzyE0&feature=youtu.be) for this one.


#### Convolutional Networks (3rd reading topic)
1. [Generalization and Network Design Strategies](http://yann.lecun.com/exdb/publis/pdf/lecun-89.pdf): The original 1989 paper where LeCun describes Convolutional networks. **Start here.**

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


#### Adversarial examples 
1. [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) : This paper got the ball rolling by pointing out how to make images that look good but are consistently misclassified by trained deepnets.

1. [Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable Images](https://arxiv.org/pdf/1412.1897.pdf): This paper shows just how screwy you can make an image and still have it misclsasified by a "well trained, highly accurate" image recognition deep net.

1. [Effective and Inconspicuous Over-the-air Adversarial Examples with Adaptive Filtering](https://interactiveaudiolab.github.io/assets/papers/oreilly_awasthi_vijayaraghavan_pardo_2021.pdf): Cutting edge research from our very own Patrick O.

####  Creating GANs
1. [Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661v1.pdf): The paper that introduced GANs. **If you read only one GAN paper, make it this one.** 

1. [2016 Tutorial on Generative Adversarial Networks](https://arxiv.org/pdf/1701.00160.pdf) by one of the creators of the GAN. This one's long, but good.

1. [DCGAN: Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434): This is an end-to-end model. Many papers build on this. **The homework uses the discriminator approach from this paper**

1. [Generative Adversarial Text to Image Synthesis](http://proceedings.mlr.press/v48/reed16.pdf) This paper describes generating images conditioned on text descriptions. Pretty interesting...

#### Recurrent Networks
1. [Chapter 10 of Deep Learning](http://www.deeplearningbook.org/): A decent starting point

1. [The Recurrent Neural Networks Tutorial](readings/RNN-tutorial-WildML.pdf): This is a 4-part tutorial that starts with an overview and then gets deep into coding up an RNN using Theano (not PyTorch) and has links to GitHub repositories with all the examples. If you just read this for the points, read Part 1. But go deep, if you're interested, and read all the parts. **NOTE** the links to the code repositories work. Many of the other hyperlinks don't.

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

1. [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044): Attention started with text, but is now applied to images. Here's an example.

1. [Listen, Attend and Spell](https://arxiv.org/pdf/1508.01211.pdf): Attention is also applied to speech, as per this example.

1. [A Tutorial in TensorFlow](https://github.com/tensorflow/nmt): Ths walks through how to use Tensorflow 1.X to build a neural  machine translation network with attention.

#### Transformer networks (Don't read until you understand attention models)
1. [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/): A good walkthrough that helps a lot with understanding transformers  ** I'd start with this one to learn about transformers.**

1. [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html): An annotated walk-through of the "Attention is All You Need" paper, complete with detailed python implementation of a transformer.

1. [Attention is All You Need](https://arxiv.org/abs/1706.03762): The paper that introduced transformers, which are a popular and more complicated kind of attention network. 

1. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf): A widely-used language model based on Transformer encoder blocks.

1. [The Illustrated GPT-2](http://jalammar.github.io/illustrated-gpt2/): A good overview of GPT-2 and its relation to Transformer decoder blocks.


#### Reinforcement Learning
1. [Reinforcement Learning: An Introduction, Chapters 3 and 6](http://www.incompleteideas.net/book/RLbook2020.pdf): This gives you the basics of what reinforcement learning (RL) is about.

1. [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602): A key paper that showed how reinforcement learning can be used with deep nets. This is discussed in class.

1. [Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/): This is the blog we base part of Homework 4 on. Reading this is a very efficient use of your time, since you can count it as a reading AND you have to read it for the homework, anyhow.

1. [Mastering the game of Go with deep neural networks and tree search](http://airesearch.com/wp-content/uploads/2016/01/deepmind-mastering-go.pdf): A famous paper that showed how RL + Deepnets = the best Go player in existence at the time.

1. [A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play](https://science.sciencemag.org/content/362/6419/1140/): This is the AlphaZero paper. AlphaZero is the best go player...and a great chess player.


 |[**Top**](#top)|  [**Calendar**](#calendar)| [**Links**](#links)| [**Readings**](#readings)|

