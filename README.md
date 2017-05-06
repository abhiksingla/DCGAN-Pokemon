# DCGAN-based-Pokemon-generation

Generative adversarial networks are a type of artificial intelligence algorithms used in unsupervised machine learning, implemented by a system of two neural networks competing against each other in a zero-sum game framework. I have used Deep Convolutional Generative Adversarial Networks to generate Pokemon images.

![el](structure_cnn.png)

# Dependencies
* Tensorflow
* Keras
* Pillow
install missing dependencies using [pip](https://pip.pypa.io/en/stable/)

# Usage
Run `python Main.py <Epochs for training> <Batch size for training> <Batch size for generation>` in the terminal. GAN's training is quite complex, training in a GPU environment is highly recommended for better results.
The dataset is available [here](https://veekun.com/dex/downloads). Move the additional dataset to logos folder.

# Credit
This work is learned and is highly based on [Youtube Tutorial](https://www.youtube.com/watch?v=-E2N1kQc8MM) by [Siraj Raval](https://github.com/llSourcell).
