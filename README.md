# NN-Shaders

I'm exploring machine learning with Pytorch and Tensorflow. I've created my own models that I'm currently integrating with [ReShade](https://reshade.me/), a tool allowing to add inject custom post-processing to video games.

See below for a description of the models I've implemented.

### NNAA

Tensorflow model that addresses [aliasing](https://en.wikipedia.org/wiki/Aliasing) in images. Basically, it is an anti-aliasing neural network that, given an image, produces a similar image but without its aliasing. It has been mainly trained for Final Fantasy XIV.

I've created this shader to try my hand at converting a Tensorflow model to a shader.