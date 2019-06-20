![](https://github.com/jomanovic/deepzipper/blob/master/display/logo.jpg)
<h4 align="center">An intelligent video compression framework utilizing neural nets</a></h4>

<p align="center">
  <a href="#description">Description</a> •
  <a href="#pipeline">Pipeline</a> •
  <a href="#how-does-it-work">How it works</a> •
  <a href="#prerequisites">Required</a> •
  <a href="#results">Results</a> •
  <a href="#license">License</a>
</p>

## Description

[What's in a frame](https://medium.com/@civonamo/https-medium-com-civonamo-whats-in-a-frame-20c941376142)? In 450 B.C. the ancient Greek philosopher Zeno contemplated the nature of time and its infinite divisibility, is motion any different he wondered? Like motion, videos persist through time meaning that ontop of the regular 3 dimensions necessary to describe any image i.e. (HEIGHT, WIDTH, CHANNELS), videos require an additional 4th dimension i.e. (TIME, HEIGHT, WIDTH, CHANNELS). The aim of deepzipper is to leverage redundency in color, spatial and temporal information in order to effectively reduce (compress) video data to it's utmost limit while at the same time preserving image definition. 

Video compression consists of two sub-tasks:

- <a href="#frame-interpolation">Frame interpolation</a> 
- <a href="#colorization">Colorization</a> 

## Pipeline
![](https://github.com/jomanovic/deepzipper/blob/master/display/pipes.jpg)

## How does it work

The model works as follows:

- Compression: 
  - Decompose a video into a series of frames [FFmpeg](https://ffmpeg.org/)
  - Convert RGB image into [YUV](https://en.wikipedia.org/wiki/YUV) image
  - Remove UV channels (compression by factor of 3)
  - Encode m consecutive images (n = 3) with Frame interpolation model.encoder (compression by a factor of 8)

- Decompression:
  - Decode sequence of encoded images with Frame interpolation model.decoder (decompression by a factor of 8)
  - Colorize images with Colorization model (decompression by a factor of 3)
  - Compose series of frames into a video [FFmpeg](https://ffmpeg.org/)

## Frame interpolation

There are two different methods of frame interpolation implemented in FrameInterp.py:

- Reconstruction which attemps to construct the original image from scratch

![](https://github.com/jomanovic/deepzipper/blob/master/display/interp.jpg)

The first models of this type which I've experimented with is the [Convolutional LSTM](https://arxiv.org/abs/1506.04214). The basic idea is to transfer hidden states both forward and backwerd (in time) in order to inform compression and decompression. 

- Residual which attemps to refine the scaled encoded image to achieve decompression

![](https://github.com/jomanovic/deepzipper/blob/master/display/residual.png)

## Colorization

- Colorization models are implemented in both ConvNet_Rec.py and ConvNet_Res.py.
![](https://github.com/jomanovic/deepzipper/blob/master/display/color.jpg)

### Prerequisites

- [TensorFlow](https://www.tensorflow.org/install/) 2.0-beta0
- [TensorFlow Probability](https://www.tensorflow.org/probability/install) 0.6
- [Pandas 0.24.2](https://pandas.pydata.org/pandas-docs/stable/install.html#) 0.24.2
- [Scikit-Learn 0.21.2](https://scikit-learn.org/stable/index.html) 0.21.2
- [Matplotlib](https://matplotlib.org/) 3.1.0
- [FFmpeg](https://ffmpeg.org/)

## Results:

Here are some of the outputs obtained by using the models mentioned above:

![Mickey Mouse "Orphans Benefit" 1941](https://github.com/jomanovic/deepzipper/blob/master/display/decompressed.gif)

## License
MIT
