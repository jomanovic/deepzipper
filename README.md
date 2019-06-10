![](https://github.com/jomanovic/deepzipper/blob/master/display/logo.jpg)

## What is it?

deepzipper is an intelligent video compression framework utilizing nerual nets. It has two main components:

- Frame Interpolation
- Colorization

## Before I show you my zipper, here is my pipeline:
![](https://github.com/jomanovic/deepzipper/blob/master/display/pipes.jpg)

## How does it work?

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

## Frame interpolation:

There are two different methods of frame interpolation:

- Reconstruction which attemps to construct the original image from scratch

![](https://github.com/jomanovic/deepzipper/blob/master/display/interpolation.jpg)

- Residual which attemps to refine the scaled encoded image to achieve decompression

![](https://github.com/jomanovic/deepzipper/blob/master/display/residual.png)

## Colorization:

![](https://github.com/jomanovic/deepzipper/blob/master/display/colorization.jpg)

### Prerequisites

- [TensorFlow](https://www.tensorflow.org/install/) 2.0-beta0
- [TensorFlow Probability](https://www.tensorflow.org/probability/install) 0.6
- [Pandas 0.24.2](https://pandas.pydata.org/pandas-docs/stable/install.html#) 0.24.2
- [Scikit-Learn 0.21.2](https://scikit-learn.org/stable/index.html) 0.21.2
- [Matplotlib](https://matplotlib.org/) 3.1.0

## Results:

Here are some of the outputs obtained by using the models mentioned above:

![Mickey Mouse "Orphans Benefit" 1941](https://github.com/jomanovic/deepzipper/blob/master/display/decompressed.gif)
