***NOTE：*** *Personally, I highly recommend you readers to download the source file and open in Typora to gain the best learning experience.*

# **Convolutional Neural Networks (CNNs)**

## Introduction

Convolutional neural networks, or CNNs, are a specialized kind of neural network for processing data that has a known grid-like topology ([Goodfellow *et al.*, 2016](# Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*. MIT Press.)). Common CNNs include 1D, 2D, and 3D convolution neural networks depending on the processed data type. One-dimensional CNNs are frequently employed in sequence models when processing time-series data and applied in natural language processing ([Goldberg, 2017](# Goldberg, Y. (2017). Neural network methods for natural language processing. *Synthesis lectures on human language technologies*, *10*(1), 1-309.)). Three-dimensional CNNs are commonly utilised in medical imaging ([Dou *et al.*, 2016](# Dou, Q., Chen, H., Yu, L., Zhao, L., Qin, J., Wang, D., ... & Heng, P. A. (2016). Automatic detection of cerebral microbleeds from MR images via 3D convolutional neural networks. *IEEE transactions on medical imaging*, *35*(5), 1182-1195.)) and video processing, such as detecting motion and character behaviour ([Ji *et al.*, 2012](# Ji, S., Xu, W., Yang, M., & Yu, K. (2012). 3D convolutional neural networks for human action recognition. *IEEE transactions on pattern analysis and machine intelligence*, *35*(1), 221-231.)). In this chapter, we mainly focus on two-dimensional CNNs, a powerful family of neural networks designed precisely for computer vision and image processing. CNN-based architectures are now ubiquitous in the field of computer vision, and have become so dominant that hardly anyone today would develop a commercial application or enter a competition related to image recognition, object detection, or semantic segmentation, without building off of this approach ([Zhang *et al.*, 2022](# Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2022). *Dive into Deep Learning.* [online] Available from: https://d2l.ai/ (Accessed 27 March 2022))).

In the previous chapters, we introduced the basic [linear neural networks](# Linear Neural Networks) and [multilayer perceptrons](# Multilayer Perceptrons (MLPs)). In this chapter, after absorbing the [preliminary](# Preliminaries) ([convolution](#Convolution) and [invariance](# Invariance)), we will firstly start with the essential [components](# Components) that constitute the backbone of all convolutional networks, containing the [convolutional layers](# Convolutional Layers), [padding and stride](# Padding and Stride) in essence, the use of [multiple channels](# Multiple Input and Multiple Output Channels) at each layer, the [pooling layers](# Pooling Layers) used to aggregate information across adjacent spatial regions, and [batch normalisation](# Batch Normalisation). Then we will see some [CNN architectures](# Architectures), including the classic [LeNet](# LeNet) and some modern CNNs ([AlexNet](# AlexNet), [GoogLeNet](# GoogLeNet), and [ResNet](# ResNet)). Next, a full working CNN [code tutorial](# Code Tutorial) in Keras[^1] will be provided to solve a simple image classification problem. Lastly, we will briefly present some cutting edge [applied scenarios and relative merits](# Application) compared with alternative methods in computer vision.

## Preliminaries

### Convolution

In the renowned [Deep Learning textbook](https://www.deeplearningbook.org/) written by Ian Goodfellow and his doctoral advisors[^2] (2016), they defined that convolutional networks are simply neural networks that use convolution in place of general matrix multiplication in at least one of their layers. Let us try to comprehend the mathematical concept of convolution as the preliminary. In mathematics, the convolution between two functions (for example $f,g:\mathbb{R}^{d}\rightarrow\mathbb{R}$) is defined as 
$$
(f*g)(\mathbf{x}) = \int{f(\mathbf{z})}g(\mathbf{x}-\mathbf{z})d\mathbf{z}.
$$
It means to measure the overlap between $f$ and $g$ when one function is “flipped” and shifted by $\mathbf{x}$. Whenever handling discrete objects, the integral turns into a sum. For instance, the definition of convolution for a two-dimensional tensor is 
$$
(f*g)(i,j) = \sum_{a}{\sum_{b}{f(a,b)g(i-a, j-b)}}.
$$

- indices for $f$ : $(a,b)$ 
  indices for $g$ : $(i-a,j-b)$

### Invariance

So how does the convolutional operation work uniting neural networks to suit computer vision? Is there any constraint? Let us consider it in a specific example, the prevalent children's game "Where’s Wally". The reader’s goal is to locate Wally in various chaotic scenes bursting with activities, while Wally typically shows up in some unlikely location. Even though Wally looks characteristically, it can be unexpectedly hard to complete due to a mass of distractions.

<img src="/Users/adorex/Downloads/University of Warwick/e-BM/5-Artificial Intelligence & Deep Learning/PMA/References/Figures/Where’s_Wally_World_Record.jpg" alt="Where’s_Wally_World_Record"  />

<center>
  Figure1: Where’s Wally World Record (By William Murphy from Dublin, Ireland)
</center>

However, in deep learning we could sweep the image with a "Wally detector", because what he looks like does not depend upon where he is located. The detector could assign a score to each patch, indicating the likelihood that the patch contains Wally. More specifically, all patches of an image will be treated in the same manner, and only a small neighborhood of pixels will be used to compute the corresponding hidden representations. In summary, we should ensure two principles, translation invariance and locality.

1. Translation Invariance: In spite of where our target appears in the image, the network should respond similarly to the same patch in the earliest layers.
2. Locality: The earliest layers of the network should emphasise local regions instead of the image contents in distant regions local regions, as these local representations can eventually be aggregated to make predictions at the whole image level.

Now let us turn it into mathematics.

#### Reconsider Full-connected Layers

When the inputs and outputs are changed from vectors to matrices with two dimensions, width and height, the corresponding weights are reshaped into a four-dimensional tensor ($(h,w)\rightarrow(h^{'},w^{'})$) , and the fully-connected layer can then be represented formally as
$$
h_{i,j} = \sum_{k,l}{w_{i,j,k,l}x_{k,l}} = \sum_{a,b}{v_{i,j,a,b}x_{i+a,j+b}}.
$$

- input : $x_{k,l}$
  weight : $w_{i,j,k,l}$
  output : $h_{i,j}$

The switch in effect is simply re-indexing the subscripts $(k,l)$ (with $k=i+a$ and $l=j+b$); thereinto, the alteration from $w$ to $v$ is only in form ($v_{i,j,a,b} = w_{i,j,i+a,j+b}$) since there is a one-to-one correspondence between coefficients in both fourth-order tensors.

#### Invoke Translation Invariance

The shift in the input $x$ should simply result in the shift in the hidden representation $h$ , and $v$ do not depend on $(i,j)$. The definition for $h$ is simplified as (with $v_{i,j,a,b} = v_{a,b}$)
$$
h_{i,j} = \sum_{a,b}{v_{a,b}x_{i+a,j+b}}.
$$
It is the convonlution in neural networks. A significant progress is that the required coefficients in $v_{a,b}$ is much fewer than in $v_{i,j,a,b}$ .

#### Invoke Locality

When assessing $h_{i,j}$, we should not drift far away from location $(i,j)$​. Consequently, when $|a|,|b|>\Delta$, we should set $v_{a,b}=0$
$$
h_{i,j} = \sum_{a=-\Delta}^{\Delta}{\sum_{b=-\Delta}^{\Delta}{v_{a,b}x_{i+a,j+b}}}.
$$

## Components

### Convolutional Layers

A convolution layer is obtained by adding a constant, say $u$ (likewise not depend on $(i,j)$) to represent biases to the equation acquired at the end of the previous section
$$
h_{i,j} = u + \sum_{a=-\Delta}^{\Delta}{\sum_{b=-\Delta}^{\Delta}{v_{a,b}x_{i+a,j+b}}}.
$$
Now we know that CNNs are a special family of neural networks that contain convolutional layers. In the above equation, $v$ can be simply refer to the weights of the convolutional layers that are often learnable parameters, which are also known as convolution kernel or filter.

There is a single problem left by now: In reality, the image data we work with are usually RGB images, i.e. they are not two-dimensional but three-dimensional tensors consisting of height, width and channels (red, green, and blue). The solution is quite simple: since the input is a three-order tensor now, we index it as $x_{i,j,k}$. Accordingly, we adjust the convolutional filter ($v_{a,b}\rightarrow v_{a,b,c}$) and formulate the hidden representations ($h_{i,j}\rightarrow h_{i,j,d}$). What we intend to output is not a single hidden representation but an entire vector of hidden representations corresponding to each spatial location. It can be imagined as many two-dimensional grids stacked on top of each other. We will discuss this in detail in the [multiple input and multiple output channels](# Multiple Input and Multiple Output Channels) section. To support multiple channels in both inputs $x$ and hidden representations $h$, we can add a fourth coordinate to $v$ : $v_{a,b,c,d}$. With all this being considered, now we have
$$
h_{i,j,d} = \sum_{a=-\Delta}^{\Delta}{\sum_{b=-\Delta}^{\Delta}{\sum_{c}{v_{a,b,c}x_{i+a,j+b,c}}}}.
$$

- indices for the output channels in the hidden representations $h$ : $d$
  kernel of the convolutional layer for multiple channels : $v$

#### The Cross-Correlation Operation

In fact, the elementwise multiplication of the input tensor with the kernel tensor that we defined above should be a correlation operation, whereas, in the standard terminology in deep learning literature, it is defined as a convolution, which actually requires flipping the two-dimensional kernel tensor both horizontally and vertically and then performs the cross-correlation operation with the input tensor.

##### 2D cross-correlation operation

![../_images/correlation.svg](https://d2l.ai/_images/correlation.svg)

<center>
  Figure2: Two-dimensional cross-correlation operation
  (Source: Dive into Deep Learning)
</center>

In Figure2, the light blue portions are the first output element as well as the input and kernel tensor elements used for the output computation: $0\times0+1\times1+3\times2+4\times3=19$ . In the two-dimensional cross-correlation operation, we begin with the convolution window at the upper-left corner of the input tensor and slide it across the input tensor, both from left to right and top to bottom. Once the convolution window slides to a certain position, the input subtensor contained in that window and the kernel tensor are multiplied elementwise, and the resulting tensor is summed up rendering a single scalar value. This result gives the value of the output tensor at the corresponding location. In the example above, the four elements of the output tensor (height: 2, width: 2) are derived from four two-dimensional cross-correlation operations. As shown in the figure, the output size is slightly smaller than the input size, because the kernel has width and height greater than one,  and it only performs the cross-correlation with locations where the kernel fits exactly within the image. To sum up, two-dimensional cross-correlation operation contains the following components:

| Symbol       | Meaning | Size                                   |
| ------------ | ------- | -------------------------------------- |
| $\mathrm{X}$ | Input   | $n_{h}\times n_{w}$                    |
| $\mathrm{W}$ | Kernel  | $k_{h}\times k_{w}$                    |
| $b$          | Bias    | $b\in\mathbb{R}$                       |
| $\mathrm{Y}$ | Output  | $(n_{h}-k_{h}+1)\times(n_{w}-k_{w}+1)$ |

- $h$ : height
  $w$ : width

Convolutional layer can be defined as
$$
\mathrm{Y} = \mathrm{X} \star \mathrm{W} + b.
$$

- $\star$ : cross-correlation operation

$\mathrm{W}$ and $b$ are learnable parameters (learned from data). The kernel size is a hyper parameter controlling the before-mentioned locality.

##### Cross-correlation vs Convolution

<img src="/Users/adorex/Downloads/University of Warwick/e-BM/5-Artificial Intelligence & Deep Learning/PMA/References/Figures/Comparison_convolution_correlation.jpg" alt="Comparison_convolution_correlation" style="zoom: 25%;" />

<center>
  Figure3: Visual comparison of convolution and cross-correlation (By Cmglee)
</center>

It is noteworthy that with kernels learned from data, the outputs of convolutional layers remain unaffected regardless of such layers’ performed operations (either strict convolution or cross-correlation). In other words, there is no difference between them in practical use due to symmetry showed in the above figure. Mathematically, they can be expressed as

- 2D cross-correlation
  $$
  y_{i,j} = \sum_{a=1}^{h}{\sum_{b=1}^{w}{w_{a,b}x_{i+a,j+b}}}
  $$

- 2D convolution
  $$
  y_{i,j} = \sum_{a=1}^{h}{\sum_{b=1}^{w}{w_{-a,-b}x_{i+a,j+b}}}
  $$

### Padding and Stride

From the previous section we know that the output shape of the convolutional layer is determined by the shape of the input and the shape of the convolution kernel. In addition, padding and stride can be used to adjust the dimensionality of the data effectively in CNNs. Let us look specifically at why we need them and how they work.

#### Padding

After applying many successive convolutions, it turns out that final outputs that are considerably smaller than the input. For instance, if we start with a $32\times32$ pixel image, 7 layers of $5\times5$ convolutions reduce the image to $4\times4$ pixels, obliterating any referable information on the boundaries of the original image. Padding can increase the height and width of the output, which is often used to give the output the same height and width as the input. The straightforward mean is to add extra pixels of filler (typically set the values to 0) around the boundary of the input image, thus increasing the effective size of the image.

![../_images/conv-pad.svg](https://d2l.ai/_images/conv-pad.svg)

<center>
  Figure4: Two-dimensional cross-correlation with padding
  (Source: Dive into Deep Learning)
</center>

The output shape will be $(n_{h}-k_{h}+p_{h}+1)\times(n_{w}-k_{w}+p_{w}+1)$, with adding a total of $p_{h}$ rows and $p_{w}$ columns of padding (roughly half on top and half on bottom). So to give the input and output the same height and width, we set $p_{h}=k_{h}-1$ and $p_{w}=k_{w}-1$, which will make it easier to predict the output shape of each layer when constructing the network. Moreover, we commonly use convolution kernels with odd height and width values for CNNs, because it results in the same number of rows on top and bottom and the same number of columns on left and right while padding, with the spatial dimensionality being preserved. If kh is even, one possibility is to pad $\lceil p_{h}/2\rceil$[^2] rows on the top of the input and $\lfloor p_{h}/2\rfloor$[^3] rows on the bottom (both sides of the width are padded in the same way). Besides, For any two-dimensional tensor $\mathrm{X}$, when the kernel size is odd and the number of padding rows and columns on all sides are the same, and its output has the same height and width as the input, we can infer that the output $y_{i,j}$ is calculated by cross-correlation of the input and convolution kernel with the window centered on $x_{i,j}$.

#### Stride

In other cases, we may want to reduce the dimensionality drastically for computational efficiency, taking the unwieldy original input resolution as an example.  The resolution of the output requires reduction, for example reducing the height and width of the output to only $\frac{1}{n}$ of the height and width of the input ($n$ is an integer greater than 1). We can move our window more than one element at a time, skipping the intermediate locations. The stride in effect refers to the number of rows and columns traversed per slide.

![../_images/conv-stride.svg](https://d2l.ai/_images/conv-stride.svg)

<center>
  Figure5: Two-dimensional cross-correlation with strides
  (Source: Dive into Deep Learning)
</center>

As shown in the example above, it is a two-dimensional cross-correlation operation with a stride of 3 vertically and 2 horizontally, i.e. 3 and 2 respectively for height and width. By observing the light blue blocks, we can recognie that the convolution window need to slide down three rows or two columns to the right to output the second element of the first column or the second element of the first row. When the convolution window continues to slide two columns to the right on the input, there is no output because the input element can not fill the window (unless another column of padding is added). In general, given the stride for the height $s_{h}$ and the stride for the width is $s_{w}$, the output shape is $\lfloor(n_{h}-k_{h}+p_{h}+s_{h})/s_{h}\rfloor\times\lfloor(n_{w}-k_{w}+p_{w}+s_{w})/s_{w}\rfloor$. If setting $p_{h}=k_{h}-1$ and $p_{w}=k_{w}-1$, the output shape will be simplified to $\lfloor(n_{h}+s_{h}-1)/s_{h}\rfloor\times\lfloor(n_{w}+s_{w}-1)/s_{w}\rfloor$. Furthermore, if the input height and width are divisible by the strides on the height and width (e.g. if the stride height and width are both 2 and the input is a multiple of 2), then the output shape will be $(n_{h}/s_{h})\times(n_{w}/s_{w})$. In practice, we rarely use inhomogeneous strides or padding, i.e., we usually have $p_{h}=p_{w}$ and $s_{h}=s_{w}$.

### Multiple Input and Multiple Output Channels

We talked in brief about channels in the [convolutional layers](# Convolutional Layers) section. We ought to be aware that one channel may work in the simple pictures not containing much information, but for those complicated images like "Where's Wally", it may result in loss of information if converting them into grayscale images using only grey channel, so it is essential to add channels into the mix. What we are facing now is not the simplified example above with just a single input and a single output channel, but the colour images that have the standard RGB channels to indicate the amount of red, green and blue. The change is that our inputs and hidden representations both become three-dimensional tensors (e.g. each RGB input image has shape $3\times h\times w$). In short, Multiple channels can be used to extend the model parameters of the convolutional layer.

#### Multiple Input Channels

When the input data contain multiple channels, we need to construct a convolution kernel with the same number of input channels as the input data, so that it can perform cross-correlation with the input data. Assuming that the number of channels for the input data is $c_{i}$ ($c$ means channel, $i$ means input), the number of input channels of the convolution kernel also needs to be $c_{i}$. For now, our cross-correlation computation should be
$$
\mathrm{Y} = \sum_{i=0}^{c_{i}}{\mathrm{X_{i,:,:}} \star \mathrm{W_{i,:,:}}}.
$$

| Symbol       | Meaning | Size                            |
| ------------ | ------- | ------------------------------- |
| $\mathrm{X}$ | Input   | $c_{i}\times n_{h}\times n_{w}$ |
| $\mathrm{W}$ | Kernel  | $c_{i}\times k_{h}\times k_{w}$ |
| $\mathrm{Y}$ | Output  | $m_{h}\times m_{w}$             |

Each channel has its own bias.

Again, let us see a specific example. In the figure below, there is a two-dimensional cross-correlation with two input channels. the first output element as well as its computation is indicated in light blue. As we can see, the two channels use separate kernel and then the results are added up. In fact, it is just performing one cross-correlation operation per channel and then doing the summation to the results.

![../_images/conv-multi-in.svg](https://d2l.ai/_images/conv-multi-in.svg)

<center>
  Figure6: Cross-correlation computation with 2 input channels
  (Source: Dive into Deep Learning)
</center>

#### Multiple Output Channels

When it comes to the output channels, it turns out that having multiple channels at each layer is also crucial like we mentioned before. This is because typically greater channel depth can be achieved by downsampling to trade off spatial resolution, and actually increasing the channel dimension is the method, especially when going higher up in the neural network. More straightforwardly, we can regard each channel as responding to some different set of features as they are designed to recognise a particular pattern (the next input channels can recognise and combine those patterns), but it is a bit more complicated in reality since representations are not learned independent but are rather optimised to be jointly practical. 

To get an output with multiple channels, we can create a kernel tensor of shape $c_{i}\times k_{h}\times k_{w}$ for every output channel with concatenating them on the output channel dimension, so that the shape of the convolution kernel is $c_{o}\times c_{i}\times k_{h}\times k_{w}$ ($o$ means output). In cross-correlation operations, the result on each output channel is calculated from the convolution kernel corresponding to that output channel and takes input from all channels in the input tensor. To conclude, now the computation should be
$$
\mathrm{Y_{i,:,:}} = \mathrm{X} \star \mathrm{W_{i,:,:}}\ \ \ for\ i=1,\dots,c_{o}.
$$

| Symbol       | Meaning | Size                                        |
| ------------ | ------- | ------------------------------------------- |
| $\mathrm{X}$ | Input   | $c_{i}\times n_{h}\times n_{w}$             |
| $\mathrm{W}$ | Kernel  | $c_{o}\times c_{i}\times k_{h}\times k_{w}$ |
| $\mathrm{Y}$ | Output  | $c_{o}\times m_{h}\times m_{w}$             |

#### $1\times 1$ Convolution Kernel

In terms of the $1\times 1$ convolution (i.e. $k_{h}=k_{w}=1$), it looks useless due to the loss of ability to recognise patterns consisting of interactions among adjacent elements in the height and width dimensions; however, it can merge the information derived from different channels, which can be used to adjust the number of channels between network layers and to control model complexity.

![../_images/conv-1x1.svg](https://d2l.ai/_images/conv-1x1.svg)

<center>
  Figure7: The cross-correlation computation uses the 1✕1 convolution kernel
  (Source: Dive into Deep Learning)
</center>

The above figure shows the cross-correlation computation using the $1\times1$ convolution kernel with 3 input channels and 2 output channels, in which the inputs and outputs have the same height and width. Each element in the output is derived from a linear combination of elements at the same position in the input image. It can be considered as a fully-connected layer with $n_{h}n_{w}\times c_{i}$ and $c_{o}\times c_{i}$ weights plus the bias, applied at every single pixel location to transform the $c_{i}$ corresponding input values into $c_{o}$ output values. In other words, it is equal to reshape the input into an array with $n_{h}n_{w}\times c_{i}$ shape.

### Pooling Layers

 The main purpose of pooling layers is to mitigate the sensitivity of convolutional layers to location. One of the major benefits of a pooling layer is to alleviate the excessive sensitivity of the convolutional layer to location. Taking vertical edge detection as an instance, if we use a kernel with $1\times2$ shape like $\begin{bmatrix}1&-1\end{bmatrix}$, the input array $\mathrm{X}$ can lead to the output array $\mathrm{Y}$, accomplishing a simple edge detection. However, as we can see in the figure, the first and third columns of $\mathrm{Y}$ are both zero, which indicates that the convolutional layer is relatively sensitive to location, i.e. the shift by one pixel might cause vastly different output. It is not what we expect because objects hardly ever occur exactly at the same place in the real world, even with a tripod and a stationary object, vibration of the camera due to the movement of the shutter might shift everything by a pixel. What we require now is translation invariance to some extent as we have to process various images with assorted light, location, scale and appearance.

<img src="/Users/adorex/Library/Application Support/typora-user-images/image-20220417061240260.png" alt="image-20220417061240260" style="zoom: 25%;" />

<center>
  Figure8: The input array X and output array Y when vertically detecting edge
</center>

#### Maximum Pooling and Average Pooling

Pooling operators consist of a fixed-shape window (also known as the pooling window) slid over all regions in the input according to its stride, computing a single output for each location traversed by the pooling window. The distinction is pooling operators are deterministic without any parameter. When taking the input elements in the pooling window, the maximum pooling operation assigns the maximum value as the output and the average pooling operation assigns the average value as the output. The light blue blocks are the the first output element and its input tensor elements used for the output computation: $\mathrm{max}(0,1,3,4)=4$. A pooling layer with a pooling window shape of $p\times q$ is called a $p\times q$ pooling layer. The pooling operation is called $p\times q$ pooling.

![../_images/pooling.svg](https://d2l.ai/_images/pooling.svg)

<center>
  Figure9: Two-dimensional maximum pooling with a pooling window shape of 2✕2
  (Source: Dive into Deep Learning)
</center>

Let us return to the example mentioned at the beginning of this section.The result of pooling layer is that whether the values of the input $\mathrm{X[i,j]}$ and $\mathrm{X[i,j+1]}$ are different or not, it always output $\mathrm{Y[i,j]=1}$, which means that if the pattern recognised by the convolutional layer moves no more than one element in height or width, it can be still detected by the utilisation of a $2\times2$ pooling layer. To sum up, maximum pooling combined with a stride larger than 1 can be used to reduce the spatial dimensions (e.g., width and height).

#### Padding, Stride, and Multiple Channels

Similar to convolutional layers, we can specify the padding, stride and window size for the pooling layer as hyper parameters to achieve a desired output shape. The difference is that they are not learnable but can be specified. There is another notable point that the pooling layer’s number of output channels is the same as the number of input channels. This is because when processing multi-channel input data, the pooling layer pools each input channel separately rather than sum them up over channels. Besides, we already have the the convolutional layers for combining multiple channels.

### Batch Normalisation

The essential reason for batch normalisation is to obtain more stable intermediate output values in each layer throughout the neural network by utilizing the mean and standard deviation of the minibatch, which results in continuous adjustment to the intermediate output of the neural network. Its principle is not too sophisticated. We first normalise the inputs in each training iteration by subtracting their mean and dividing by their standard deviation, both of which are estimated based on the statistics of the current minibatch, then we apply a scale coefficient and a scale offset.

Fomally, we calculate the mean $\mu_{B}$ and standard deviation $\sigma_{B}^2$ as follows ($B$ means batch)
$$
\mu_{B} = \frac{1}{|B|}\sum_{i\in B}{x_{i}}
\ \ \ \ \ \ and\ \ \ \ \ \ 
\sigma_{B}^2 = \frac{1}{|B|}\sum_{i\in B}{(x_{i}-\mu_{B})^2+\epsilon}.
$$
(Add a small constant $\epsilon>0$ to the variance estimate to ensure that we never attempt division by zero)

Then the batch normalisation can be expressed as
$$
x_{i+1} = \gamma ~ \frac{x_{i}-\mu_{B}}{\sigma_{B}} + \beta
$$

- $\gamma$ and $\beta$ are learnable parameters

The batch normalisation methods for fully-connected layers and convolutional layers are slightly different. For fully-connected layers, its output $\mathbf{h}$ can be expressed as follows
$$
\mathbf{h} = \phi(\mathrm{BN}(\mathbf{Wx+b})).
$$

- $\mathbf{W}$ : the weight parameter
  $\mathbf{b}$ : the bias parameter
  $\mathbf{x}$ : the input to the fully-connected layer
  $\mathbf{h}$ : the output to the fully-connected layer
  $\phi()$ : the activation function
  $\mathrm{BN}()$ : the batch normalisation

When it comes to the convolution layers, when the convolution has multiple output channels, we need to carry out batch normalisation for each of the outputs of these channels, and each channel has its own scale and shift parameters, both of which are scalars. Denoting the height and width of the convolution output respectively by $p$ and $q$, and assuming that our minibatches contain $m$ examples, For convolutional layers, we carry out each batch normalization over the $m⋅p⋅q$ elements per output channel simultaneously, which differs from working on the feature dimension for the fully-connected layers. 

It is worth mentioning that batch normalisation was originally invented to reduce internal covariate shift ([Ioffe and Szegedy, 2015](# Ioffe, S., & Szegedy, C. (2015). Batch normalization: accelerating deep network training by reducing internal covariate shift. *arXiv preprint arXiv:1502.03167*.)), but subsequently [Teye *et al*. (2018)](# Teye, M., Azizpour, H., & Smith, K. (2018). Bayesian uncertainty estimation for batch normalized deep networks. *arXiv preprint arXiv:1802.06455*.) and [Luo *et al*. (2018)](# Luo, P., Wang, X., Shao, W., & Peng, Z. (2018). Towards understanding regularization in batch normalization. *arXiv preprint*.) supposed that it controls model complexity by adding noise to each minibatch.

## Architectures

### LeNet

LeNet was introduced by and named for [LeCun (1998)](# LeCun, Y., Bottou, L., Bengio, Y., Haffner, P., & others. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, *86*(11), 2278–2324.) in 1989, a researcher at [AT&T Bell Labs](https://history.aip.org/phn/21507001.html), for the purpose of recognizing handwritten digits in images. The prototype was designed to identify handwritten zip code numbers provided by the US Postal Service, and eventually it was adapted to recognise digits for processing deposits in ATM machines. In addition, LeNet left a worldwidely prevalent database, [the MNIST database](http://yann.lecun.com/exdb/mnist/), which has a training set of 60,000 examples, and a test set of 10,000 examples. To this day, it is still a good database for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting.

![image-20220417182143286](/Users/adorex/Library/Application Support/typora-user-images/image-20220417182143286.png)

<center>
  Figure10: Data flow in LeNet
  (Source: Dive into Deep Learning)
</center>

Let us turn to the data flow in LeNet. Firstly, we input $32\times32$ image into the first convolutional layer (with a $5\times5$ kernel and a sigmoid activation function), which has 6 output channels. The output is called feature map whose shape is given by batch size, number of channel, height, width. Then it is transfered into a subsequent average pooling layer with a $2\times2$ window shape. ReLU function and max-pooling had not yet been made in the 1990s while they work better. Next, we pass the output from the pooling layer to the second convolutional layer ($5\times5$ kernel and sigmoid activation function as well) with 16 channels this time, then perform pooling like before. Finally, we flatten each example in the output from the convolutional block to the dense block. In other words, we transform this four-dimensional input into the two-dimensional input expected by fully-connected layers. The dense block has three fully-connected layers, with 120, 84, and 10 outputs respectively. The 10-dimensional output layer corresponds to the number of possible output classes.

To sum up, LeNet is a successful CNN published early with outstanding performance on computer vision tasks, in which the convolutional layers are used to learn spatial Information in image while increase the number of channels, and fully-connected layers are employed for transfering the output into classes.

### AlexNet

AlexNet is a new variant of a CNN proposed by [Alex Krizhevsky, Ilya Sutskever, and Geoff Hinton (2012)](# Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. *Advances in neural information processing systems* (pp. 1097–1105).) that achieved excellent performance in the 2012 ImageNet Large Scale Visual Recognition Challenge and won it by a phenomenally large margin. The ultimate breakthrough it made can be attributed to two key advance, data and hardware. On the one had, the [ImageNet](https://image-net.org/) dataset was released in 2009, challenging researchers to learn models from 1 million examples, 1000 each from 1000 distinct categories of objects. On the other hand, the implementation of Graphical processing units (GPUs) in deep CNN has changed the pattern making deep learning feasible. GPUs were optimised for high throughput $4\times4$ matrix-vector products needed for many computer graphics tasks, which is strikingly similar to that required to calculate convolutional layers, while deep learning models are acquisitive to computational resources.

AlexNet has a similar structure to that of LeNet, but uses more convolutional layers and a larger parameter space to fit the large-scale ImageNet dataset as a key step from shallow to deep networks used nowadays. Figure11 compares the strcture of AlexNet (a simplified version) to the original LeNet. As the figure shows, AlexNet is much deeper than the comparatively small LeNet consisting of five convolutional layers, two fully-connected hidden layers, and one fully-connected output layer.

![../_images/alexnet.svg](https://d2l.ai/_images/alexnet.svg)

<center>
  Figure11: From LeNet (left) to AlexNet (right)
  (Source: Dive into Deep Learning)
</center>

Let us check the architecture details illustrated in the diagram below. There are four main modifications:

1. The convolution window shape is $11\times11$ in the first layer. This is because most images in ImageNet with more pixels are much more higher and wider than that in the MNIST, a larger window is demmanded to capture the target. The shape in the second layer is reduced to $5\times5$, followed by $3\times3$.
2. Maximum pooling layers are added with a window shape of 3×3 and a stride of 2 after the first, second, and fifth convolutional layers. 
3. AlexNet has ten times more convolution channels than LeNet.
4. There are two fully-connected layers with 4096 outputs after the last convolutional layer, which produce model parameters of nearly 1 GB. Because of the limited memory in early GPUs, the original AlexNet used a dual data stream design, but GPU memory is comparatively abundant now.

![image-20220418030128659](/Users/adorex/Library/Application Support/typora-user-images/image-20220418030128659.png)

<center>
  Figure12: Data flow in AlexNet
  (Source: PowerPoint in class)
</center>

Besides, AlexNet utilised a simpler ReLU activation function mitigating vanishing gradient without the exponentiation operation in the sigmoid activation function, which is more applicable to different parameter initialisation methods. Moreover, AlexNet added a great deal of image data augmentation, such as flipping, clipping, and color changes, enabling the greater model robustness and larger sample size that effectively reduces overfitting. Lastly, AlexNet controls the model complexity of the fully-connected layer by [dropout](# Dropout), while LeNet only uses [weight decay](# Weight Decay). 

What we could learn from this promotion is that, dropout, ReLU, and preprocessing (like data augmentation) are the other key steps in achieving excellent performance in computer vision tasks, except data and hardware. Though it seems that there are only a few more lines of code in AlexNet’s implementation than in LeNet, there has been many years for researchers to embrace this conceptual change and take advantage of its excellent experimental results due to the lack of efficient computational tools in that period.

### GoogLeNet

It is hard to say that the name GoogLeNet not pays homage to the initial LeNet, whereas they are almost distinct. GoogLeNet proposed by [Szegedy *et al.* (2015)](# Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., … Rabinovich, A. (2015). Going deeper with convolutions. *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 1–9).) provides an interesting solution to select the hyper parameters in the convolutional layer. We may be bothered when choosing the size of the convolutional kernel, but GooLeNet will not because they desire them all. The figure below illustrates the basic convolutional block in GoogLeNet called an "Inception block", whose name is probably inspired by Christopher Nolan's film "Inception". 

![../_images/inception.svg](https://d2l.ai/_images/inception.svg)

<img src="/Users/adorex/Library/Application Support/typora-user-images/image-20220419013905135.png" alt="image-20220419013905135" style="zoom:60%;" />

<center>
  Figure13: Structure of the Inception block with (down) and without (up) channel number
  (Source: Dive into Deep Learning)
</center>

As depicted in the figure13, the inception block consists of four parallel paths. The first three paths extract information from different spatial sizes by using convolutional layers with window sizes of $1\times1$, $3\times3$, $5\times5$ ; thereinto, the middle two paths perform a $1\times1$ convolution on the input to reduce the number of channels, reducing the model’s complexity. The fourth path uses a $3\times3$ maximum pooling layer, followed by a $1\times1$ convolutional layer to change the number of channels. It is notable that the four paths all use appropriate padding to give the input and output the same height and width. In the end, the block’s output is composed of the outputs from each path, concatenated at the channel dimension. The ratio of the number of channels assigned in the Inception block is obtained through a large number of experiments on the ImageNet dataset.  The hyperparameters of the Inception block are the number of output channels per layer. In short, the blue blocks above are employed to extract spatial information while the white blocks are only used to change the number of channels. Comparing with single $3\times3$ or $5\times5$ convolutional layers, the inception block has fewer parameter number and less complexity.

|                     | Parameters | FLOPS[^5] |
| ------------------- | ---------- | --------- |
| **Inception**       | 0.16 M     | 128 M     |
| **$3\times3$ Conv** | 0.44 M     | 346 M     |
| **$5\times5$ Conv** | 1.22 M     | 963 M     |

![../_images/inception-full.svg](https://d2l.ai/_images/inception-full.svg)

<center>
  Figure14: The GoogLeNet architecture
  (Source: Dive into Deep Learning)
</center>

The above figure is a simplified GoogLeNet architecture who uses a stack of a total of 9 inception blocks and global average pooling to generate its estimates. 

| Stage   |                            Graph                             | Description                                                  |
| :------ | :----------------------------------------------------------: | ------------------------------------------------------------ |
| Stage 5 | <img src="/Users/adorex/Library/Application Support/typora-user-images/image-20220418162746351.png" alt="image-20220418162746351" style="zoom:85%;" /> | Output shape: $1024\times1\times1$<br />(a vector)           |
| Stage 4 | <img src="/Users/adorex/Library/Application Support/typora-user-images/image-20220418162557836.png" alt="image-20220418162557836" style="zoom:85%;" /> | Increasing the channel number                                |
| Stage 3 | <img src="/Users/adorex/Library/Application Support/typora-user-images/image-20220418162330023.png" alt="image-20220418162330023" style="zoom:50%;" /> | Different channel assignment                                 |
| Stage 2 | <img src="/Users/adorex/Library/Application Support/typora-user-images/image-20220418161957962.png" alt="image-20220418161957962" style="zoom:50%;" /> | Similar to AlexNet and LeNet<br />(with larger height and width kept) |
| Stage 1 | <img src="/Users/adorex/Library/Application Support/typora-user-images/image-20220418161708129.png" alt="image-20220418161708129" style="zoom:50%;" /> | Similar to AlexNet and LeNet                                 |

Several variants of the inception block appear afterwards. The summary of their main progress is depicted in the table below.

| Variants          | Description                                                  | References                                                   |
| ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Inception BN (V2) | - Add a batch normalization layer                            | [Ioffe & Szegedy, 2015](# Ioffe, S., & Szegedy, C. (2015). Batch normalization: accelerating deep network training by reducing internal covariate shift. *arXiv preprint arXiv:1502.03167*.) |
| Inception V3      | - Make adjustments to the Inception block<br />$5\times5\ \rightarrow\ $Several $3\times3$ <br />$5\times5\ \rightarrow\ $$1\times7$ and $7\times1$<br />$3\times3\ \rightarrow\ $$1\times3$ and $3\times1$<br />- Use label smoothing for model regularization | [Szegedy *et al*., 2016](# Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the inception architecture for computer vision. *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 2818–2826).) |
| Inception V4      | - Include the residual connection                            | [Szegedy *et al*., 2017](# Szegedy, C., Ioffe, S., Vanhoucke, V., & Alemi, A. A. (2017). Inception-v4, inception-resnet and the impact of residual connections on learning. *Thirty-First AAAI Conference on Artificial Intelligence*.) |

### ResNet

With the advancement of CNNs as they go deeper, the confronting problem is how adding layers can increase the complexity and expressiveness of the network. ResNet solved the problem as the winner of the ImageNet Large Scale Visual Recognition Challenge in 2015 proposed by [He *et al*. (2016)](# He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 770–778).), which has a major and profound influence on the design of subsequent deep neural networks, both for convolutional and sequential nature. Its heart is the residual block showed in the comparison graph below. Before we introduce it there is another mathematical concept required to be clarified, the function classes.

Given a specific network architecture $\mathcal{F}$ who includes learning rates and other hyperparameter settings, for all $f\in\mathcal{F}$, there are some set of parameters (e.g. weights and biases) that can be obtained through training on a suitable dataset. 
Assuming that $f^{*}$ is "the chosen one" we truly desire, if $f^{*}\in\mathcal{F}$, we can acquire it by training. But if not, we try to find a $f^{*}_{\mathcal{F}}$ which is our best bet within $\mathcal{F}$. So how to get a function which is extremely close to the chosen $f^{*}$? We may need a more powerful architecture $\mathcal{F^{'}}$, i.e. a $f^{*}_{\mathcal{F^{'}}}$ that is better than $f^{*}_{\mathcal{F}}$. As a  result, we need nestedfunction classes to ensure the way to $f^{*}$. For non-nested function classes, a larger function class does not guarantee to get closer to the function $f^{*}$.

Thus, only if larger function classes contain the smaller ones are we guaranteed the growing expressive power of the network. For deep neural networks, if we can train the newly-added layer into an identity function $f(\mathbf{x})= \mathbf{x}$, the new model will at least be as effective as the original model. As the new model may get a better solution to fit the training dataset, the added layer might make it easier to reduce training errors.

![../_images/residual-block.svg](https://d2l.ai/_images/residual-block.svg)

<center>
  Figure15: A regular block (left) vs. a residual block (right)
  (Source: Dive into Deep Learning)
</center>

Now let us look at the figure above, denoteing the input by $\mathbf{x}$, $f(\mathbf{x})$ is the desired underlying mapping we want to obtain by learning, also used as the input to the activation function on the top. On the left of the figure, what need to be turned out from  learning is directly the mapping $f(\mathbf{x})$, while on the right, it is the residual mapping $f(\mathbf{x})-\mathbf{x}$ which can learn the identity function more easily, such as pushing parameters in the weight layer to zero so that $f(\mathbf{x})$ is the above-mentioned identity function. With residual blocks, inputs can forward propagate faster through the residual connections across layers. 

Let us focus on the ResNet-18 architecture, the first two layers of ResNet are the same as those of the GoogLeNet we described before: the $7\times7$ convolutional layer with 64 output channels and a stride of 2 is followed by the $3\times3$ maximum pooling layer with a stride of 2. The difference is the batch normalization layer added after each convolutional layer in ResNet. ResNet uses four modules made up of residual blocks, each of which uses several residual blocks with the same number of output channels. The number of channels in the first module is the same as the number of input channels. In the first residual block for each of the subsequent modules, the number of channels is doubled compared with that of the previous module, and the height and width are halved. Lastly, just like GoogLeNet, a global average pooling layer, followed by the fully-connected layer output is added. Together with , there are 18 layers in total. The name ResNet-18 is derived from 18 layers in total (4 convolutional layers in each module excluding the $1\times1$ layer, the first $7\times7$ convolutional layer and the final fully-connected layer).



![../_images/resnet18.svg](https://d2l.ai/_images/resnet18.svg)

<center>
  Figure16: The ResNet-18 architecture
  (Source: Dive into Deep Learning)
</center>

## Code Tutorial

This tutorial will walk through a CNN in Keras. We will be working with a [10 monkey species](https://www.kaggle.com/datasets/slothkong/10-monkey-species) set from Kaggle, where we are predicting which species a monkey is. The full code can be viewed [here](https://github.com/AdOrEx913/AIDL/blob/main/AIDL_PMA_Code_Tutorial.ipynb).

To begin with, we have to prepare our API key from Kaggle.[^Hover your cursor here to see how]

```python
from google.colab import files
files.upload()
```

```python
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 /root/.kaggle/kaggle.json
```

Download the data now.[^Where to find the folliwing information]

```python
!kaggle datasets download -d slothkong/10-monkey-species
```

![image-20220418194454309](/Users/adorex/Library/Application Support/typora-user-images/image-20220418194454309.png)	

Unzip the downloaded zip folder and create a new folder.

```python
!unzip 10-monkey-species.zip -d 10-monkey-species
```

![image-20220418195010267](/Users/adorex/Library/Application Support/typora-user-images/image-20220418195010267.png)	

Create some variables to store the path to the directories.

```python
train_dir = "/content/10-monkey-species/training/training"
val_dir = "/content/10-monkey-species/validation/validation"
```

Add a generator to create augmentations for our training data, while validation data only need normalisation (i.e. to be divided by 255 pixels). The data augmentations we involved here by setting parameters are rotation, shifting, shearing, zooming, flipping, and mode filling.

```python
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest',
                                   rescale = 1./255)
val_datagen = ImageDataGenerator(rescale = 1./255)
```

Create our dataset variables.

| Variable    | Value | Description                      |
| ----------- | ----- | -------------------------------- |
| height      | 128   | the height of Input images       |
| width       | 128   | the width of Input images        |
| channels    | 3     | the channels of Input images     |
| batch_size  | 64    | the batch size                   |
| num_classes | 10    | the number of the output classes |

```python
height = 128
width = 128
channels = 3
batch_size = 64
num_classes = 10

training_set = train_datagen.flow_from_directory(train_dir,
                                                 target_size = (height, width),
                                                 batch_size = batch_size,
                                                 seed = 7,
                                                 class_mode = "categorical")

val_set = val_datagen.flow_from_directory(val_dir,
                                          target_size = (height, width),
                                          batch_size = batch_size,
                                          seed = 7,
                                          shuffle = False,
                                          class_mode = 'categorical')
```

![image-20220418202859674](/Users/adorex/Library/Application Support/typora-user-images/image-20220418202859674.png)	

Let us start building our CNN model now. The graph below presents its structure.

<img src="/Users/adorex/Downloads/University of Warwick/e-BM/5-Artificial Intelligence & Deep Learning/PMA/Model Structure.svg" alt="Model Structure" style="zoom: 40%;" />

<center>
  Figure17: The sructure of our example model
</center>

```python
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D 

cnn = Sequential()

cnn.add(Conv2D(filters=32, kernel_size=3, padding='SAME', activation='relu', 
               input_shape=(width, height, channels)))
cnn.add(BatchNormalization())
cnn.add(MaxPooling2D(pool_size=2, strides=2, padding='SAME'))
cnn.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
cnn.add(BatchNormalization())
cnn.add(MaxPooling2D(pool_size=2, strides=2, padding='SAME'))
cnn.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
cnn.add(BatchNormalization())
cnn.add(MaxPooling2D(pool_size=2, strides=2, padding='SAME'))
cnn.add(Flatten())
cnn.add(Dense(128, activation='relu'))
cnn.add(Dropout(0.25))
cnn.add(Dense(num_classes, activation='softmax'))

cnn.summary()
```

![image-20220418205000199](/Users/adorex/Library/Application Support/typora-user-images/image-20220418205000199.png)	

It turns out that millions of parameters are ready to be learned. In this example, we employ the `adam` and `accuracy` respectively as our optimizer and metric funtion (to judge the performance of our model). More details about the description of the arguments in `compile()` method can be looked up [here](https://keras.io/api/models/model_training_apis/#compile-method). And we only trained 30 epochs this time just for demonstration. 

```python
from keras import losses

cnn.compile(optimizer = 'adam', 
            loss = losses.CategoricalCrossentropy(), 
            metrics = ['accuracy'])

history = cnn.fit(training_set, validation_data = val_set, epochs = 30)
```

![image-20220418212054262](/Users/adorex/Library/Application Support/typora-user-images/image-20220418212054262.png)

Visualise the results.

```python
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 0.7])
plt.legend(loc='upper left')
```

<img src="/Users/adorex/Downloads/University of Warwick/e-BM/5-Artificial Intelligence & Deep Learning/PMA/accuracy_result.png" alt="accuracy_result"  />	

We can see the accuracy on both the training set and the validation set have steadily improved. Probably, more training is required for our model to achieve satisfactory performance.

## Application

As we mentioned at the beginning, CNNs play an influential role in computer vision and image processing. In this part, we will walk through several prominent business applications of CNNs. First and foremost, the primary domain applying CNNs is image recognition and classification. From the fundamental image tagging to visual search and recommendation engines, CNNs are capable of deconstruct image and learn their features from the input image data instead of extracting them manually. Taking the recommendation engine as an example, on the e-commerce platforms, we can usually see "Products related to this item" when you scroll down a product page to view its details, or "You may also want to buy" when you purchased some stuff. CNNs are a practical tool to achieve this based on the users' shopping behaviours. The other noteworthy utilisation of image recognition is face recognition which is one of the best-known branch. Comparing with other creatures like animals or plants, recognising human faces is more complicated tasks for the computer to understand as it demands higher accuracy and numerous samples. Nowadays, filters are widely used when taking photos, jumping from the autogenerated basic layout of the face and attach new elements or effects, which displays the usage of CNNs on entertainment. They can also be applied to formal scenes such as identification and surveillance. Another image recognition variation is optical character recognition (OCR), designed to process written and print symbols, graphs, and charts, which basically combines with natural language processing (NLP), for instance, personal signature recognition. Besides, CNNs are likewise welcomed the medical world, for example, detecting the anomalies on the X-ray images, assessing health risk, and discovering drug.

In comparison to other algorithms in computer vision, including k-nearest neighbors algorithm (KNN), support-vector machine (SVM), and backpropagation neural networks (BPNN) applied in MLPs, the core superiority is that they preferably leverage the prior knowledge that nearby pixels are typically related to each other to build efficient models for learning from image data. It is a creative way for exploiting some known structures in images. Moreover, with retaining the spatial structure, CNNs contain more parsimonious models that require far fewer parameters.

## References

###### Dou, Q., Chen, H., Yu, L., Zhao, L., Qin, J., Wang, D., ... & Heng, P. A. (2016). Automatic detection of cerebral microbleeds from MR images via 3D convolutional neural networks. *IEEE transactions on medical imaging*, *35*(5), 1182-1195.

###### Goldberg, Y. (2017). Neural network methods for natural language processing. *Synthesis lectures on human language technologies*, *10*(1), 1-309.

###### Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*. MIT Press.

###### He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 770–778).

###### He, K., Zhang, X., Ren, S., & Sun, J. (2016). Identity mappings in deep residual networks. *European conference on computer vision* (pp. 630–645).

###### Ioffe, S., & Szegedy, C. (2015). Batch normalization: accelerating deep network training by reducing internal covariate shift. *arXiv preprint arXiv:1502.03167*.

###### Ji, S., Xu, W., Yang, M., & Yu, K. (2012). 3D convolutional neural networks for human action recognition. *IEEE transactions on pattern analysis and machine intelligence*, *35*(1), 221-231.

###### Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. *Advances in neural information processing systems* (pp. 1097–1105).

###### Luo, P., Wang, X., Shao, W., & Peng, Z. (2018). Towards understanding regularization in batch normalization. *arXiv preprint*.

###### LeCun, Y., Bottou, L., Bengio, Y., Haffner, P., & others. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, *86*(11), 2278–2324.

###### Szegedy, C., Ioffe, S., Vanhoucke, V., & Alemi, A. A. (2017). Inception-v4, inception-resnet and the impact of residual connections on learning. *Thirty-First AAAI Conference on Artificial Intelligence*.

###### Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the inception architecture for computer vision. *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 2818–2826).

###### Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., … Rabinovich, A. (2015). Going deeper with convolutions. *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 1–9).

###### Teye, M., Azizpour, H., & Smith, K. (2018). Bayesian uncertainty estimation for batch normalized deep networks. *arXiv preprint arXiv:1802.06455*.

###### Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2022). *Dive into Deep Learning.* [online] Available from: https://d2l.ai/ (Accessed 27 March 2022)

## Appendix

### Linear Neural Networks

#### Linear Model

$n$ : the number of examples in the dataset

$i$ : the data sample index

Each input : $\mathbf{x}^{(i)}=[x_{1}^{(i)},x_{2}^{(i)}]^{\top}$

Corresponding label : $y^{(i)}$

- When the inputs consist of $d$ features, the prediction $\hat{y}$ is :
  $\hat{y}=w_{1}x_{1}+\dots+w_{d}x_{d}+b$

  - $w$ : the weights
  - $b$ : the bias

- Express using a dot product :

  $\hat{y}=\mathbf{w}^{\top}\mathbf{x}+b$

#### Loss Function

A measure of fitness

The loss function quantifies the distance between the real and predicted value of the target.

The most popular loss function in regression problems is the squared error.

<img src="https://d2l.ai/_images/fit-linreg.svg" alt="../_images/fit-linreg.svg" style="zoom:75%;" />

<center>
  Figure18: Fit data with a linear model
  (Source: Dive into Deep Learning)
</center>

#### Gradient Descent

Gradient Descent iteratively reduces the error by updating the parameters in the direction that incrementally lowers the loss function. The most naive application of gradient descent consists of taking the derivative of the loss function, which is an average of the losses computed on every single example in the dataset. In practice, this can be extremely slow because we must pass over the entire dataset before making a single update. Thus, we will often settle for sampling a random minibatch of examples every time we need to compute the update, a variant called: Minibatch Stochastic Gradient Descent. The steps of the algorithm are the following: 

1. Initialize the values of the model parameters, typically at random; 
2. Iteratively sampling random minibatches from the data, updating the parameters in the direction of the negative gradient.

#### Hyperparameter Tuning

The parameters that are tunable but not updated in the training loop are called hyperparameters. Hyperparameter tuning is the process by which hyperparameters are chosen, and typically requires adjusting based on the results of the training loop as assessed on a separate validation dataset (or validation set).

#### Neural Network Diagram

Linear regression is a single-layer neural network. Linear regression models can be considered as neural networks consisting of just a single artificial neuron, or as single-layer neural networks.

<img src="https://d2l.ai/_images/singleneuron.svg" alt="../_images/singleneuron.svg" style="zoom:100%;" />

<center>
  Figure19: Linear regression is a single-layer neural network
  (Source: Dive into Deep Learning)
</center>

##### Feature Dimensionality

The number of inputs (or feature dimensionality) in the input layer

##### Fully-connected Layer / Dense Layer

Since for linear regression, every input is connected to every output (in this case there is only one output), we can regard this transformation (the output layer in the above figure) as a fully-connected layer or dense layer.

#### Softmax Regression

Softmax regression is also a single-layer neural network just as in linear regression. And since the calculation of each output depends on all inputs, its output layer can also be described as fully-connected layer.

![../_images/softmaxreg.svg](https://d2l.ai/_images/softmaxreg.svg)

<center>
  Figure20: Softmax regression is a single-layer neural network
  (Source: Dive into Deep Learning)
</center>

### Multilayer Perceptrons (MLPs)

#### Hidden Layers

![../_images/mlp.svg](https://d2l.ai/_images/mlp.svg)

<center>
  Figure21: An MLP with a hidden layer of 5 hidden units
  (Source: Dive into Deep Learning)
</center>

#### Activation Functions

##### ReLU Function

ReLU : rectified linear unit

$\mathrm{ReLu}(x)=\mathrm{max}(x,0)$

<img src="https://d2l.ai/_images/output_mlp_76f463_15_0.svg" alt="../_images/output_mlp_76f463_15_0.svg" style="zoom:80%;" />

| Input    | Derivative | Result               |
| -------- | ---------- | -------------------- |
| Negative | 0          | the argument vanish  |
| Positive | 1          | the argument through |

##### Sigmoid Function

The sigmoid function transforms its inputs, for which values lie in the domain $\mathbb{R}$, to outputs that lie on the interval $(0, 1)$. It squashes any input in the range $(-\infty, +\infty)$ to some value in the range $(0, 1)$ :

$\mathrm{sigmoid}(x)=\frac{1}{1+\mathrm{exp}(-x)}$

<img src="https://d2l.ai/_images/output_mlp_76f463_39_0.svg" alt="../_images/output_mlp_76f463_39_0.svg" style="zoom:80%;" />

##### Tanh Function

The tanh (hyperbolic tangent) function also squashes its inputs, transforming them into elements on the interval $(-1,1)$ :

$\mathrm{tanh}(x)=\frac{1-\mathrm{exp}(-2x)}{1+\mathrm{exp}(-2x)}$

<img src="https://d2l.ai/_images/output_mlp_76f463_63_0.svg" alt="../_images/output_mlp_76f463_63_0.svg" style="zoom:80%;" />

#### Model Selection

##### Error

The training error is the error of our model as calculated on the training dataset, while generalization error is the expectation of our model’s error were we to apply it to an infinite stream of additional data examples drawn from the same underlying data distribution as our original sample.

##### Underfitting & Overfitting

Underfitting means that a model is not able to reduce the training error. When training error is much lower than validation error, there is overfitting.

The techniques used to combat overfitting are called regularisation. It adds a penalty term to the loss function on the training set to reduce the complexity of the learned model.

![../_images/capacity-vs-error.svg](https://d2l.ai/_images/capacity-vs-error.svg)

<center>
  Figure22: Influence of model complexity on underfitting and overfitting
</center>

#### Weight Decay

Weight decay (commonly called $L_{2}$ regularisation), might be the most widely-used technique for regularising parametric machine learning models. The technique measures the complexity of a function by its distance from zero. One particular choice for keeping the model simple is weight decay using an $L_{2}$ penalty. This leads to weight decay in the update steps of the learning algorithm. The weight decay functionality is provided in optimizers from deep learning frameworks.

#### Dropout

Beyond controlling the number of dimensions and the size of the weight vector, dropout is yet another tool to avoid overfitting. Often they are used jointly. Dropout replaces an activation $h$ with a random variable with expected value $h$. It is only used during training.

#### Forward Propagation & Backward Propagation

Forward propagation sequentially calculates and stores intermediate variables within the computational graph defined by the neural network. It proceeds from the input to the output layer. Backpropagation sequentially calculates and stores the gradients of intermediate variables and parameters within the neural network in the reversed order.

When training deep learning models, forward propagation and back propagation are interdependent. Training requires significantly more memory than prediction.

#### Gradient Vanishing

The sigmoid’s gradient vanishes both when its inputs are large and when they are small. When backpropagating through many layers, unless we are in the zone where the inputs to many of the sigmoids are close to zero, the gradients of the overall product may vanish. Vanishing and exploding gradients are common issues in deep networks. Great care in parameter initialization is required to ensure that gradients and parameters remain well controlled. ReLU activation functions mitigate the vanishing gradient problem that can accelerate convergence.

#### Distribution Shift

In many cases training and test sets do not come from the same distribution. This is called distribution shift. Under the corresponding assumptions, covariate and label shift can be detected and corrected for at test time. Failure to account for this bias can become problematic at test time.



[^1]: Keras is a deep learning API written in Python, developed with a focus on enabling fast experimentation, running on top of the machine learning platform [TensorFlow](https://www.tensorflow.org/).
[^2]: The inventors of generative adversarial networks (GANs) 
[^3]: $\lceil\ \rceil$​ denotes the ceiling function
[^4]: $\lfloor\ \rfloor$​ denotes the floor function
[^Hover your cursor here to see how]: 1⃣️Set up an account (if you haven't got one already). 2⃣️Click on your avatar in the top right and "Account" from the dropdown menu. 3⃣️Scroll down the page and you'll find a button to "Create New API Token". 4⃣️Your API key is downloaded to your PC now.
[^Where to find the folliwing information]: Click the three vertical dots on the right on the dataset page, and select "Copy API command".
[^5]: Floating point Operations Per Second