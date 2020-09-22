import tensorflow as tf
from convolve4d import *
import numpy as np


# In[5]:

def initializer_fullyconnect(in_channel, out_channel, stddev_factor=1.0, mode='Glorot'):
    """Initialization in the style of Glorot 2010.
    stddev_factor should be 1.0 for linear activations, and 2.0 for ReLUs"""
    if mode == 'Glorot':
        stddev = np.sqrt(stddev_factor / np.sqrt(in_channel*out_channel))
    else:
        stddev = 1.0 # standard initialization
    return tf.truncated_normal([in_channel, out_channel],
                                mean=0.0, stddev=stddev)


# In[15]:

def initializer_conv2d(in_channels, out_channels, mapsize,
                       stddev_factor=1.0, mode='Glorot'):
    """Initialization in the style of Glorot 2010.
    stddev_factor should be 1.0 for linear activations, and 2.0 for ReLUs"""
    
    if mode == 'Glorot':
        stddev = np.sqrt(stddev_factor / (np.sqrt(in_channels*out_channels)*mapsize*mapsize))
    else:
        stddev = 1.0
    
    return tf.truncated_normal([mapsize, mapsize, in_channels, out_channels],
                                mean=0.0, stddev=stddev)


# In[16]:

def initializer_conv4d(in_channels, out_channels, mapsize, mapsize2,
                       stddev_factor=1.0, mode='Glorot'):
    """Initialization in the style of Glorot 2010.
    stddev_factor should be 1.0 for linear activations, and 2.0 for ReLUs"""
    if mode == 'Glorot':
        stddev = np.sqrt(stddev_factor / (np.sqrt(in_channels*out_channels)*mapsize*mapsize*mapsize2*mapsize2))
    else:
        stddev = 1.0

#     if mapsize2 is None:
#         init = tf.truncated_normal([mapsize, mapsize, mapsize, mapsize, prev_units, num_units],
#                                 mean=0.0, stddev=stddev)
#     else:
    return tf.truncated_normal([mapsize, mapsize, mapsize2, mapsize2, in_channels, out_channels],
                                mean=0.0, stddev=stddev)


# In[12]:

def conv4d(x, in_channels, out_channels, kernel_size_1=3, kernel_size_2=3, 
           stride_1=1, stride_2=1, padding='SAME', stddev_factor=1.0, trainable=True):

    assert len(x.get_shape().as_list()) == 6 and "Previous layer must be 6-dimensional     (batch, height, width, sview, tview, channels)"
    
    weight4d_init = initializer_conv4d(in_channels, out_channels, mapsize=kernel_size_1, 
                                       mapsize2=kernel_size_2, stddev_factor=stddev_factor)
    
    filter4d = tf.get_variable(name='weight', 
                               initializer=weight4d_init, 
                               dtype=tf.float32, 
                               trainable=trainable)
    
    out = convolve4d(input=x, filter=filter4d, 
                     strides=[1, stride_1, stride_1, stride_2, stride_2, 1], 
                     padding=padding)
    
    print('conv4d layer added:', out.get_shape())
    return out


def conv_layer(x, filter_shape, stride, trainable=True):
    filter_ = tf.get_variable(
        name='weight', 
        shape=filter_shape,
        dtype=tf.float32, 
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=trainable)
    return tf.nn.conv2d(
        input=x,
        filter=filter_,
        strides=[1, stride, stride, 1],
        padding='SAME')


def relu(x):
    ''' ReLU: activation function
    '''
    out = tf.nn.relu(x)

    print('ReLU added:', out.get_shape())
    return out

def elu(x):
    ''' ELU: activation function
    '''
    out = tf.nn.elu(x)
    
    print ('ELU added:', out.get_shape())
    return out

def lrelu(x, trainbable=None):
    alpha = 0.2
    return tf.maximum(alpha * x, x)

# In[28]:

def leakyrelu(x, leak=0.2):
    '''Adds a leaky ReLU (LReLU) activation function to this model'''
    t1  = .5 * (1 + leak)
    t2  = .5 * (1 - leak)
    out = t1 * x + t2 * tf.abs(x)
    
    print('leakyrelu layer added:', out.get_shape())
    return out


# In[29]:

def prelu(x, trainable=True):
    alpha = tf.get_variable(
        name='alpha', 
        shape=x.get_shape()[-1],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=trainable)
    return tf.maximum(0.0, x) + alpha * tf.minimum(0.0, x)


# In[31]:

def sigmoid(x):
    '''Append a sigmoid (0,1) activation function layer to the model.'''
    out = tf.nn.sigmoid(x)
    
    print('sigmoid layer added:', out.get_shape())
    return out

# def softmax(x):
#     """Adds a softmax operation to this model"""

#     this_input = tf.square(x)
#     reduction_indices = list(range(1, len(this_input.get_shape())))
#     acc = tf.reduce_sum(this_input, reduction_indices=reduction_indices, keep_dims=True)
#     out = this_input / (acc+epsilon)
    
#     print 'softmax layer added:', out.get_shape()
#     return out


# In[30]:

def mean_layer(x):
    ''' Mean Layer: calculate the mean value of other dimensions except
        the first (batchsize) and the last (channels) dimensions.
    '''
    prev_shape = x.get_shape().as_list()
    reduction_indices = range(len(prev_shape))
    assert len(reduction_indices) > 2 and "Can't average a (batch, activation) tensor"
    reduction_indices = reduction_indices[1:-1]
    out = tf.reduce_mean(x, reduction_indices=reduction_indices)
    
    print('mean layer added:', out.get_shape())
    return out


def max_pooling_layer(x, size, stride):
    return tf.nn.max_pool(
        value=x,
        ksize=[1, size, size, 1],
        strides=[1, stride, stride, 1],
        padding='SAME')
# In[36]:
def batch_normalize(x, is_training, decay=0.99, epsilon=0.001, trainable=True):
    def bn_train():
        batch_mean, batch_var = tf.nn.moments(x, axes=[0, 1, 2, 3, 4])
        train_mean = tf.assign(
            pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(
            pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(
                x, batch_mean, batch_var, beta, scale, epsilon)

    def bn_inference():
        return tf.nn.batch_normalization(
            x, pop_mean, pop_var, beta, scale, epsilon)

    dim = x.get_shape().as_list()[-1]
    beta = tf.get_variable(
        name='beta',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.0),
        trainable=trainable)
    scale = tf.get_variable(
        name='scale',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.1),
        trainable=trainable)
    pop_mean = tf.get_variable(
        name='pop_mean',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=False)
    pop_var = tf.get_variable(
        name='pop_var', 
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.constant_initializer(1.0),
        trainable=False)
    print('batch_norm added:', x.get_shape())
    return tf.cond(is_training, bn_train, bn_inference)


def batch_normalize2d(x, is_training, decay=0.99, epsilon=0.001, trainable=True):
    def bn_train():
        batch_mean, batch_var = tf.nn.moments(x, axes=[0, 1, 2])
        train_mean = tf.assign(
            pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(
            pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(
                x, batch_mean, batch_var, beta, scale, epsilon)

    def bn_inference():
        return tf.nn.batch_normalization(
            x, pop_mean, pop_var, beta, scale, epsilon)

    dim = x.get_shape().as_list()[-1]
    beta = tf.get_variable(
        name='beta',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.0),
        trainable=trainable)
    scale = tf.get_variable(
        name='scale',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.1),
        trainable=trainable)
    pop_mean = tf.get_variable(
        name='pop_mean',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=False)
    pop_var = tf.get_variable(
        name='pop_var', 
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.constant_initializer(1.0),
        trainable=False)
    print('batch_norm added:', x.get_shape())
    return tf.cond(is_training, bn_train, bn_inference)


# def upscaleNdx2(x):
#     ''' Upscaling Layer: upscale the inputs by 2 times of the original size.
#     '''
#     sh = x.get_shape().as_list()
#     dim = len(sh[1:-1])
#     out = (tf.reshape(x, [-1] + sh[-dim:]))
#     for i in range(dim, 0, -1):
#         out = tf.concat([out, out], i)
#     out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
#     out = tf.reshape(out, out_size, name='upsampling')

#     if out.get_shape().as_list()[4] % 2 == 0:
#         out = out[:,:,:,1:,1:,:]

#     print 'upsampling added:', out.get_shape()
#     return out


def upscaleViewx2(x):
    ''' Upscaling Layer: upscale the view of the inputs by K times of the original size.
    '''
    sh = x.get_shape().as_list()
    dim = len(sh[1:-1])
    out = (tf.reshape(x, [-1] + sh[-dim:]))
    for i in range(dim, 2, -1):
        out = tf.concat([out, out], i)
    out_size = [-1] + [s for s in sh[1:3]] + [s * 2 for s in sh[3:-1]] + [sh[-1]]
    out = tf.reshape(out, out_size, name='upsampling')
    
    if out.get_shape().as_list()[4] % 2 == 0:
        out = out[:,:,:,1:,1:,:]

    print('angular upsampling added:', out.get_shape())
    return out


def upscale_with_piexl_shuffleNdx2(x, K=2):
    ''' Upscaling Layer: upscale the inputs by K times of the original size.
        For view use nearest neighboor
        For spacial use pixel shuffle
    '''
    
    sh = x.get_shape().as_list()
    dim = len(sh[1:-1])
    out = (tf.reshape(x, [-1] + sh[-dim:]))
    for i in range(dim, 2, -1):
        out = tf.concat([out, out], i)
    out_size = [-1] + [s for s in sh[1:3]] + [s * 2 for s in sh[3:-1]] + [sh[-1]]
    out = tf.reshape(out, out_size, name='upsampling')
    
    if out.get_shape().as_list()[4] % 2 == 0:
        out = out[:,:,:,1:,1:,:]
    
    ###### spacial pixel shuffle ######
    out = spacial_pixel_shuffle(out, K)

    print('angular-spacial upsampling added:', out.get_shape())
    return out


def pixel_shuffle(x, r, n_split):
    def PS(x, r):
        bs, a, b, c = x.get_shape().as_list()  # x.shape = [batchsize, mapsize, mapsize, channel]
        x = tf.reshape(x, (bs, a, b, r, r))
        x = tf.transpose(x, (0, 1, 2, 4, 3))
        x = tf.split(x, a, 1)
        x = tf.concat([tf.squeeze(x_) for x_ in x], 2)
        x = tf.split(x, b, 1)
        x = tf.concat([tf.squeeze(x_) for x_ in x], 2)
        return tf.reshape(x, (bs, a*r, b*r, 1))

    xc = tf.split(x, n_split, 3)
    return tf.concat([PS(x_, r) for x_ in xc], 3)


def angular_pixel_shuffle(x, rs):
    def PS(x, r):
        bs, a, b, c = x.get_shape().as_list()  # x.shape = [batchsize, mapsize, mapsize, channel]
        x = tf.reshape(x, (bs, a, b, r, r))
        x = tf.transpose(x, (0, 1, 2, 4, 3))
        x = tf.split(x, a, 1)
        x = tf.concat([tf.squeeze(x_) for x_ in x], 2)
        x = tf.split(x, b, 1)
        x = tf.concat([tf.squeeze(x_) for x_ in x], 2)
        return tf.reshape(x, (bs, a*r, b*r, 1))

    xc = tf.split(x, n_split, 3)
    return tf.concat([PS(x_, r) for x_ in xc], 3)


def spacial_pixel_shuffle(x, rs):
    batchsize, in_height, in_width, in_sview, in_tview, channels = x.get_shape().as_list()
    out_channels = channels / (rs**2)
    out_height = in_height * rs
    out_width = in_width * rs
    out_sview = in_sview
    out_tview = in_tview

    out = tf.transpose(x, (0, 5, 1, 2, 3, 4))
    out = tf.reshape(out, (batchsize, out_channels, rs, rs, in_height, in_width, in_sview, in_tview))
    out = tf.transpose(out, (0, 1, 4, 2, 5, 3, 6, 7))
    out = tf.reshape(out, (batchsize, out_channels, out_height, out_width, out_sview, out_tview))
    out = tf.transpose(out, (0, 2, 3, 4, 5, 1))

    return out


def angular_pixel_shuffle_3_5(x, rv):
    batchsize, in_height, in_width, in_sview, in_tview, channels = x.get_shape().as_list()
    out_channels = channels / (rv**2)
    out_height = in_height
    out_width = in_width
    out_sview = in_sview * rv
    out_tview = in_tview * rv

    out = tf.transpose(x, (0, 1, 2, 5, 3, 4))
    shape = tf.shape(out)
    out = tf.reshape(out , [shape[0], shape[1], shape[2], out_channels, rv, rv, in_sview, in_tview])
    out = tf.transpose(out, (0, 1, 2, 3, 6, 4, 7, 5))
    out = tf.reshape(out, (shape[0], shape[1], shape[2], out_channels, rv*in_sview, rv*in_tview))
    out = tf.transpose(out, (0, 1, 2, 4, 5, 3))

    if in_sview % 2 == 1 or in_tview % 2 == 1:
        # with tf.variable_scope('angular_shuffle'):
        #     out = conv4d(out, out_channels, out_channels, kernel_size_1=1, kernel_size_2=2, padding='VALID')
        out = out[:, :, :, :(rv*in_sview-1), :(rv*in_tview-1), :]

    return out


def batch_norm(x, scale=True):
    ''' Append a batch normalization layer to the model.
        The input features should be 2 or more dimension, where 1st dimension is batchsize
        and the last dimension is channel for mode 'NHWC'.

        input:
            features:    [batchsize, dimension1, dimension2, dimension3, ..., channels]

        refer to ArXiv 1502.03167v3 for details.
    '''


    # TBD: This appears to be very flaky, often raising InvalidArgumentError internally
    out = tf.contrib.layers.batch_norm(x, scale=scale)
    
    print('batch norm layer added:', out.get_shape())
    return out


# In[37]:

def flatten_layer(x):
    ''' Flatten: flat the outputs of the last layer.s
    '''
    input_shape = x.get_shape().as_list()
    dim = input_shape[1] * input_shape[2] * input_shape[3]
    transposed = tf.transpose(x, (0, 3, 1, 2))
    
    return tf.reshape(transposed, [-1, dim])


def pixel_shuffle_layer(x, r, n_split):
    ''' Pixel shuffle: shuffle the pixels of each images in the batchsize.
    '''
    
    def PS(x, r):
        bs, a, b, c = x.get_shape().as_list()
        x = tf.reshape(x, (bs, a, b, r, r))
        x = tf.transpose(x, (0, 1, 2, 4, 3))
        x = tf.split(x, a, 1)
        x = tf.concat([tf.squeeze(x_) for x_ in x], 2)
        x = tf.split(x, b, 1)
        x = tf.concat([tf.squeeze(x_) for x_ in x], 2)
        return tf.reshape(x, (bs, a*r, b*r, 1))

    xc = tf.split(x, n_split, 3)
    
    return tf.concat([PS(x_, r) for x_ in xc], 3)


def full_connection_layer(x, out_dim, trainable=True):
    in_dim = x.get_shape().as_list()[-1]
    W = tf.get_variable(
        name='weight',
        shape=[in_dim, out_dim],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.1),
        trainable=trainable)
    b = tf.get_variable(
        name='bias',
        shape=[out_dim],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=trainable)
    return tf.add(tf.matmul(x, W), b)
