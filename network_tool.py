import tensorflow as tf
import numpy as np

tf.set_random_seed(6)
#%%
def conv(layer_name, x, out_channels, kernel_size=[3,3], stride=[1,1,1,1], paddings='SAME', is_pretrain=True):
    '''Convolution op wrapper, use RELU activation after convolution
    Args:
        layer_name: e.g. conv1, pool1...
        x: input tensor, [batch_size, height, width, channels]
        out_channels: number of output channels (or comvolutional kernels)
        kernel_size: the size of convolutional kernel, VGG paper used: [3,3]
        stride: A list of ints. 1-D of length 4. VGG paper used: [1, 1, 1, 1]
        is_pretrain: if load pretrained parameters, freeze all conv layers. 
        Depending on different situations, you can just set part of conv layers to be freezed.
        the parameters of freezed layers will not change when training.
    Returns:
        4D tensor
    '''
    #print(x.get_shape())
    in_channels = x.get_shape()[-1]
    with tf.variable_scope(layer_name):
        w = tf.get_variable(name='weights',
                            trainable=is_pretrain,
                            shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                            initializer=tf.truncated_normal_initializer(stddev=0.1)) # default is uniform distribution initialization
        #print(w)
        b = tf.get_variable(name='biases',
                            trainable=is_pretrain,
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.0))
        x = tf.nn.conv2d(x, w, stride, padding=paddings, name='conv')
        x = tf.nn.bias_add(x, b, name='bias_add')
        x = tf.nn.relu(x, name='relu')
        return x, w, b

#%%
def pool(layer_name, x, kernel=[1,2,2,1], stride=[1,2,2,1], paddings='SAME', is_max_pool=True):
    '''Pooling op
    Args:
        x: input tensor
        kernel: pooling kernel, VGG paper used [1,2,2,1], the size of kernel is 2X2
        stride: stride size, VGG paper used [1,2,2,1]
        paddings: padding size
        is_max_pool: boolen
                    if True: use max pooling
                    else: use avg pooling
    '''
    #print(x.get_shape())
    if is_max_pool:
        x = tf.nn.max_pool(x, kernel, strides=stride, padding=paddings, name=layer_name)
    else:
        x = tf.nn.avg_pool(x, kernel, strides=stride, padding=paddings, name=layer_name)
    return x
	
def LRN(layer_name, x):
	x = tf.nn.lrn(x, depth_radius=3, bias=1.0, alpha=0.00005,
					beta=0.75, name=layer_name)
	return x

def RELU(layer_name, x):
	x = tf.nn.relu(x, name=layer_name)
	return x
	

#%%
def batch_norm(x, training=True):
    '''Batch normlization(I didn't include the offset and scale)
    '''
    '''
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(x, [0])
    x = tf.nn.batch_normalization(x,
                                  mean=batch_mean,
                                  variance=batch_var,
                                  offset=None,
                                  scale=None,
                                  variance_epsilon=epsilon)
    '''
    x = tf.layers.batch_normalization(x, training=training, momentum=0.9)
    return x

#%%
def FC_layer(layer_name, x, out_nodes, training, leakyrelu = False):
    '''Wrapper for fully connected layers with RELU activation as default
    Args:
        layer_name: e.g. 'FC1', 'FC2'
        x: input feature map
        out_nodes: number of neurons for current FC layer
    '''
    shape = x.get_shape()
    if len(shape) == 4:
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        size = shape[-1].value

    with tf.variable_scope(layer_name):
        w = tf.get_variable('weights',
                            shape=[size, out_nodes],
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable('biases',
                            shape=[out_nodes],
                            initializer=tf.constant_initializer(0.0))
        flat_x = tf.reshape(x, [-1, size]) # flatten into 1D
        
        x = tf.nn.bias_add(tf.matmul(flat_x, w), b)
        if leakyrelu:
            x = tf.nn.leaky_relu(x)
        else:
            x = batch_norm(x, training)
            x = tf.nn.relu(x)
        return x, w

#%%
def losses(logits, labels):
    '''Compute loss from logits and labels
    Args:
        logits: logits tensor, float, [batch_size, n_classes]
        labels: label tensor, tf.int32, [batch_size]
        
    Returns:
        loss tensor of float type
    '''
    #logits = tf.Print(logits, [logits], "AT network_tool.py 124 logits", summarize=240)
    #labels = tf.Print(labels, [labels], "AT network_tool.py 125 labels", summarize=20)
    #print("AT network_tool.py 126 labels = {}".format(labels))
    #print("AT network_tool.py 127 logits = {}".format(logits))
    with tf.name_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,name='cross-entropy')
        #cross_entropy = tf.Print(cross_entropy, [cross_entropy], "AT network_tool.py 129 cross_entropy")
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope+'/loss', loss)
        return loss
    
#%%
