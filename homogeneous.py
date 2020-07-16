import tensorflow as tf
import network_tool

#%%
def Gao_rgb(x, n_classes, is_pretrain=True):
    
    x, w1, b1 = network_tool.conv('rgb/conv1', x, 9, kernel_size=[5,5], stride=[1,1,1,1], is_pretrain=is_pretrain)
    x  = network_tool.pool('rgb/pool1', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
    
    x, w2, b2 = network_tool.conv('rgb/conv2', x, 18, kernel_size=[5,5], stride=[1,1,1,1], is_pretrain=is_pretrain)
    x = network_tool.pool('rgb/pool2', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
    
    x, w3, b3 = network_tool.conv('rgb/conv3', x, 36, kernel_size=[5,5], stride=[1,1,1,1], is_pretrain=is_pretrain)
    x = network_tool.pool('rgb/pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
    
    x, w4, b4 = network_tool.conv('rgb/conv4', x, 72, kernel_size=[5,5], stride=[1,1,1,1], is_pretrain=is_pretrain)
    x = network_tool.pool('rgb/pool4', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
    
    x2, fw1 = network_tool.FC_layer('rgb/fc5', x, out_nodes=144)
    #x = network_tool.batch_norm(x2)
    x, fw2 = network_tool.FC_layer('rgb/fc6', x2, out_nodes=80)
    #x = network_tool.batch_norm(x)
    #x = tf.nn.sigmoid(x)
    #x = network_tool.FC_layer('rgb/fc7', x, out_nodes=n_classes)

    return x, w1, w2, w3, w4, fw1, fw2, x2
        

#%%

#%%
def Gao_depth(x, n_classes, is_pretrain=True):
    
    x, ww1, bb1 = network_tool.conv('depth/conv1', x, 10, kernel_size=[5,5], stride=[1,1,1,1], is_pretrain=is_pretrain)
    x = network_tool.pool('depth/pool1', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
    
    x, ww2, bb2 = network_tool.conv('depth/conv2', x, 20, kernel_size=[5,5], stride=[1,1,1,1], is_pretrain=is_pretrain)
    x = network_tool.pool('depth/pool2', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
    
    x, ww3, bb3 = network_tool.conv('depth/conv3', x, 40, kernel_size=[5,5], stride=[1,1,1,1], is_pretrain=is_pretrain)
    x = network_tool.pool('depth/pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
    
    x, ww4, bb4 = network_tool.conv('depth/conv4', x, 80, kernel_size=[5,5], stride=[1,1,1,1], is_pretrain=is_pretrain)
    x = network_tool.pool('depth/pool4', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
    
    x, fww1 = network_tool.FC_layer('depth/fc5', x, out_nodes=160)
    #x = network_tool.batch_norm(x)
    x, fww2 = network_tool.FC_layer('depth/fc6', x, out_nodes=80)
    #x = network_tool.batch_norm(x)
    #x = tf.nn.sigmoid(x)
    #x = network_tool.FC_layer('depth/fc7', x, out_nodes=n_classes)

    return x, ww1, ww2, ww3, ww4, fww1, fww2
        

#%%
