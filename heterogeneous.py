import tensorflow as tf
import network_tool
tf.set_random_seed(6)
#%%
def Gao_rgb(x, n_classes, is_pretrain=True, training=True):
    
    # x, w1, b1 = network_tool.conv('rgb/conv1', x, 96, kernel_size=[7,7], stride=[1,2,2,1], paddings='VALID', is_pretrain=is_pretrain)
    # x = network_tool.pool('rgb/pool1', x, kernel=[1,3,3,1], stride=[1,2,2,1], paddings='VALID', is_max_pool=True)
    
    # x, w2, b2 = network_tool.conv('rgb/conv2', x, 256, kernel_size=[5,5], stride=[1,2,2,1], paddings='VALID', is_pretrain=is_pretrain)
    # x = network_tool.pool('rgb/pool2', x, kernel=[1,3,3,1], stride=[1,2,2,1], paddings='VALID', is_max_pool=True)
    
    # x, w3, b3 = network_tool.conv('rgb/conv3', x, 384, kernel_size=[3,3], stride=[1,1,1,1], paddings='SAME', is_pretrain=is_pretrain)
    
    # x, w4, b4 = network_tool.conv('rgb/conv4', x, 384, kernel_size=[3,3], stride=[1,1,1,1], paddings='SAME', is_pretrain=is_pretrain)
    
    # x, w5, b5 = network_tool.conv('rgb/conv5', x, 256, kernel_size=[3,3], stride=[1,1,1,1], paddings='SAME', is_pretrain=is_pretrain)
    # x = network_tool.pool('rgb/poo5', x, kernel=[1,3,3,1], stride=[1,2,2,1], paddings='VALID', is_max_pool=True)
    
    # x, fw1 = network_tool.FC_layer('rgb/fc6', x, out_nodes=512, training=training)
	
    # x2, fw2 = network_tool.FC_layer('rgb/fc7', x, out_nodes=144, training=training)

    # x, fw3 = network_tool.FC_layer('rgb/fc8', x2, out_nodes=80, training=training)

    x, w1, b1 = network_tool.conv('rgb/conv1', x, 32, kernel_size=[3,3], stride=[1,1,1,1], paddings='SAME', is_pretrain=is_pretrain)
    x = network_tool.pool('rgb/pool1', x, kernel=[1,2,2,1], stride=[1,2,2,1], paddings='VALID', is_max_pool=True)
    
    print("AT heterogeneous.py 28 w1 = {}".format(w1))
    w1 = tf.Print(w1, [w1], "AT heterogeneous.py 28 w1", summarize=20)

    x, w2, b2 = network_tool.conv('rgb/conv2', x, 32, kernel_size=[3,3], stride=[1,1,1,1], paddings='SAME', is_pretrain=is_pretrain)
    x = network_tool.pool('rgb/pool2', x, kernel=[1,2,2,1], stride=[1,2,2,1], paddings='VALID', is_max_pool=True)

    x, w3, b3 = network_tool.conv('rgb/conv3', x, 32, kernel_size=[3,3], stride=[1,1,1,1], paddings='SAME', is_pretrain=is_pretrain)
    
    x, w4, b4 = network_tool.conv('rgb/conv4', x, 32, kernel_size=[3,3], stride=[1,1,1,1], paddings='SAME', is_pretrain=is_pretrain)
    
    x, w5, b5 = network_tool.conv('rgb/conv5', x, 64, kernel_size=[3,3], stride=[1,1,1,1], paddings='SAME', is_pretrain=is_pretrain)
    x = network_tool.pool('rgb/pool5', x, kernel=[1,2,2,1], stride=[1,2,2,1], paddings='VALID', is_max_pool=True)

    x, w6, b6 = network_tool.conv('rgb/conv6', x, 64, kernel_size=[3,3], stride=[1,1,1,1], paddings='SAME', is_pretrain=is_pretrain)
    
    x, w7, b7 = network_tool.conv('rgb/conv7', x, 64, kernel_size=[3,3], stride=[1,1,1,1], paddings='SAME', is_pretrain=is_pretrain)
    
    x, w8, b8 = network_tool.conv('rgb/conv8', x, 128, kernel_size=[3,3], stride=[1,2,2,1], paddings='SAME', is_pretrain=is_pretrain)
    
    x, w9, b9 = network_tool.conv('rgb/conv9', x, 128, kernel_size=[3,3], stride=[1,1,1,1], paddings='SAME', is_pretrain=is_pretrain)

    x, w10, b10 = network_tool.conv('rgb/conv10', x, 256, kernel_size=[3,3], stride=[1,1,1,1], paddings='SAME', is_pretrain=is_pretrain)
    x = network_tool.pool('rgb/pool10', x, kernel=[1,2,2,1], stride=[1,2,2,1], paddings='VALID', is_max_pool=True)
    
    #x, fw1 = network_tool.FC_layer('rgb/fc11', x, out_nodes=512, training=training)
	
    x2, fw2 = network_tool.FC_layer('rgb/fc12', x, out_nodes=144, training=training)

    x, fw3 = network_tool.FC_layer('rgb/fc13', x2, out_nodes=80, training=training)
    
    return x
        

#%%

#%%
def Gao_depth(x, n_classes, is_pretrain=True, training=True):
    
    # paddings='SAME'
    # x, ww1, bb1 = network_tool.conv('depth/conv1', x, 10, kernel_size=[5,5], stride=[1,2,2,1], paddings=paddings, is_pretrain=is_pretrain)
    # x = network_tool.pool('depth/pool1', x, kernel=[1,2,2,1], stride=[1,2,2,1], paddings=paddings, is_max_pool=True)
    
    # x, ww2, bb2 = network_tool.conv('depth/conv2', x, 20, kernel_size=[5,5], stride=[1,2,2,1], paddings=paddings, is_pretrain=is_pretrain)
    # x = network_tool.pool('depth/pool2', x, kernel=[1,2,2,1], stride=[1,2,2,1], paddings=paddings, is_max_pool=True)
    
    # x, ww3, bb3 = network_tool.conv('depth/conv3', x, 40, kernel_size=[5,5], stride=[1,2,2,1], paddings=paddings, is_pretrain=is_pretrain)
    # x = network_tool.pool('depth/pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], paddings=paddings, is_max_pool=True)
    
    # x, ww4, bb4 = network_tool.conv('depth/conv4', x, 80, kernel_size=[5,5], stride=[1,2,2,1], paddings=paddings, is_pretrain=is_pretrain)
    # x = network_tool.pool('depth/pool4', x, kernel=[1,2,2,1], stride=[1,2,2,1], paddings=paddings, is_max_pool=True)
    
    # x, fww1 = network_tool.FC_layer('depth/fc5', x, out_nodes=160, training=training)
    # x, fww2 = network_tool.FC_layer('depth/fc6', x, out_nodes=80, training=training)

    x, ww1, bb1 = network_tool.conv('depth/conv1', x, 32, kernel_size=[3,3], stride=[1,1,1,1], paddings='SAME', is_pretrain=is_pretrain)
    x = network_tool.pool('depth/pool1', x, kernel=[1,2,2,1], stride=[1,2,2,1], paddings='VALID', is_max_pool=True)

    x, ww2, bb2 = network_tool.conv('depth/conv2', x, 32, kernel_size=[3,3], stride=[1,1,1,1], paddings='SAME', is_pretrain=is_pretrain)
    
    x, ww3, bb3 = network_tool.conv('depth/conv3', x, 32, kernel_size=[3,3], stride=[1,1,1,1], paddings='SAME', is_pretrain=is_pretrain)
    x = network_tool.pool('depth/pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], paddings='VALID', is_max_pool=True)

    x, ww4, bb4 = network_tool.conv('depth/conv4', x, 32, kernel_size=[3,3], stride=[1,1,1,1], paddings='SAME', is_pretrain=is_pretrain)
    
    x, ww5, bb5 = network_tool.conv('depth/conv5', x, 32, kernel_size=[3,3], stride=[1,1,1,1], paddings='SAME', is_pretrain=is_pretrain)
    x = network_tool.pool('depth/pool5', x, kernel=[1,2,2,1], stride=[1,2,2,1], paddings='VALID', is_max_pool=True)
    
    x, ww6, bb6 = network_tool.conv('depth/conv6', x, 64, kernel_size=[3,3], stride=[1,1,1,1], paddings='SAME', is_pretrain=is_pretrain)
    
    x, ww7, bb7 = network_tool.conv('depth/conv7', x, 64, kernel_size=[3,3], stride=[1,1,1,1], paddings='SAME', is_pretrain=is_pretrain)
    x = network_tool.pool('depth/pool7', x, kernel=[1,2,2,1], stride=[1,2,2,1], paddings='VALID', is_max_pool=True)

    x, w8, b8 = network_tool.conv('depth/conv8', x, 128, kernel_size=[3,3], stride=[1,1,1,1], paddings='SAME', is_pretrain=is_pretrain)
    
    x, w9, b9 = network_tool.conv('depth/conv9', x, 128, kernel_size=[3,3], stride=[1,1,1,1], paddings='SAME', is_pretrain=is_pretrain)
    x = network_tool.pool('depth/pool9', x, kernel=[1,3,3,1], stride=[1,1,1,1], paddings='VALID', is_max_pool=True)

    
    x, fw1 = network_tool.FC_layer('depth/fc10', x, out_nodes=160, training=training)
	
    x, fw2 = network_tool.FC_layer('depth/fc11', x, out_nodes=80, training=training)
   
    return x
        

#%%
