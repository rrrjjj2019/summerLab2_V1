from __future__ import division
import os
import os.path
import math
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import process_input
import network_tool
import heterogeneous		#heterogeneous arch
import homogeneous			#homogeneous arch
import draw
import parameter_GFLOPS

from tensorflow.python import debug as tf_debug

tf.set_random_seed(6)
DIM = 160
SPLIT_RATIO = 0.3
N_CLASSES = 119
BATCH_SIZE = 10
IMG_W = 64  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 64
CAPACITY = 2000
global learning_rate
#learning_rate = 0.0001
IS_PRETRAIN = True
EPOCH = 20
min_time = 1000000000
First_test = True

#Second_test = False

DATA_DIR = '/home/NASICLab/nmsocug1/lab2_Li/lab2/dataset/'
LOG_DIR = '/home/NASICLab/nmsocug1/lab2_Li/lab2/tensorflow/'

pre_images_rgb_tra, pre_images_depth_tra, pre_labels_tra, pre_images_rgb_tes, pre_images_depth_tes, pre_labels_tes, N_TRAIN, N_TEST = process_input.pre_process(DATA_DIR,SPLIT_RATIO)

MAX_STEP = int(math.ceil(N_TRAIN/BATCH_SIZE) * EPOCH)
over50epoch = int(math.ceil(N_TRAIN/BATCH_SIZE) * 50)
step_per_epoch = int(math.ceil(N_TRAIN/BATCH_SIZE))
pretrain = False
global inistep
inistep = 0

print("AT train.py 39")

def rotation(images_rgb, images_depth, phase):
    if True:
        if phase >= 10:
            rotate = 0.0
            std = 0
            print('phase >= 101')
        elif phase >= 9:
            rotate = 0.0
            std = 0
            print('phase >= 91')
        elif phase >= 8:
            rotate = 0.0
            std = 0
            print('phase >= 81')
        elif phase >= 7:
            rotate = 0.0
            std = 0
            print('phase >= 71')
        elif phase >= 6:
            rotate = np.random.uniform(-10,10)
            std = 5
            print('phase >= 61')
        elif phase >= 5:
            rotate = np.random.uniform(-8,8)
            std = 4
            print('phase >= 51')
        elif phase >= 4:
            rotate = np.random.uniform(-6,6)
            std = 3
            print('phase >= 41')
        elif phase >= 3:
            rotate = np.random.uniform(-4,4)
            std = 2
            print('phase >= 31')
        elif phase >= 2:
            rotate = np.random.uniform(-2,2)
            std = 1
            print('phase >= 21')
        elif phase >= 1:
            rotate = 0.0
            std = 0.0
            print('phase >= 11')
        else:
            rotate = 0.0
            std = 0
            print('phase <= 10')
        
        paddings_rgb = tf.constant([[0, 0], [19, 19], [19, 19], [0, 0]])
        images_rgb = tf.pad(images_rgb, paddings_rgb, 'CONSTANT')
        noise_rgb = tf.random_normal((images_rgb.shape), mean=0.0, stddev=std, dtype=tf.float32)
        images_rgb = tf.add(images_rgb, noise_rgb)
        images_rgb = tf.contrib.image.rotate(images_rgb, rotate, 'BILINEAR')
        images_rgb = tf.image.resize_image_with_crop_or_pad(images_rgb, IMG_W, IMG_H)
        
        paddings_depth = tf.constant([[0, 0], [19, 19], [19, 19], [0, 0]])
        images_depth = tf.pad(images_depth, paddings_depth, 'CONSTANT')
        noise_depth = tf.random_normal((images_depth.shape), mean=0.0, stddev=std, dtype=tf.float32)
        temp = tf.add(images_depth, noise_depth)
        noise_depth = tf.where(tf.greater(temp, 255), 255-images_depth, noise_depth)
        images_depth = tf.add(images_depth, noise_depth)
        images_depth = tf.contrib.image.rotate(images_depth, rotate, 'BILINEAR')
        images_depth = tf.image.resize_image_with_crop_or_pad(images_depth, IMG_W, IMG_H)
        
        return images_rgb, images_depth
        
def train():

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    
    #init = tf.global_variables_initializer()
    sess = tf.Session(config=tfconfig)
	
    #sess.run(init)

    with tf.Session() as sess:

        #print("#####################AT train.py 103 ######################")
        my_global_step = tf.Variable(0, name='global_step', trainable=False)
        global_phase = tf.Variable(0, trainable=False, dtype=tf.int32)
        phase_placeholder = tf.placeholder(tf.int32, name='phase')
        switch_phase_op = global_phase.assign(phase_placeholder)
        images_rgb, images_depth, labels, temp_rgb, temp_depth = process_input.get_batch(pre_images_rgb_tra, pre_images_depth_tra, pre_labels_tra,
                                                IMG_W, IMG_H, 
                                                BATCH_SIZE, CAPACITY, True)
        
        #print("AT train.py 127 temp_rgb = {}".format(temp_rgb))

        sess.run(global_phase.initializer)
        #print("AT train.py 130 global_phase = {}".format(sess.run(global_phase)))
        new_rgb, new_depth = tf.case(
            {tf.equal(global_phase, 1): lambda: rotation(temp_rgb, temp_depth, 1),
            tf.equal(global_phase, 2): lambda: rotation(temp_rgb, temp_depth, 2),
            tf.equal(global_phase, 3): lambda: rotation(temp_rgb, temp_depth, 3),
            tf.equal(global_phase, 4): lambda: rotation(temp_rgb, temp_depth, 4),
            tf.equal(global_phase, 5): lambda: rotation(temp_rgb, temp_depth, 5),
            tf.equal(global_phase, 6): lambda: rotation(temp_rgb, temp_depth, 6),
            tf.equal(global_phase, 7): lambda: rotation(temp_rgb, temp_depth, 7),
            tf.equal(global_phase, 8): lambda: rotation(temp_rgb, temp_depth, 8),
            tf.equal(global_phase, 9): lambda: rotation(temp_rgb, temp_depth, 9),
            tf.greater_equal(global_phase, 10): lambda: rotation(temp_rgb, temp_depth, 10)},
            default = lambda: rotation(temp_rgb, temp_depth, 0),
            exclusive=True)
        #print("AT train.py 144 new_rgb = {}".format(new_rgb))
        new_rgb = tf.map_fn(lambda new_rgb: tf.image.per_image_standardization(new_rgb), new_rgb)
        #print("AT train.py 146 new_rgb = {}".format(new_rgb))
        new_depth = tf.map_fn(lambda new_depth: tf.image.per_image_standardization(new_depth), new_depth)
        rgb_fc2 = heterogeneous.Gao_rgb(new_rgb, N_CLASSES, IS_PRETRAIN, True)# output is rgb after conv2
        depth_fc2 = heterogeneous.Gao_depth(new_depth, N_CLASSES, IS_PRETRAIN, True)# output is depth agter conv2

        print('----------')
        print(rgb_fc2)
        print(depth_fc2)

        
        distance_loss_rgb, distance_loss_depth, distance_loss_corre = calculate_dis_train(rgb_fc2, depth_fc2, labels, 2, 0.5, 1, 0.5)

        prelogits = tf.concat([rgb_fc2, depth_fc2], 1, name='concat')

        #print("AT train.py 159 prelogits = {}".format(prelogits))
        #prelogits = tf.Print(prelogits, [prelogits], "AT train.py 163 prelogits")
        
        prelogits = tf.check_numerics(prelogits, "prelogits")

        sigmoid_alpha = 1 #Add a constant to enlarge the values input into
        prelogits = tf.nn.sigmoid(tf.multiply(prelogits,sigmoid_alpha))
        
        #prelogits = tf.Print(prelogits, [prelogits], "AT train.py 169 prelogits")
        #print("AT train.py 165 prelogits = {}".format(prelogits))
        prelogits = tf.check_numerics(prelogits, "prelogits")

        logits = slim.fully_connected(prelogits, N_CLASSES, activation_fn=None,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                    weights_regularizer=slim.l2_regularizer(0.0),
                    scope='Logits', reuse=False)

        cf_loss = network_tool.losses(logits, labels)
        #cf_loss = tf.Print(cf_loss, [cf_loss], "AT train.py 180 cf_loss")

        loss = cf_loss #+ 0.1*distance_loss_rgb + 0.1*distance_loss_depth + 0.001*distance_loss_corre

        #loss = tf.Print(loss, [loss], "AT train.py 176 loss")

        #loss = tf.Variable(1000, trainable=True, dtype=tf.float32)

        #print("AT train.py 176 loss = {}".format(loss)) 
        
        learning_rate = 0.0001
        optimizer = tf.train.RMSPropOptimizer(learning_rate= learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step= my_global_step)
        
        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_list += bn_moving_vars
        saver = tf.train.Saver(var_list=var_list, max_to_keep = 175)
        summary_op = tf.summary.merge_all()

        # tfconfig = tf.ConfigProto(allow_soft_placement=True)
        # tfconfig.gpu_options.allow_growth = True
        
        init = tf.global_variables_initializer()
        # sess = tf.Session(config=tfconfig)
        
        sess.run(init)
        
        
        if pretrain:
            print("Reading pretrain %s checkpoints..." % LOG_DIR)
            ckpt = tf.train.get_checkpoint_state(LOG_DIR)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
                inistep = int(global_step)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
        
        # global inistep
        # global learning_rate
        check = 1
        Max = 0
        inistep = 0
        loss_list = []
        accu_list = []
        try:
            for step in np.arange(inistep,MAX_STEP):
                if coord.should_stop():
                        break
                phase = max((int(step*BATCH_SIZE/N_TRAIN)-1)//10, 0)         
                sw_op, t = sess.run([switch_phase_op, global_phase], feed_dict={phase_placeholder:phase})
                
                if phase == check:
                    print('global phase {}'.format(phase))
                    print('t {}'.format(t))
                    print('sw_op {}'.format(sw_op))				
                    check += 1
                # print('global phase {}'.format(phase))
                # print('t {}'.format(t))
                # print('sw_op {}'.format(sw_op))	 				
                images, images_1, rgb_fc, depth_fc, loss_value, _  = sess.run([temp_rgb, images_rgb, rgb_fc2, depth_fc2, loss, train_op])
                
                if step > over50epoch:
                    learning_rate = 0.00001

                if step % 50 == 0:                 
                    print ('Epoch:%d ,Step: %d, loss: %.4f' % (int(step*BATCH_SIZE/N_TRAIN),step, loss_value))

                if step % 100 == 0:
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)                
        
                if step % 2000 == 0 or (step + 1) == MAX_STEP:
                    checkpoint_path = os.path.join(LOG_DIR, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
                
                
                if step % 12000 == 0:
                    precision = test()
                    if precision > Max:
                        print("I have the highest accuracy when step = %d!" % step)
                        Max = precision
                
                if step % step_per_epoch == 0:
                    # if(Second_test):
                    #     import datetime
                    #     start = datetime.datetime.now()
                    #     precision = test()
                    #     end = datetime.datetime.now()
                    #     inteval = start - end
                    #     with open("time.txt","a+") as f:
                    #         f.write('time: {};'.format((inteval.microsecond)/6576))
                    # else:
                    #     precision = test()
                    
                    global min_time
                    import datetime
                    start = datetime.datetime.now()
                    precision = test()
                    end = datetime.datetime.now()
                    inteval = start - end
                    if(inteval.microseconds < min_time):
                        min_time = inteval.microseconds
                    with open("time.txt","w") as f:
                        f.write('time: {};'.format((min_time)/6576))

                    accu_list.append(precision)
                    loss_list.append(loss_value)
                
            draw.draw_loss(EPOCH, loss_list, 'loss')
            draw.draw_accu(EPOCH, accu_list, 'accu')
                    
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
            
        coord.join(threads)
        # sess.close()
    



def test():
    with tf.Graph().as_default() as graph:
        
        # reading test data
        images_rgb, images_depth, labels = process_input.get_batch(pre_images_rgb_tes, pre_images_depth_tes, pre_labels_tes,
                                              IMG_W, IMG_H, 
                                              BATCH_SIZE, CAPACITY, False)
    
        print(pre_images_rgb_tes[:5])
        print(pre_images_depth_tes[:5])
        # rgb_fc2 = Gao.Gao_rgb(images_rgb, N_CLASSES, IS_PRETRAIN, False)# output is rgb after conv2
        # depth_fc2 = Gao.Gao_depth(images_depth, N_CLASSES, IS_PRETRAIN, False)# output is depth agter conv2

        rgb_fc2 = heterogeneous.Gao_rgb(images_rgb, N_CLASSES, IS_PRETRAIN, False)# output is rgb after conv2
        depth_fc2 = heterogeneous.Gao_depth(images_depth, N_CLASSES, IS_PRETRAIN, False)# output is depth agter conv2

        prelogits = tf.concat([rgb_fc2, depth_fc2], 1, name='concat')

        sigmoid_alpha = 1 #Add a constant to enlarge the values input into
        prelogits = tf.nn.sigmoid(tf.multiply(prelogits,sigmoid_alpha))

        logits = slim.fully_connected(prelogits, N_CLASSES, activation_fn=None,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                    weights_regularizer=slim.l2_regularizer(0.0),
                    scope='Logits', reuse=False)
        
        global First_test
        #global Second_test
        if(First_test):
            parameter_GFLOPS.stats_graph(graph)
            First_test = False
            #Second_test = True

        top_k_op = tf.nn.in_top_k(logits, labels, 1)
        argmax = tf.argmax(logits,1)
        equal_bool = tf.equal(tf.cast(argmax,tf.int32),tf.cast(labels,tf.int32))
        saver = tf.train.Saver(tf.global_variables())
        
        with tf.Session() as sess:
            
            print("Reading %s checkpoints..." % LOG_DIR)
            ckpt = tf.train.get_checkpoint_state(LOG_DIR)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
                return
        
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess = sess, coord = coord)
            
            try:
                num_iter = int(math.ceil(N_TEST / BATCH_SIZE))
                true_count = 0
                act_count = 0
                total_sample_count = num_iter * BATCH_SIZE
                print(total_sample_count)
                step = 0
                
                while step < num_iter and not coord.should_stop():
                    predictions, get_bool, get_argmax, get_logits, get_labels, rgb, depth, get_prelogits, img_rgb, img_depth = sess.run([top_k_op, equal_bool, argmax, logits, labels, rgb_fc2, depth_fc2, prelogits, images_rgb, images_depth])

                    act_count += np.sum(get_bool)			
                    true_count += np.sum(predictions)
                    step += 1
                    true_precision = act_count / total_sample_count
                    precision = true_count / total_sample_count
		
                print('in_top_k')
                print(true_count)
                print('precision = %.3f' % precision)
                print('our_argmax')
                print(act_count)
                print('precision = %.3f' % true_precision)
                
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)
            return true_precision
    

def calculate_dis_train(rgb, depth, labels, U1, T1, U2, T2):

        bias_rgb1 = T1-U1
        bias_rgb2 = T1+U1
        bias_depth1 = T2-U2
        bias_depth2 = T2+U2


        dist_rgb = 0
        dist_depth = 0
        dist_corre = 0
   
        for i in range(BATCH_SIZE):
            for j in range(BATCH_SIZE):

                if i==0:
                    temp_corre = tf.sqrt(tf.reduce_sum(tf.square(rgb[j,:]-depth[j,:])))
                    dist_corre = tf.add(temp_corre, dist_corre)
                    
                if i!=j:
                    temp_rgb = tf.sqrt(tf.reduce_sum(tf.square(rgb[i,:]-rgb[j,:])))
                    temp_depth = tf.sqrt(tf.reduce_sum(tf.square(depth[i,:]-depth[j,:])))
                    def f1(x):
                        temp2_rgb = tf.add(x, bias_rgb1)
                        return tf.nn.relu(temp2_rgb)
                    def f2(x):
                        temp2_rgb = tf.subtract(bias_rgb2, x)
                        return tf.nn.relu(temp2_rgb)
                    def f3(x):
                        temp2_depth = tf.add(x, bias_depth1)
                        return tf.nn.relu(temp2_depth)
                    def f4(x):
                        temp2_depth = tf.subtract(bias_depth2, x)
                        return tf.nn.relu(temp2_depth)


                    temp1 = tf.cond(tf.equal(labels[i], labels[j]), lambda:f1(temp_rgb), lambda:f2(temp_rgb))
                    dist_rgb = tf.add(0.0, temp1) if i == 0 and j == 1 else tf.add(dist_rgb, temp1)
                    temp2 = tf.cond(tf.equal(labels[i], labels[j]), lambda:f3(temp_depth), lambda:f4(temp_depth))
                    dist_depth = tf.add(0.0, temp2) if i == 0 and j == 1 else tf.add(dist_depth, temp2)
                    # print("AT train.py 382 dist_rgb = {}".format(dist_rgb))
                    # print("AT train.py 383 dist_depth = {}".format(dist_depth))
                    # print("AT train.py 384 dist_corre = {}".format(dist_corre))

        temp1_batch = tf.Variable(float(BATCH_SIZE**2))
        temp2_batch = tf.Variable(float(BATCH_SIZE))
        #sess.run(temp1.initializer)
        #sess.run(temp2.initializer)

        dist_rgb = tf.divide(dist_rgb, temp1_batch)
        dist_depth = tf.divide(dist_depth, temp1_batch)
        dist_corre = tf.divide(dist_corre, temp2_batch)
        # tf.clip_by_value(dist_rgb, 1, 1e8)
        # tf.clip_by_value(dist_rgb, 1, 1e8)
        # tf.clip_by_value(dist_rgb, 1, 1e8)
        print("AT train.py 392 dist_rgb = {}".format(dist_rgb))
        print("AT train.py 393 dist_depth = {}".format(dist_depth))
        print("AT train.py 394 dist_corre = {}".format(dist_corre))
        return dist_rgb, dist_depth, dist_corre

        


def calculate_quan_loss(prelogits):

    return tf.cast(-tf.reduce_mean(tf.square(tf.subtract(prelogits,0.5))),tf.float32)

def calculate_distri_loss(prelogits):

    bin_mean = tf.reshape(tf.tile(tf.reduce_mean(prelogits,0),[BATCH_SIZE]),[BATCH_SIZE,DIM])

    return tf.cast(-tf.reduce_mean(tf.square(tf.subtract(prelogits, bin_mean))),tf.float32)

def calculate_dis_prelogits(prelogits, labels, U1, T1):
    
    dist = 0
    bias1 = T1-U1
    bias2 = T1+U1

    for i in range(BATCH_SIZE):
        for j in range(BATCH_SIZE):
            if i != j:
                temp = tf.sqrt(tf.reduce_sum(tf.square(prelogits[i,:]-prelogits[j,:]), 0))
                def f1(x):
                    temp2 = tf.add(x, bias1)
                    return tf.nn.relu(temp2)
                def f2(x):
                    temp2 = tf.subtract(bias2, x)
                    return tf.nn.relu(temp2)
                temp1 = tf.cond(tf.equal(labels[i], labels[j]), lambda:f1(temp), lambda:f2(temp))
                dist = tf.add(0.0, temp1) if i == 0 and j == 1 else tf.add(dist, temp1)
    dist = tf.divide(dist, tf.Variable(float(BATCH_SIZE**2)))
    return dist

train()
_ = test()
