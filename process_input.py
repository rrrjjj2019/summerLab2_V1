import tensorflow as tf
import numpy as np
import os
import math


# In this file
# There are two functions:
# pre_process(file_dir) and get_batch(image, label, image_W, image_H, batch_size, capacity)



### label image and return
def pre_process(data_dir,ratio):
    
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
	
    training_list = []
    training_list_RGB = []
    training_list_DEPTH = []

    testing_list = []
    testing_list_RGB = []
    testing_list_DEPTH = []
	
    #get path of training file
    training_rgb_dir = data_dir+'training/RGB_training'
    training_depth_dir = data_dir+'training/DEPTH_training'
    for dir_withlabel in sorted(os.listdir(training_rgb_dir)):
        rgb_img_dir = training_rgb_dir + '/' + dir_withlabel
        depth_img_dir = training_depth_dir + '/' + dir_withlabel
        for image_name in sorted(os.listdir(rgb_img_dir)):
            #temp = (rgb_img_dir+'/'+image_name, depth_img_dir+'/'+image_name, int(dir_withlabel))
            temp = rgb_img_dir+'/'+image_name
            training_list_RGB.append(temp)
        for image_name in sorted(os.listdir(depth_img_dir)):
            #temp = (rgb_img_dir+'/'+image_name, depth_img_dir+'/'+image_name, int(dir_withlabel))
            temp = depth_img_dir+'/'+image_name
            training_list_DEPTH.append(temp)
        for i in range(len(training_list_RGB)):
            #print(len(training_list_RGB), end = ", ")
            #print(len(training_list_DEPTH))
            temp = (training_list_RGB[i], training_list_DEPTH[i], int(dir_withlabel))
            # print("AT process_input 47 temp = " , end = "")
            # print(temp)
            training_list.append(temp)
        training_list_RGB.clear()
        training_list_DEPTH.clear()
            
    
    #get path of testing file
    testing_rgb_dir = data_dir+'testing/RGB_testing'
    testing_depth_dir = data_dir+'testing/DEPTH_testing'
    for dir_withlabel in sorted(os.listdir(testing_rgb_dir)):
        rgb_img_dir = testing_rgb_dir + '/' + dir_withlabel
        depth_img_dir = testing_depth_dir + '/' + dir_withlabel
        for image_name in sorted(os.listdir(rgb_img_dir)):
            #temp = (rgb_img_dir+'/'+image_name, depth_img_dir+'/'+image_name, int(dir_withlabel))
            temp = rgb_img_dir+'/'+image_name
            testing_list_RGB.append(temp)
        for image_name in sorted(os.listdir(depth_img_dir)):
            #temp = (rgb_img_dir+'/'+image_name, depth_img_dir+'/'+image_name, int(dir_withlabel))
            temp = depth_img_dir+'/'+image_name
            testing_list_DEPTH.append(temp)
        for i in range(len(testing_list_RGB)):
            print(len(testing_list_RGB), end = ", ")
            print(len(testing_list_DEPTH), end = ", ")
            print(int(dir_withlabel))
            temp = (testing_list_RGB[i], testing_list_DEPTH[i], int(dir_withlabel))
            testing_list.append(temp)
        testing_list_RGB.clear()
        testing_list_DEPTH.clear()
			
    
    np.random.seed(6)
    np.random.shuffle(training_list)
    np.random.shuffle(testing_list)
	
    tra_rgb = []
    tra_dep = []
    tra_label = []

    for i in range(len(training_list)):
        get_tuple = training_list[i]
        tra_rgb.append(get_tuple[0])
        tra_dep.append(get_tuple[1])
        tra_label.append(get_tuple[2])
        
    tes_rgb = []
    tes_dep = []
    tes_label = []

    for i in range(len(testing_list)):
        get_tuple = testing_list[i]
        tes_rgb.append(get_tuple[0])
        tes_dep.append(get_tuple[1])
        tes_label.append(get_tuple[2])

    print('There are %d train and %d test' % (len(training_list), len(testing_list)))    
        
    return tra_rgb, tra_dep, tra_label, tes_rgb, tes_dep, tes_label, len(training_list), len(testing_list)
    
### get batch with tensor type
def get_batch(image_rgb, image_depth, label, image_W, image_H, batch_size, capacity, training = True):
    '''
    Args:
        image_rgb: list type
        image_depth: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
        
    Returns:
        
        image_batch_rgb: 4D tensor [batch_size, width, height, channels], dtype= tf.float32
        image_batch_depth: 4D tensor [batch_size, width, height, channels], dtype= tf.float32
        label_batch: 1D_tensor [batch_size], dtype=tf.int32
        
    '''
    
    ###############################################################
    #process RGB image
    image_rgb = tf.cast(image_rgb, tf.string)
    label_rgb = tf.cast(label, tf.int32)
    
    # make an input queue
    input_queue_rgb = tf.train.slice_input_producer([image_rgb, label_rgb], shuffle = False)
    
    label_rgb = input_queue_rgb[1]
    image_contents_rgb = tf.read_file(input_queue_rgb[0])
  
    image_rgb = tf.image.decode_jpeg(image_contents_rgb, channels=3)
    
    ######################
    # data argumentation should go to here
    ######################
    image_rgb = tf.image.resize_image_with_crop_or_pad(image_rgb, image_W, image_H)
    image_rgb_2 = image_rgb
    image_rgb = tf.image.per_image_standardization(image_rgb)
    print('image_rgb')
    print(image_rgb)
    ################################################################
    
    ###############################################################
    #process Depth image
    image_depth = tf.cast(image_depth, tf.string)
    label_depth = tf.cast(label, tf.int32)
    
    # make an input queue
    input_queue_depth = tf.train.slice_input_producer([image_depth, label_depth], shuffle = False)
    
    label_depth = input_queue_depth[1]
    image_contents_depth = tf.read_file(input_queue_depth[0])
  
    image_depth = tf.image.decode_jpeg(image_contents_depth, channels=1)
    
    ######################
    # data argumentation should go to here
    ######################
    image_depth = tf.image.resize_image_with_crop_or_pad(image_depth, image_W, image_H)
    image_depth_2 = image_depth
    image_depth = tf.image.per_image_standardization(image_depth)
    ################################################################
    
    #image = tf.concat([image_rgb, image_depth], 2, name='concat')
    #print(image)
    
    image_batch_rgb, label_batch_rgb, image_batch_rgb2 = tf.train.batch([image_rgb, label_rgb, image_rgb_2],
                                                  batch_size = batch_size,
                                                  num_threads = 1,
                                                  capacity = capacity)
    
    #
    image_batch_depth, label_batch_depth, image_batch_depth2 = tf.train.batch([image_depth, label_depth, image_depth_2],
                                                  batch_size = batch_size,
                                                  num_threads = 1,
                                                  capacity = capacity)
                                                  
    label_batch_rgb = tf.reshape(label_batch_rgb, [batch_size])
    image_batch_rgb = tf.cast(image_batch_rgb, tf.float32)
    image_batch_rgb2 = tf.cast(image_batch_rgb2, tf.float32)
	
    label_batch_depth = tf.reshape(label_batch_depth, [batch_size])
    image_batch_depth = tf.cast(image_batch_depth, tf.float32)
    image_batch_depth2 = tf.cast(image_batch_depth2, tf.float32)
    if training:
        return image_batch_rgb, image_batch_depth, label_batch_rgb, image_batch_rgb2, image_batch_depth2
    else:
        return image_batch_rgb, image_batch_depth, label_batch_rgb 

'''
for debug
if __name__ == '__main__':
    DATA_DIR = '/home/ubuntu/Hsuan/dataset_sep/'
    pre_images_rgb_tra, pre_images_depth_tra, pre_labels_tra, pre_images_rgb_tes, pre_images_depth_tes, pre_labels_tes, N_TRAIN, N_TEST = pre_process(DATA_DIR,0.3)
    images_rgb, images_depth, labels, temp_rgb_2 = get_batch(pre_images_rgb_tra, pre_images_depth_tra, pre_labels_tra,
                                              128, 128, 
                                              30, 2000)
    sess = tf.Session()
    with sess.as_default():
        print(temp_rgb_2)
        
        a = sess.run(temp_rgb_2[0])
        img = Image.fromarry(a)
        img.save('xx.png')
'''      