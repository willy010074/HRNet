from utils import (
    imsave,
    prepare_data,
    read_image,
    merge,
    slic
)

import time
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input, Reshape, Convolution2D, MaxPooling2D

from tqdm import trange

class HRCNN(object):

    def __init__(self, 
                 sess, 
                 image_H=480,
                 image_W=640, 
                 batch_size=128,
                 c_dim=1, 
                 checkpoint_dir=None, 
                 sample_dir=None):
    
        self.sess = sess
        self.is_grayscale = (c_dim == 1)
        self.image_H = image_H
        self.image_W = image_W
        self.batch_size = batch_size
        
        self.c_dim = c_dim
        
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.build_model()
    
    def build_model(self):
        #self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images') # crop image
        #self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, 1], name='labels')
        
        self.images = tf.placeholder(tf.float32, [None, self.image_H, self.image_W, self.c_dim], name='images') # whole image
        self.labels = tf.placeholder(tf.float32, [None, self.image_H, self.image_W, 1], name='labels')
        self.slic_labels = tf.placeholder(tf.float32, [None, self.image_H, self.image_W, 1], name='slic_labels')
        
        self.weights = {
            'w16to16': tf.Variable(tf.random_normal([3, 3, 16, 16], stddev=1e-3), name='w16to16'),
            'w32to32': tf.Variable(tf.random_normal([3, 3, 32, 32], stddev=1e-3), name='w32to32'),
            'w64to64': tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=1e-3), name='w64to64'),
            'w16to64': tf.Variable(tf.random_normal([3, 3, 16, 64], stddev=1e-3), name='w16to64'),
            'w16to32': tf.Variable(tf.random_normal([3, 3, 16, 32], stddev=1e-3), name='w16to32')
        }
        self.biases = {
            'b1': tf.Variable(tf.zeros([64]), name='b1'),
            'b2': tf.Variable(tf.zeros([32]), name='b2'),
            'b3': tf.Variable(tf.zeros([1]), name='b3')
        }
        
        #self.pred = self.model()
        self.pred = self.short_cut_model()
        #self.pred = self.David_Eigen_model()
        #self.pred = self.test_model()
        
        # Loss function (MSE)
        #self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
        
        # scale-invariant mean squared error
        #self.loss = tf.reduce_mean(self.scale_invariant_MSE(self.pred, self.labels))
        
        # my Loss function
        #self.loss = self.my_loss(self.pred, self.labels)
        self.loss = self.my_loss(self.pred, self.labels, self.slic_labels, self.images)
        
        self.saver = tf.train.Saver()
    
    def train(self, config):
        self.train_op = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(self.loss)
        
        tf.initialize_all_variables().run()
        
        counter = 0
        start_time = time.time()
        
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        
        if config.is_train:
            print("Training...")
            
            # Load data path
            RGB_data, depth_data = prepare_data(self.sess, dataset=config.dataset)
            
            for ep in range(config.epoch):
                batch_counter = 0
                batch_images = []
                batch_labels = []
                batch_slic_labels = []
                for i in range(len(RGB_data)):
                    '''KITTI'''
#                    _, input_ = read_image("kitti_dataset\\resize_RGB\\" + RGB_data[i]) # RGB image
#                    _, label_ = read_image("kitti_dataset\\resize_depth\\" + depth_data[i], True) # Depth map. (Ground truth)
#                    _, slic_label_ = read_image("kitti_dataset\\resize_slic_depth\\" + RGB_data[i], True) # SLIC Depth map. (Ground truth)
                    
                    '''NYU depth V2'''
#                    _, input_ = read_image("nyu_dataset\\resize_RGB\\" + RGB_data[i]) # RGB image
#                    _, label_ = read_image("nyu_dataset\\resize_depth\\" + depth_data[i], True) # Depth map. (Ground truth)
#                    _, slic_label_ = read_image("nyu_dataset\\resize_slic_depth\\" + RGB_data[i], True) # SLIC Depth map. (Ground truth)
                    
                    '''NYU2 depth V2'''
                    _, input_ = read_image("nyu2_dataset\\resize_RGB\\" + RGB_data[i]) # RGB image
                    _, label_ = read_image("nyu2_dataset\\resize_depth\\" + depth_data[i], True) # Depth map. (Ground truth)
                    _, slic_label_ = read_image("nyu2_dataset\\resize_slic_depth\\" + RGB_data[i], True) # SLIC Depth map. (Ground truth)
                    
#                    label_ = np.zeros(input_.shape)
#                    label_[:,:,0] = depth_
#                    label_[:,:,1] = depth_
#                    label_[:,:,2] = depth_
                     
                    if len(input_.shape) == 3:
                        h, w, _ = input_.shape
                    else:
                        h, w = input_.shape
                    
                    # crop image
#                    for x in range(0, h-config.image_size+1, config.stride):
#                        for y in range(0, w-config.image_size+1, config.stride):
#                            sub_input = input_[x:x+config.image_size, y:y+config.image_size] # [33 x 33]
#                            sub_label = label_[x:x+config.image_size, y:y+config.image_size] # [33 x 33]
#                            
#                            # Make channel value
#                            #sub_input = sub_input.reshape([config.image_size, config.image_size, 1])  
#                            sub_label = sub_label.reshape([config.label_size, config.label_size, 1])
#                            
#                            batch_images.append(sub_input)
#                            batch_labels.append(sub_label)
#                            batch_counter += 1
#                            
#                            if batch_counter == config.batch_size :
#                                counter += 1
#                                _, err = self.sess.run([self.train_op, self.loss], feed_dict={self.images: batch_images, self.labels: batch_labels})
#                                
#                                if counter % 1000 == 0:
#                                    print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
#                                              % ((ep+1), counter, time.time()-start_time, err))
#                                
#                                if counter % 1000 == 0:
#                                    self.save(config.checkpoint_dir, counter)
#                                
#                                batch_images.clear()
#                                batch_labels.clear()
#                                batch_counter = 0
                    
                    # whole image
                    label_ = label_.reshape([self.image_H, self.image_W, 1])
                    slic_label_ = slic_label_.reshape([self.image_H, self.image_W, 1])
                    batch_images.append(input_)
                    batch_labels.append(label_)
                    batch_slic_labels.append(slic_label_)
                    batch_counter += 1
                    
                    if batch_counter == config.batch_size :
                        counter += 1
                        #_, err = self.sess.run([self.train_op, self.loss], feed_dict={self.images: batch_images, self.labels: batch_labels})
                        _, err = self.sess.run([self.train_op, self.loss], feed_dict={self.images: batch_images, self.labels: batch_labels, self.slic_labels: batch_slic_labels})
                        
                        if counter % 10 == 0:
                            print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
                                      % ((ep+1), counter, time.time()-start_time, err))
                        
                        if counter % 10 == 0:
                            self.save(config.checkpoint_dir, counter)
                        
                        batch_images.clear()
                        batch_labels.clear()
                        batch_slic_labels.clear()
                        batch_counter = 0
        
        else:
            print("Testing...")
            RGB_data, depth_data = prepare_data(self.sess, dataset=config.dataset)
            
            for i in trange(len(RGB_data)):
                start_time = time.time()
                '''KITTI'''
#                _, input_ = read_image("kitti_dataset\\resize_RGB\\" + RGB_data[i]) # RGB image
#                _, label_ = read_image("kitti_dataset\\resize_depth\\" + depth_data[i], True) # Depth map. (Ground truth)
                
                '''NYU depth V2'''
#                _, input_ = read_image("nyu_dataset\\resize_RGB\\" + RGB_data[i]) # After mapping. (coarse depth map)
#                _, label_ = read_image("nyu_dataset\\resize_depth\\" + depth_data[i], True) # Depth map. (Ground truth)
                
                '''NYU2 depth V2'''
                _, input_ = read_image("nyu2_dataset\\resize_RGB\\" + RGB_data[i]) # After mapping. (coarse depth map)
                _, label_ = read_image("nyu2_dataset\\resize_depth\\" + depth_data[i], True) # Depth map. (Ground truth)
                 
                if len(input_.shape) == 3:
                    h, w, _ = input_.shape
                else:
                    h, w = input_.shape
                
                #nx = ny = 0 
                batch_images = []
                batch_labels = []
                #batch_slic_labels = []
                
                label_ = label_.reshape([self.image_H, self.image_W, 1])
                #slic_label_ = slic_label_.reshape([self.image_H, self.image_W, 1])
                batch_images.append(input_)
                batch_labels.append(label_)
                #batch_slic_labels.append(slic_label_)
                
                # crop image
#                for x in range(0, h-config.image_size+1, config.stride):
#                    nx += 1; ny = 0
#                    for y in range(0, w-config.image_size+1, config.stride):
#                        ny += 1
#                        sub_input = input_[x:x+config.image_size, y:y+config.image_size] # [33 x 33]
#                        sub_label = label_[x:x+config.image_size, y:y+config.image_size] # [33 x 33]
#                        
#                        # Make channel value
#                        #sub_input = sub_input.reshape([config.image_size, config.image_size, 1])  
#                        sub_label = sub_label.reshape([config.label_size, config.label_size, 1])
#                        
#                        batch_images.append(sub_input)
#                        batch_labels.append(sub_label)
                        
                result = self.pred.eval({self.images: batch_images, self.labels: batch_labels})
                #result = self.pred.eval({self.images: batch_images, self.labels: batch_labels, self.slic_labels: batch_slic_labels})
                
                # crop image
                #result = merge(config, result, [nx, ny])
                result = result.squeeze()
                stop_time = time.time()
                print(stop_time-start_time)
                
                image_path = os.path.join(os.getcwd(), config.sample_dir)
                image_path = os.path.join(image_path, "result_" + RGB_data[i].split("\\")[-1])
                imsave(result, image_path)
    
#    def scale_invariant_MSE(self, pred, label):
#        first_log = tf.keras.backend.log(tf.keras.backend.clip(pred, tf.keras.backend.epsilon(), np.inf) + 1.)
#        second_log = tf.keras.backend.log(tf.keras.backend.clip(label, tf.keras.backend.epsilon(), np.inf) + 1.)
#        
#        return tf.keras.backend.mean(tf.keras.backend.square(first_log - second_log), axis=-1) - (0.5 * tf.keras.backend.square(tf.keras.backend.mean(first_log - second_log, axis=-1)))
    
    def my_loss(self, pred, label, slic_label, image):
        w1 = 0.5 # MSE
        w2 = 0.5 # SLIC
        w3 = 2.0 # Gradient
        w4 = 0.5 # SSIM
        w5 = 1.0 # smooth

        d = pred - label        
        
        slic_loss = tf.reduce_mean(tf.square(pred - slic_label))
        ssim_loss = 1 - tf.image.ssim(pred,label,255)
        #total = first_val + second_val    
        
        mse_loss = tf.reduce_mean(tf.square(d)) 
        #scale_invariant_error = tf.reduce_mean(tf.square(d))- (l/tf.square(self.image_H*self.image_W)*tf.square(d))
        
        image_grad_y, image_grad_x = tf.image.image_gradients(image)
        gt_grad_y, gt_grad_x = tf.image.image_gradients(label)
        pred_grad_y, pred_grad_x = tf.image.image_gradients(pred)
        
        gradient_loss = tf.reduce_mean(tf.abs(gt_grad_x - pred_grad_x) + tf.abs(gt_grad_y - pred_grad_y))
        
        smooth_loss = tf.reduce_mean(tf.abs(pred_grad_x)*tf.exp(tf.abs(image_grad_x)*(-1)) + tf.abs(pred_grad_y)*tf.exp(tf.abs(image_grad_y)*(-1)))

        #return mse_loss*w1 + ssim_loss*w2 + gradient_loss*w3
        #return (mse_loss*w1 + slic_loss*w2) * ssim_loss*w4 + gradient_loss*w3
        return mse_loss*w1 + slic_loss*w2 + gradient_loss*w3 + ssim_loss*w4 + smooth_loss*w5
        #return 0.1*point_difference + (ssim/2)
    
    def filter_genarator(self, kernel_size, in_channel, out_channel):
        return tf.Variable(tf.random_normal([kernel_size,kernel_size,in_channel,out_channel], stddev=1e-3))
    
    def conv_genarator(self, input_feature, filter_, strides):
        train_node = tf.placeholder_with_default(True, ())
        net = tf.nn.conv2d(input_feature, filter_, strides=[1,strides,strides,1], padding='SAME')
        net = tf.layers.batch_normalization(net, training=train_node)
        
        return tf.nn.relu(net)
    
#    def atrous_conv_genarator(self, input_feature, filter_, rate):
#        train_node = tf.placeholder_with_default(True, ())
#        net = tf.nn.atrous_conv2d(input_feature, filter_, rate, padding="SAME")
#        net = tf.layers.batch_normalization(net, training=train_node)
#        
#        return tf.nn.relu(net)
    
    def upsampling(self, input_feature, new_size):
        size = [new_size,new_size]
        
        return tf.keras.layers.UpSampling2D(size, interpolation='bilinear')(input_feature)
    
#    def maxpooling(self, input_feature, size, strides):        
#        return tf.keras.layers.MaxPool2D(size, strides, padding='SAME')(input_feature)
    
    def conv_block(self, input_feature, filter_):
        conv1 = self.conv_genarator(input_feature, filter_, 1) + input_feature
        conv2 = self.conv_genarator(conv1, filter_, 1) + input_feature + conv1
        conv3 = self.conv_genarator(conv2, filter_, 1) + input_feature + conv1 + conv2
        conv4 = self.conv_genarator(conv3, filter_, 1) + input_feature + conv1 + conv2 + conv3
        
        return conv4

    def short_cut_model(self):
        '''
            first part
        '''
        conv1_input = self.conv_genarator(self.images, self.filter_genarator(1,3,16), 1) # input layer
        conv1 = self.conv_block(conv1_input, self.filter_genarator(3,16,16)) # 1-1 block 1
        conv1 = conv1 + conv1_input # shortcut
        temp1 = conv1
        conv1 = self.conv_block(conv1, self.filter_genarator(3,16,16)) # 1-1 block 2
        conv1 = conv1 + temp1 # shortcut
        
        '''
            second part
        '''
        conv2_1_input = self.conv_genarator(conv1, self.filter_genarator(1,16,16), 1) # 2-1 input layer
        conv2_1_out = self.conv_block(conv2_1_input, self.filter_genarator(3,16,16)) # 2-1 block 1
        conv2_1_out = conv2_1_out + conv2_1_input # shortcut
        temp2_1 = conv2_1_out
        conv2_1_out = self.conv_block(conv2_1_out, self.filter_genarator(3,16,16)) # 2-1 block 2
        conv2_1_out = conv2_1_out + temp2_1 # shortcut
        conv2_1_out_downsample1 = self.conv_genarator(conv2_1_out, self.filter_genarator(3,16,32), 2)
        conv2_1_out_downsample2 = self.conv_genarator(conv2_1_out, self.filter_genarator(3,16,64), 2)
        conv2_1_out_downsample2 = self.conv_genarator(conv2_1_out_downsample2, self.filter_genarator(3,16,64), 2)

        conv2_2 = self.conv_genarator(conv1, self.filter_genarator(1,16,32), 2) # 2-2 input layer
        conv2_2_out = self.conv_block(conv2_2, self.filter_genarator(3,32,32)) #2-2 block 1
        conv2_2_out = conv2_2_out + conv2_2 # shortcut
        temp2_2 = conv2_2_out
        conv2_2_out = self.conv_block(conv2_2_out, self.filter_genarator(3,32,32)) #2-2 block 2
        conv2_2_out = conv2_2_out + temp2_2
        conv2_2_out_upsample = self.conv_genarator(conv2_2_out, self.filter_genarator(1,32,16), 1)
        conv2_2_out_upsample = self.upsampling(conv2_2_out_upsample, 2)
        conv2_2_out_upsample = self.conv_genarator(conv2_2_out_upsample, self.filter_genarator(3,16,16), 1) # 3x3 conv for upsample
        conv2_2_out_upsample = tf.nn.dropout(conv2_2_out_upsample, rate = 0.5) # dropout
        conv2_2_out_downsample = self.conv_genarator(conv2_2_out, self.filter_genarator(3,32,64), 2)
        
        '''
            third part
        '''
        conv3_1 = tf.concat([conv2_1_out, conv2_2_out_upsample], -1) # concat 2-1 and 2-2
        conv3_1_input = self.conv_genarator(conv3_1, self.filter_genarator(1,32,16), 1) # 3-1 input layer
        conv3_1_out = self.conv_block(conv3_1_input, self.filter_genarator(3,16,16)) #3-1 block 1
        conv3_1_out = conv3_1_out + conv3_1_input # shortcut
        temp3_1 = conv3_1_out
        conv3_1_out = self.conv_block(conv3_1_out, self.filter_genarator(3,16,16)) #3-1 block 2
        conv3_1_out = conv3_1_out + temp3_1 # shortcut
        
        conv3_2 = tf.concat([conv2_1_out_downsample1,conv2_2_out], -1) # concat 2-1 and 2-2
        conv3_2 = self.conv_genarator(conv3_2, self.filter_genarator(1,64,32), 1) # 3-2 input layer
        conv3_2_out = self.conv_block(conv3_2, self.filter_genarator(3,32,32))# 3-2 block 1
        conv3_2_out = conv3_2_out + conv3_2 # shortcut
        temp3_2 = conv3_2_out
        conv3_2_out = self.conv_block(conv3_2_out, self.filter_genarator(3,32,32))# 3-2 block 2
        conv3_2_out = conv3_2_out + temp3_2 # shortcut
        conv3_2_out_upsample = self.conv_genarator(conv3_2_out, self.filter_genarator(1,32,16), 1)
        conv3_2_out_upsample = self.upsampling(conv3_2_out_upsample, 2)
        conv3_2_out_upsample = self.conv_genarator(conv3_2_out_upsample, self.filter_genarator(3,16,16), 1) # 3x3 conv for upsample
        conv3_2_out_upsample = tf.nn.dropout(conv3_2_out_upsample, rate = 0.5) # dropout
        
        conv3_3 = tf.concat([conv2_1_out_downsample2, conv2_2_out_downsample], -1) # concat 2-1 and 2-2
        conv3_3 = self.conv_genarator(conv3_3, self.filter_genarator(1,128,64), 1) # 3-3 input layer
        conv3_3_out = self.conv_block(conv3_3, self.filter_genarator(3,64,64)) # 3-3 block 1
        conv3_3_out = conv3_3_out + conv3_3 # shortcut
        temp3_3 = conv3_3_out
        conv3_3_out = self.conv_block(conv3_3_out, self.filter_genarator(3,64,64)) # 3-3 block 2
        conv3_3_out = conv3_3_out + temp3_3 # shortcut
        conv3_3_out_upsample2 = self.conv_genarator(conv3_3_out, self.filter_genarator(1,64,16), 1)
        conv3_3_out_upsample2 = self.upsampling(conv3_3_out_upsample2, 4)
        conv3_3_out_upsample2 = self.conv_genarator(conv3_3_out_upsample2, self.filter_genarator(3,16,16), 1) # 3x3 conv for upsample
        conv3_3_out_upsample2 = tf.nn.dropout(conv3_3_out_upsample2, rate = 0.5) # dropout
        
        '''
            output
        '''
        fin_conv = tf.concat([conv3_1_out, conv3_2_out_upsample], -1)
        fin_conv = tf.concat([fin_conv, conv3_3_out_upsample2], -1)
        fin_conv = self.conv_genarator(fin_conv, self.filter_genarator(1,48,1), 1)
        
        return fin_conv
    
    def save(self, checkpoint_dir, step):
        model_name = "SRCNN.model"
        model_dir = "hrcnn"
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)
    
    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        model_dir = "hrcnn"
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
