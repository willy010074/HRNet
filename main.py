from model import HRCNN
#from utils import input_setup

#import numpy as np
import tensorflow as tf

import pprint
import os

flags = tf.app.flags
flags.DEFINE_integer("epoch", 1000, "Number of epoch")
flags.DEFINE_integer("batch_size", 1, "The size of batch images")

#flags.DEFINE_integer("image_H", 376, "The size of image to use (KITTI)")
#flags.DEFINE_integer("image_W", 1244, "The size of image to use (KITTI)")
flags.DEFINE_integer("image_H", 456, "The size of image to use (NYU depth V2)")
flags.DEFINE_integer("image_W", 608, "The size of image to use (NYU depth V2)")

flags.DEFINE_float("learning_rate", 1e-2, "The learning rate of gradient descent algorithm")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color.")
flags.DEFINE_integer("scale", 3, "The size of scale factor for preprocessing input image")
flags.DEFINE_integer("stride", 16, "The size of stride to apply input image")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory")
flags.DEFINE_string("sample_dir", "sample", "Name of sample directory")

#flags.DEFINE_string("dataset", "kitti_dataset", "Name of dataset (KITTI)")
#flags.DEFINE_string("dataset", "nyu_dataset", "Name of dataset(NYU depth V2)")
flags.DEFINE_string("dataset", "nyu2_dataset", "Name of dataset(NYU depth V2)")

flags.DEFINE_boolean("is_train", False, "True for training, False for testing")
FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

def main(_):
    pp.pprint(flags.FLAGS.__flags)
    
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    
    with tf.Session() as sess:
        hrcnn = HRCNN(sess, 
                      image_H=FLAGS.image_H, 
                      image_W=FLAGS.image_W, 
                      batch_size=FLAGS.batch_size,
                      c_dim=FLAGS.c_dim, 
                      checkpoint_dir=FLAGS.checkpoint_dir,
                      sample_dir=FLAGS.sample_dir)
        
        hrcnn.train(FLAGS)

if __name__ == '__main__':
    tf.app.run()
