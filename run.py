import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import cv2
import scipy.misc
import numpy as np
import time

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    model_dir = './trained_model/'
    output_dir = './runs'
    #tests.test_for_kitti_dataset(data_dir)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./trained_model/Semantic_seg_trained.ckpt.meta')
        print("Graph imported...")
        saver.restore(sess,tf.train.latest_checkpoint('./trained_model'))
        print("Model restored successfully!")

        # extract tensors, including input tensor and hyperparameter tensor.
        graph = tf.get_default_graph()
        input_image = graph.get_tensor_by_name('image_input:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        logits = graph.get_tensor_by_name('logits:0')  # note that here we call the TENSOR as the result of operation, not the operation itself. Call it by operation_name::x.

        with mss.mss() as sct:
            # Part of the screen to capture
            monitor = {"top": 200, "left": 100, "width": 640, "height": 480}

            while "Screen capturing":
                last_time = time.time()

                # Get raw pixels from the screen, save it to a Numpy array
                screen = numpy.array(sct.grab(monitor))
                screen = numpy.flip(screen[:, :, :3], 2)
                screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

                


