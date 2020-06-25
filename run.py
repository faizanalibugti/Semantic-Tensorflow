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
import mss

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to inference on your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def gen_test_output(sess, logits, keep_prob, image_pl, screen, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep probability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the image
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    image = scipy.misc.imresize(screen, image_shape)
    im_softmax = sess.run([tf.nn.softmax(logits)],{keep_prob: 1.0, image_pl: [image]})
    im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im = scipy.misc.toimage(image)
    street_im.paste(mask, box=None, mask=mask)
    street_im = np.array(street_im)

    return street_im

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
            monitor = {"top": 0, "left": 0, "width": 640, "height": 480}

            while "Screen capturing":
                last_time = time.time()

                # Get raw pixels from the screen, save it to a Numpy array
                screen = np.array(sct.grab(monitor))
                screen = np.flip(screen[:, :, :3], 2)
                screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

                image_output = gen_test_output(sess, logits, keep_prob, input_image, screen, image_shape)

                print("fps: {}".format(1 / (time.time() - last_time)))

                cv2.imshow('Screen Capture', screen)
                cv2.imshow('Semnatic Segmentation', image_output)

                if cv2.waitKey(25) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break

if __name__ == '__main__':
    run()
