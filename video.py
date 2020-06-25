import numpy as np
import time
import os.path
import scipy.misc
import tensorflow as tf
from distutils.version import LooseVersion
from glob import glob
import warnings
from moviepy.editor import VideoFileClip

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to inference on your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def predict_video(sess, image_shape, logits, keep_prob, input_image):
    video_dir = "./sample/"
    video_library =   [["GOPR0706_cut1.mp4", [210, 470]]]
    for video_data in video_library:
        rect = video_data[1]
        video_output = video_data[0][:-4] +"_out.mp4"
        clip1 = VideoFileClip(video_dir + video_data[0])
        video_clip = clip1.fl_image(lambda frame: predict_frame(frame, rect, sess, image_shape, logits, keep_prob, input_image))
        video_clip.write_videofile(video_output, audio=False)


def predict_frame(im, rect, sess, image_shape, logits, keep_prob, image_pl):
    original = im.copy()
    roi = im[rect[0]:rect[1],0:720]

    image = scipy.misc.imresize(roi, image_shape)

    im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
    im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im = scipy.misc.toimage(image)
    street_im.paste(mask, box=None, mask=mask)
    upscale_pred = scipy.misc.imresize(street_im, (rect[1]-rect[0],720))
    original[rect[0]:rect[1], 0:720] = upscale_pred
    return original

def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    model_dir = './trained_model/'
    output_dir = './runs'
    #tests.test_for_kitti_dataset(data_dir)

    with tf.Session() as sess:
        # Download pretrained vgg model
        # helper.maybe_download_pretrained_vgg(data_dir)

        saver = tf.train.import_meta_graph('./trained_model/Semantic_seg_trained.ckpt.meta')
        print("Graph imported...")
        saver.restore(sess,tf.train.latest_checkpoint('./trained_model'))
        print("Model restored successfully!")

        # extract tensors, including input tensor and hyperparameter tensor.
        graph = tf.get_default_graph()
        input_image = graph.get_tensor_by_name('image_input:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        logits = graph.get_tensor_by_name('logits:0')  # note that here we call the TENSOR as the result of operation, not the operation itself. Call it by operation_name::x.

        print('Inferencing on video')
        last_time = time.time()
        predict_video(sess, image_shape, logits, keep_prob, input_image)
        print("Inference Time: {} seconds".format(time.time() - last_time))

if __name__ == '__main__':
    run()
