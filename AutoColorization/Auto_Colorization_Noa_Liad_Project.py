import os
import random
import skimage
import scipy.io
import datetime
import numpy as np
from PIL import Image
from skimage import io
import tensorflow as tf
import skimage.transform
from skimage import io, color
import BatchDatasetReader as dataset

# Paths
Desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
project_path = Desktop + '\\Auto_Colorization_Noa_Liad\\'
train_folder_path = project_path + 'train\\'
test_folder_path = train_folder_path + 'test\\'
model_folder_path = project_path + 'model\\'
images_folder_path = project_path + 'images\\'
results_folder_path = project_path + 'results\\'
images_gray_folder_path = project_path + 'Gray\\'
# for downloading the weights:
# http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-16.mat
weights_vgg16 = project_path + '\\weights\\weights-vgg-16.mat'

# Global Variable
image_dim = 224
test_precentage = 0.3
validation_precentage = 0
test_images_num = 0
batch_size = 2
learning_rate = 0.0001
MAX_ITERATION = 1000
adam_opt_beta1 = 0.9
restore_model = True
plot_train_loss = False
mode = 'test' #can be train
info_hypercolumn_data = False

def Gray_Datasets():
    if not os.path.exists(images_gray_folder_path):
        os.mkdir(images_gray_folder_path)
    dtype = np.float64
    # running on the images' folder
    for img in os.listdir(images_folder_path):
        if os.path.isdir(images_folder_path + '/' + img) or ".DS_Store" in str(img):
            continue
        path_image = images_folder_path + str(img)
        path_gray = images_gray_folder_path + str(img).split('.')[0] + "_gray." + str(img).split('.')[1]
        im = skimage.io.imread(path_image)
        # normalize
        if dtype == np.uint8:
            im = im
        elif dtype in {np.float16, np.float32, np.float64}:
            im = im.astype(dtype) / 255
        else:
            raise ValueError('Unsupported dtype')
        # Turning to Gray

        r, g, b = im.shape

        # calculate L = (r+g+b/3)
        gray = np.zeros(shape=(r, g))
        im[:] = im.mean(axis=-1, keepdims=1)
        #for ind1, val1 in enumerate(im):
        #    for ind2, val2 in enumerate(val1):
        #        gray[ind1][ind2] = val2[0]
        # print(im[0][0][0])
        if gray.dtype == np.uint8:
            pil_im = Image.fromarray(im)
        else:
            pil_im = Image.fromarray((im * 255).astype(np.uint8))
        # Save the new image
        pil_im.save(path_gray)

def vgg16_architecture(weights, image):  # load the pre-trained VGG16
    """
        VGG-16 Description:
        'conv1_1', 'relu1_1',  # 224*224*1 (originally 224*224*3)
        'conv1_2', 'relu1_2',  # 224*224*64
        'pool1',  # Max_Pooling
        'conv2_1', 'relu2_1',  # 112*112*128
        'conv2_2', 'relu2_2',  # 112*112*128
        'pool2',  # Max_Pooling
        'conv3_1', 'relu3_1',  # 56*56*256
        'conv3_2', 'relu3_2',  # 56*56*256
        'conv3_3', 'relu3_3',  # 56*56*256
        'pool3',  # Max_Pooling
        'conv4_1', 'relu4_1',  # 28*28*512
        'conv4_2', 'relu4_2',  # 28*28*512
        'conv4_3', 'relu4_3',  # 28*28*512
        'pool4',  # Max_Pooling
        'conv5_1', 'relu5_1',  # 14*14*512
        'conv5_2', 'relu5_2',  # 14*14*512
        'conv5_3', 'relu5_3',  # 14*14*512
    """
    layers = (
        'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3',
    )
    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i + 2][0][0][0][0]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = get_variable(bias.reshape(-1), name=name + "_b")
            current = tf.nn.bias_add(tf.nn.conv2d(current, kernels, strides=[1, 1, 1, 1], padding="SAME"), bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
        elif kind == 'pool':
            tf.nn.max_pool(current, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            #current = avg_pool_2x2(current)
        net[name] = current
    return net

def VGG16_net_HyperColumns(image, train_phase):
    model_data = scipy.io.loadmat(weights_vgg16)
    weights = np.squeeze(model_data['layers'])
    tf.placeholder(tf.float32, shape=[None, None, None, 1], name='L_images')
    with tf.variable_scope("HyperColumns") as scope:
        # VGG takes in 3channel (RGB) images.
        # In order to input 1-channel (gray) image,
        # define a new filter that takes in gray color image and map it into 64 channels so as to fit VGG conv1_2
        W0 = weight_variable([image_dim, image_dim, 1, 64], name="W0")
        b0 = bias_variable([64], name="b0")
        conv0 = tf.nn.bias_add(tf.nn.conv2d(image, W0, strides=[1, 1, 1, 1], padding="SAME"), b0)
        #conv0 = conv2d_basic(image, W0, b0)
        hrelu0 = tf.nn.relu(conv0, name="relu")
        image_net = vgg16_architecture(weights, hrelu0)

        relu1_2 = image_net["relu1_2"]
        layer_relu1_2 = tf.image.resize_bilinear(relu1_2, (image_dim, image_dim))
        relu2_1 = image_net["relu2_1"]
        layer_relu2_1 = tf.image.resize_bilinear(relu2_1, (image_dim, image_dim))
        relu2_2 = image_net["relu2_2"]
        layer_relu2_2 = tf.image.resize_bilinear(relu2_2, (image_dim, image_dim))
        relu3_1 = image_net["relu3_1"]
        layer_relu3_1 = tf.image.resize_bilinear(relu3_1, (image_dim, image_dim))
        relu3_2 = image_net["relu3_2"]
        layer_relu3_2 = tf.image.resize_bilinear(relu3_2, (image_dim, image_dim))
        relu3_3 = image_net["relu3_3"]
        layer_relu3_3 = tf.image.resize_bilinear(relu3_3, (image_dim, image_dim))
        relu4_1 = image_net["relu4_1"]
        layer_relu4_1 = tf.image.resize_bilinear(relu4_1, (image_dim, image_dim))
        relu4_2 = image_net["relu4_2"]
        layer_relu4_2 = tf.image.resize_bilinear(relu4_2, (image_dim, image_dim))
        relu4_3 = image_net["relu4_3"]
        layer_relu4_3 = tf.image.resize_bilinear(relu4_3, (image_dim, image_dim))
        relu5_1 = image_net["relu5_1"]
        layer_relu5_1 = tf.image.resize_bilinear(relu5_1, (image_dim, image_dim))
        relu5_2 = image_net["relu5_2"]
        layer_relu5_2 = tf.image.resize_bilinear(relu5_2, (image_dim, image_dim))
        relu5_3 = image_net["relu5_3"]
        layer_relu5_3 = tf.image.resize_bilinear(relu5_3, (image_dim, image_dim))
        # dense layer
        HyperColumns = tf.concat([layer_relu1_2, \
                                  layer_relu2_1, layer_relu2_2, \
                                  layer_relu3_1, layer_relu3_2, layer_relu3_3, \
                                  layer_relu4_1, layer_relu4_2, layer_relu4_3, \
                                  layer_relu5_1, layer_relu5_2, layer_relu5_3], 3)
        wc1 = weight_variable([1, 1, 4160, 2], name="wc1")
        wc1_biase = bias_variable([2], name="wc1_biase")
        pred_AB_conv = tf.nn.conv2d(HyperColumns, wc1, [1, 1, 1, 1], padding='SAME')
        pred_AB = tf.nn.bias_add(pred_AB_conv, wc1_biase)
    return tf.concat(values=[image, pred_AB], axis=3, name="pred_image")

def train(loss, var_list):
    global adam_opt_beta1
    #Using Adam Optimzer
    optimizer = tf.train.AdamOptimizer(learning_rate, adam_opt_beta1)
    grads = optimizer.compute_gradients(loss, var_list=var_list)
    #Gradients Summary
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + "/gradient", grad)
    return optimizer.apply_gradients(grads)

def weight_variable(shape, stddev=0.02, name=None):
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)

def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)

def save_image(image, save_dir, name):
    image = color.lab2rgb(image)
    io.imsave(os.path.join(save_dir, name + ".png"), image)

def get_variable(weights, name):
    init = tf.constant_initializer(weights, dtype=tf.float32)
    var = tf.get_variable(name=name, initializer=init, shape=weights.shape)
    return var

def read_dataset():
    global test_images_num
    result = divide_to_datasets(images_folder_path, testing_percentage=test_precentage,validation_percentage=validation_precentage)
    training_images = result['train']
    testing_images = result['test']
    validation_images = result['validation']
    test_images_num = len(testing_images)
    print("Training: %d, Validation: %d, Test: %d" % (len(training_images), len(validation_images), len(testing_images)))
    return training_images, testing_images, validation_images

def divide_to_datasets(image_dir, testing_percentage, validation_percentage):
    training_images = []
    print("Looking for images in '" + image_dir + "'")
    if len(os.listdir(image_dir)) == 0:
        print('No Images found')
    else:
        training_images.extend([file for file in os.listdir(image_dir)])
        #training_images.remove('.DS_Store')
        random.shuffle(training_images)
        num_of_images = len(training_images)
        validation_offset = int(validation_percentage * num_of_images)
        validation_images = training_images[:validation_offset]
        test_offset = int(testing_percentage * num_of_images)
        testing_images = training_images[validation_offset:validation_offset + test_offset]
        training_images = training_images[validation_offset + test_offset:]
        result = {'train': training_images, 'test': testing_images, 'validation': validation_images}
    return result

def main():
    global learning_rate
    global plot_train_loss
    global restore_model
    global test_precentage
    global validation_precentage

    print("Changing the images to grayscale")
    Gray_Datasets()
    print("Creating the network...")
    train_phase = tf.placeholder(tf.bool, name="train_phase")
    images = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='L_images')
    lab_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="LAB_images")
    pred_image = VGG16_net_HyperColumns(images, train_phase)
    gen_loss_mse = tf.reduce_mean(2 * tf.nn.l2_loss(pred_image - lab_images)) / (image_dim * image_dim * 100 * 100)
    tf.summary.scalar("HyperColumns_loss_MSE", gen_loss_mse)

    train_variables = tf.trainable_variables()
    for v in train_variables:
        if v is not None:
            tf.summary.histogram(v.op.name, v)
            tf.add_to_collection("reg_loss", tf.nn.l2_loss(v))
    train_op = train(gen_loss_mse, train_variables)

    if mode == 'train':
        print("Reading Dataset...")
        train_images, testing_images, validation_images = read_dataset()
        image_options = {"resize": True, "resize_size": image_dim, "color": "LAB"}
        batch_reader_train = dataset.BatchDatset(train_images, image_options)
        #batch_reader_validate = dataset.BatchDatset(validation_images, image_options)
        batch_reader_testing = dataset.BatchDatset(testing_images, image_options)
        #print(1)
        #print(batch_reader_train)
        #print(batch_reader_train.files)
        #print(batch_reader_train.image_options)
        #print(batch_reader_train.images)
        #print(batch_reader_train.batch_offset)
        #print(batch_reader_train.epochs_completed)
    elif mode == 'test':
        validation_precentage = 0
        test_precentage = 0
        testing_images,_,_ = read_dataset()
        image_options = {"resize": True, "resize_size": image_dim, "color": "LAB"}
        batch_reader_testing = dataset.BatchDatset(testing_images, image_options)

    print("Setting up session")
    sess = tf.Session()
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    if restore_model == True:
        ckpt = tf.train.get_checkpoint_state(model_folder_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored...")
        else:
            print("Model was not found...")

    #If you want to see the Hypercolumn data
    if info_hypercolumn_data == True :
        print('printing out the trainable variables...')
        variables_names = [v.name for v in tf.trainable_variables()]
        values = sess.run(variables_names)
        for k, v in zip(variables_names, values):
            print("Variable: ", k)
            print("Shape: ", v.shape)

    if mode == 'train':
        mse_train_list = []
        print("Train Phase")
        for itr in range(MAX_ITERATION):
            print("Iteration: " + str(itr))
            #reading the relevant images
            l_image, color_images = batch_reader_train.next_batch(batch_size)
            feed_dict = {images: l_image, lab_images: color_images, train_phase: True}

            if itr % 5 == 0:
                mse, summary_str = sess.run([gen_loss_mse, summary_op], feed_dict=feed_dict)
                mse_train_list.append(mse)
                #train_writer.add_summary(summary_str, itr)
                print("Step: %d, MSE: %g" % (itr, mse))

            if itr != 0 and itr % 15 == 0:
                saver.save(sess, model_folder_path + "model.ckpt", itr)
                pred = sess.run(pred_image, feed_dict=feed_dict)
                idx = np.random.randint(0, batch_size)
                save_dir = train_folder_path
                save_image(color_images[idx], save_dir, str(itr) + "_color")
                save_image(pred[idx].astype(np.float64), save_dir, str(itr) + "_our_pred")
                print("%s --> Model saved" % datetime.datetime.now())

            sess.run(train_op, feed_dict=feed_dict)
            if itr != 0 and itr % 10 == 0:
                learning_rate = (learning_rate * 0.5)

        print("Test Part")
        test_images_num = len(testing_images)
        print("Reading the test images")
        l_image, color_images = batch_reader_testing.get_N_images(test_images_num)
        feed_dict = {images: l_image, lab_images: color_images, train_phase: False}
        print("Entering the Images to the VGG-16")
        pred = sess.run(pred_image, feed_dict=feed_dict)
        save_dir = test_folder_path
        print("Saving the photos in test folder")
        for itr in range(test_images_num):
            save_image(color_images[itr], save_dir, str(itr) + "_color")
            save_image(pred[itr].astype(np.float64), save_dir, str(itr) + "_our_pred")
        print("--- Images saved on test run ---")

    elif mode == "test":
        print("Reading your images")
        test_images_num = len(os.listdir(images_folder_path))
        l_image, color_images = batch_reader_testing.get_N_images(test_images_num)
        feed_dict = {images: l_image, lab_images: color_images, train_phase: False}
        print("Entering the Images to the VGG-16")
        pred = sess.run(pred_image, feed_dict=feed_dict)
        save_dir = os.path.join(results_folder_path)
        print("Saving the photos in results folder")
        for itr in range(test_images_num):
            save_image(color_images[itr], save_dir, str(itr) + "_color")
            save_image(pred[itr].astype(np.float64), save_dir, str(itr) + "_our_pred")
        print("--- Images saved on test folder ---")

if __name__ == "__main__":
    main()