#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" CIS6261TML -- Project Option 1 -- part1.py

# This file contains the part1 code
"""

import sys
import os

import time

import numpy as np

import sklearn
from sklearn.metrics import confusion_matrix

# we'll use tensorflow and keras for neural networks
import tensorflow as tf
import tensorflow.keras as keras

import Submission.utils as utils # we need this

    
"""
## Plots an adversarial perturbation, i.e., original input orig_x, adversarial example adv_x, and the difference (perturbation)
"""
def plot_adversarial_example(pred_fn, orig_x, adv_x, labels, fname='adv_exp.png', show=True, save=True):
    perturb = adv_x - orig_x
    
    # compute confidence
    in_label, in_conf = utils.pred_label_and_conf(pred_fn, orig_x)
    
    # compute confidence
    adv_label, adv_conf = utils.pred_label_and_conf(pred_fn, adv_x)
    
    titles = ['{} (conf: {:.2f})'.format(labels[in_label], in_conf), 'Perturbation',
              '{} (conf: {:.2f})'.format(labels[adv_label], adv_conf)]
    
    images = np.r_[orig_x, perturb, adv_x]
    
    # plot images
    utils.plot_images(images, fig_size=(8,3), titles=titles, titles_fontsize=12,  out_fp=fname, save=save, show=show)  


######### Prediction Fns #########

"""
## Basic prediction function
"""
def basic_predict(model, x):
    return model(x)


#### TODO: implement your defense(s) as a new prediction function
#### Put your code here
from scipy.ndimage import gaussian_filter, median_filter, generic_filter, maximum_filter, minimum_filter, uniform_filter, fourier_ellipsoid, fourier_gaussian, fourier_uniform, spline_filter, gaussian_laplace, laplace, rotate, gaussian_gradient_magnitude

def apply_generic_filtering(images, size=3):
    filtered_images = []
    for image in images:
        filtered_image = generic_filter(image, size=size)
        filtered_images.append(filtered_image)
    return np.array(filtered_images)

import numpy as np
import tensorflow as tf

def apply_gaussian_smoothing(images, sigma=1.0):
    smoothed_images = []
    for image in images:
        smoothed_image = gaussian_filter(image, sigma=sigma)
        smoothed_images.append(smoothed_image)
    return np.array(smoothed_images)

def apply_median_filtering(images, size):
    filtered_images = []
    for image in images:
        filtered_image = median_filter(image, size=size)
        filtered_images.append(filtered_image)
    return np.array(filtered_images)

def PGD_attack(model, images, labels):
    epsilon = 8.0  # Maximum allowed perturbation for each pixel
    alpha = 2.0  # Step size for each iteration
    num_iter = 10  # Number of iterations

    filtered_images = []
    for image, label in zip(images, labels):
        filtered_image = pgd_attack(model, image, label, epsilon, alpha, num_iter)
        filtered_images.append(filtered_image)
    return np.array(filtered_images)

def defense_predict_fn(model, images):
    # Our Defense: Double median filter (2, 2, 2)
    images = apply_median_filtering(images, size=2)
    images = apply_median_filtering(images, size=2)

    return model.predict(images)

def generate_adversarial_examples(model, x, y, alpha=0.1):
    x_adv = np.empty_like(x)
    for i, (x_sample, y_sample) in enumerate(zip(x, y)):
        x_adv_sample = do_untargeted_fgsm(model, x_sample, y_sample, alpha)
        x_adv[i] = x_adv_sample
    return x_adv
    


######### Membership Inference Attacks (MIAs) #########

"""
## A very simple threshold-based MIA - Blackbox MIA attack
"""
def simple_conf_threshold_mia(predict_fn, x, thresh=0.9999):   
    pred_y = predict_fn(x)
    pred_y_conf = np.max(pred_y, axis=-1)
    return (pred_y_conf > thresh).astype(int)
    
    
#### TODO [optional] implement new MIA attacks.
#### Put your code here
# def whitebox_mia(model, shadow_model, attack_train_x, attack_train_y, attack_test_x, attack_test_y):
#     # Train the shadow model on the same architecture as the target model
#     shadow_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     shadow_model.fit(attack_train_x, attack_train_y, epochs=10, batch_size=64, verbose=0)

#     # Calculate the loss values for the training and test data
#     attack_train_loss = shadow_model.evaluate(attack_train_x, attack_train_y, verbose=0)[0]
#     attack_test_loss = shadow_model.evaluate(attack_test_x, attack_test_y, verbose=0)[0]

#     # Define a threshold for classifying a data point as in or out of the training set
#     threshold = (attack_train_loss + attack_test_loss) / 2

#     # Perform the white-box attack by comparing the losses
#     all_data_x = np.concatenate((attack_train_x, attack_test_x), axis=0)
#     all_data_losses = shadow_model.evaluate(all_data_x, verbose=0)[0]

#     # Classify the data points based on the threshold
#     predictions = (all_data_losses > threshold).astype(int)

#     true_labels = np.concatenate((np.ones(attack_train_x.shape[0]), np.zeros(attack_test_x.shape[0])), axis=0)

#     return predictions, true_labels


  
######### Adversarial Examples #########

  
#### TODO [optional] implement new adversarial examples attacks.
#### Put your code here  
#### Note: you can have your code save the data to file so it can be loaded and evaluated in Main() (see below).
    
def pgd_attack(model, x, y, epsilon, alpha, num_iter):
    x = tf.convert_to_tensor(np.expand_dims(x, axis=0), dtype=tf.float32)
    y = tf.convert_to_tensor(y.reshape((1, -1)), dtype=tf.float32)

    x_adv = tf.identity(x)
    for _ in range(num_iter):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            y_pred = model(x_adv)
            y = np.argmax(y_pred)
            loss = tf.keras.losses.categorical_crossentropy(y, y_pred)

        gradient = tape.gradient(loss, x_adv)
        perturbation = alpha * tf.sign(gradient)
        x_adv = x_adv + perturbation

        # Project the perturbed image back into the epsilon-ball around the original image
        delta = tf.clip_by_value(x_adv - x, -epsilon, epsilon)
        x_adv = x + delta

        # Ensure the perturbed image is still within the valid data range [0, 255]
        x_adv = tf.clip_by_value(x_adv, 0, 255)

    return x_adv.numpy()

def done_fn_gn(model, x_in, x_adv, target, i, conf=0.8, max_iter=100):
    if i >= max_iter:
        return True
    
    y_pred_v = model.predict(x_adv, verbose=0)[0]
    y_pred = np.argmax(y_pred_v, axis=-1)
        
    return y_pred == target and y_pred_v[y_pred] >= conf
"""
## Fast Gradient Sign Method (FGSM) for untargetted perturbations
##
"""
def do_untargeted_fgsm(model, in_x, y, alpha):
    grad_vec = gradient_of_loss_wrt_input(model, in_x, y)
    perturb = alpha * tf.sign(grad_vec)  
    adv_x = in_x + perturb 
    adv_x = tf.clip_by_value(adv_x, 0, 255.0)
    return adv_x 
"""
## Gradient Noise Attack. Returns the adversarial perturbation.
## How does this attack work????
## The attack runs until the termination condition or the maximum number of iterations is reached (whichever occurs first)
"""
def gradient_noise_attack(model, x_input, y_input, max_iter, terminate_fn, alpha=5, sf=1e-12):
    x_in = tf.convert_to_tensor(np.expand_dims(x_input, axis=0), dtype=tf.float32)
    x_adv = x_in                        # initial adversarial example
    y_flat = np.argmax(y_input)

    for i in range(0, max_iter):

        # grab the gradient of the loss (given y_input) with respect to the input!
        grad_vec = gradient_of_loss_wrt_input(model, x_adv, y_input)
        
        ### why might the following two lines be a good idea?
        if np.sum(np.abs(grad_vec)) < sf:
            grad_vec = 0.0001 * tf.random.normal(grad_vec.shape) 
        
        # create perturbation
        r = tf.random.uniform(grad_vec.shape)
        perturb = alpha * r * tf.sign(grad_vec)
        
        # add perturbation
        x_adv = x_adv + perturb

        iters = i+1 # save the number of iterations so we can return it
        
        x_adv = tf.clip_by_value(x_adv, 0, 255.0)
        
        # set the most likely incorrect label as target
        y_pred = model(x_adv)[0].numpy()
        y_pred[y_flat] = 0
        target_class_number = np.argmax(y_pred, axis=-1)

        # check if we should stop the attack early
        if terminate_fn(model, x_in, x_adv, target_class_number, iters):
            break

    return x_adv.numpy().astype(int), iters
    
### gradient_of_loss_wrt_input function 
"""
## Computes the gradient of the categorical cross entropy loss with respect to the input ('x')
"""
def gradient_of_loss_wrt_input(model, x, y):
    x = tf.convert_to_tensor(np.expand_dims(x, axis=0), dtype=tf.float32)
    y = tf.convert_to_tensor(y.reshape((1, -1)), dtype=tf.float32) 
    with tf.GradientTape() as tape:
        tape.watch(x)
        y_pred = model(x)
        Loss = tf.keras.losses.categorical_crossentropy(y, y_pred)
    gradient = tape.gradient(Loss, x)
    return gradient
  
######### Main() #########
   
if __name__ == "__main__":


    # Let's check our software versions
    print('### Python version: ' + __import__('sys').version)
    print('### NumPy version: ' + np.__version__)
    print('### Scikit-learn version: ' + sklearn.__version__)
    print('### Tensorflow version: ' + tf.__version__)
    print('### TF Keras version: ' + keras.__version__)
    print('------------')


    # global parameters to control behavior of the pre-processing, ML, analysis, etc.
    seed = 42

    # deterministic seed for reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # keep track of time
    st = time.time()

    #### load the data
    print('\n------------ Loading Data & Model ----------')
    
    train_x, train_y, test_x, test_y, val_x, val_y, labels = utils.load_data()
    num_classes = len(labels)
    assert num_classes == 10 # cifar10
    
    ### load the target model (the one we want to protect)
    target_model_fp = './target-model.h5'

    model, _ = utils.load_model(target_model_fp)
    # # model.summary() ## you can uncomment this to check the model architecture (ResNet)
    
    # Code for generating adversarial examples
    alpha = 0.085  # We can change alpha value to control perturbations
    train_x_adv = generate_adversarial_examples(model, train_x, train_y, alpha)

    model.fit(train_x_adv, train_y, epochs=10, batch_size=16)

    st_after_model = time.time()
        
    ### let's evaluate the raw model on the train and test data
    train_loss, train_acc = model.evaluate(train_x, train_y, verbose=0)
    test_loss, test_acc = model.evaluate(test_x, test_y, verbose=0)
    print('[Raw Model] Train accuracy: {:.2f}% --- Test accuracy: {:.2f}%'.format(100*train_acc, 100*test_acc))
    
    
    ### let's wrap the model prediction function so it could be replaced to implement a defense
    predict_fn = lambda x: basic_predict(model, x)
    # predict_fn = lambda x: defense_predict_fn(model, x)
    
    ### now let's evaluate the model with this prediction function
    pred_y = predict_fn(train_x)
    train_acc = np.mean(np.argmax(train_y, axis=-1) == np.argmax(pred_y, axis=-1))
    
    pred_y = predict_fn(test_x)
    test_acc = np.mean(np.argmax(test_y, axis=-1) == np.argmax(pred_y, axis=-1))
    print('[Model] Train accuracy: {:.2f}% --- Test accuracy: {:.2f}%'.format(100*train_acc, 100*test_acc))
        
    
    ### evaluating the privacy of the model wrt membership inference
    mia_eval_size = 2000
    mia_eval_data_x = np.r_[train_x[0:mia_eval_size], test_x[0:mia_eval_size]]
    mia_eval_data_in_out = np.r_[np.ones((mia_eval_size,1)), np.zeros((mia_eval_size,1))]
    assert mia_eval_data_x.shape[0] == mia_eval_data_in_out.shape[0]
    
    # so we can add new attack functions as needed
    print('\n------------ Privacy Attacks ----------')
    mia_attack_fns = []
    mia_attack_fns.append(('Simple MIA Attack', simple_conf_threshold_mia))
    
    for i, tup in enumerate(mia_attack_fns):
        attack_str, attack_fn = tup
        
        in_out_preds = attack_fn(predict_fn, mia_eval_data_x).reshape(-1,1)
        print("S", in_out_preds.shape, mia_eval_data_in_out.shape)
        assert in_out_preds.shape == mia_eval_data_in_out.shape, 'Invalid attack output format'
        
        cm = confusion_matrix(mia_eval_data_in_out, in_out_preds, labels=[0,1])
        tn, fp, fn, tp = cm.ravel()
        
        attack_acc = np.trace(cm) / np.sum(np.sum(cm))
        attack_adv = tp / (tp + fn) - fp / (fp + tn)
        attack_precision = tp / (tp + fp)
        attack_recall = tp / (tp + fn)
        attack_f1 = tp / (tp + 0.5*(fp + fn))
        print('{} --- Attack accuracy: {:.2f}%; advantage: {:.3f}; precision: {:.3f}; recall: {:.3f}; f1: {:.3f}'.format(attack_str, attack_acc*100, attack_adv, attack_precision, attack_recall, attack_f1))
    
    
    
    ### evaluating the robustness of the model wrt adversarial examples
    print('\n------------ Adversarial Examples ----------')
    advexp_fps = []
    advexp_fps.append(('Adversarial examples attack0', 'advexp0.npz'))
    advexp_fps.append(('Adversarial examples attack1', 'advexp1.npz'))
    
    for i, tup in enumerate(advexp_fps):
        attack_str, attack_fp = tup
        
        data = np.load(attack_fp)
        adv_x = data['adv_x']
        benign_x = data['benign_x']
        benign_y = data['benign_y']
        
        benign_pred_y = predict_fn(benign_x)
        #print(benign_y[0:10], benign_pred_y[0:10])
        benign_acc = np.mean(benign_y == np.argmax(benign_pred_y, axis=-1))
        
        adv_pred_y = predict_fn(adv_x)
        #print(benign_y[0:10], adv_pred_y[0:10])
        adv_acc = np.mean(benign_y == np.argmax(adv_pred_y, axis=-1))
        
        print('{} --- Benign accuracy: {:.2f}%; adversarial accuracy: {:.2f}%'.format(attack_str, 100*benign_acc, 100*adv_acc))
        
    print('------------\n')

    et = time.time()
    
    print('Elapsed time -- total: {:.1f} seconds (data & model loading: {:.1f} seconds)'.format(et - st, st_after_model - st))

    sys.exit(0)
