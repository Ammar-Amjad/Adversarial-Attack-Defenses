#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" CIS6261TML -- Project; Part 2-- part2.py

This file consists of code for Part 2 of the Project.
The code does the following:

1. Train a CNN on the fashion-MNIST dataset
2. Implement the defense devised in Part 1 of the Project.
3. Craft adversarial examples using various attacks
4. Determine and evaluate the efficacy of the implemented defense 
"""

import os
import numpy as np
import tensorflow as tf 
import nets
import attacks
import utils
from scipy.ndimage import median_filter
from sklearn.metrics import confusion_matrix
import defense

# --- Config ---
num_epochs = 10
num_adv_samples = 20
eps = 20


# Some global constants
keras = tf.keras
num_classes = 10 # Fashion-MNIST is a drop-in replacement for MNIST and has the same number of classes
dirpath = os.path.join(os.getcwd(), 'models')
base_fp = '{}/{}'.format(dirpath, 'fashion-mnst-cnn.h5')


def load_fashion_mnist_data(training_size = 50000):
  ''' This function loads the Fashion-MNIST dataset and returns this data after preprocessing '''
  (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

  # Fashion MNIST has overall shape (60000, 28, 28) -- 60k images, each is 28x28 pixels same as MNIST
  print('Loaded mnist data; shape: {} [y: {}], test shape: {} [y: {}]'.format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

  # Put the labels in "one-hot" encoding using keras' to_categorical()
  y_train = keras.utils.to_categorical(y_train, num_classes)
  y_test = keras.utils.to_categorical(y_test, num_classes)

  # let's split the training set further
  aux_idx = training_size

  x_aux = x_train[aux_idx:,:]
  y_aux = y_train[aux_idx:,:]

  x_temp = x_train[:aux_idx,:]
  y_temp = y_train[:aux_idx,:]

  x_train = x_temp
  y_train = y_temp

  return (x_train, y_train), (x_test, y_test), (x_aux, y_aux)


def apply_median_filtering(images, size):
  filtered_images = []
  for image in images:
    filtered_image = median_filter(image, size=size)
    filtered_images.append(filtered_image)
  return np.array(filtered_images)


def defense_predict(model, images):
  return defense.defense_predict(model, images)


def basic_predict(model, x): 
  return model(x)

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

def generate_adversarial_examples(model, x, y, alpha=0.1):
    x_adv = np.empty_like(x)
    for i, (x_sample, y_sample) in enumerate(zip(x, y)):
        x_adv_sample = do_untargeted_fgsm(model, x_sample, y_sample, alpha)
        x_adv[i] = x_adv_sample
    return x_adv
"""
## A very simple threshold-based MIA
"""
def simple_conf_threshold_mia(predict_fn, x, thresh=0.9999):   
  pred_y = predict_fn(x)
  pred_y_conf = np.max(pred_y, axis=-1)
  return (pred_y_conf > thresh).astype(int)

def main():

  print('### Python version: ' + __import__('sys').version)
  print('### NumPy version: ' + np.__version__)
  print('### Tensorflow version: ' + tf.__version__)
  print('### TF Keras version: ' + keras.__version__)
  print('------------')

  # Set a seed for numpy and tensorflow
  UFID = 62861020
  np.random.seed(UFID)
  tf.random.set_seed(UFID)

  # Loading the dataset
  train, test, aux = load_fashion_mnist_data(5000)
  x_train, y_train = train
  x_test, y_test = test
  x_aux, y_aux = aux

  does_model_exist = utils.check_if_model_exists(dirpath, base_fp)
  model = lambda x: x
  
  if does_model_exist == False:
    print("Model not found! Let's train it from scratch")

    # Get the model
    model = nets.create_fashion_mnist_cnn_model()

    # Train the model 
    _, train_accuracy, _, test_accuracy = nets.train_model(model, x_train, y_train, x_test, y_test, x_aux, y_aux, num_epochs)
    print('[Raw Model] Train accuracy: {:.2f}% --- Test accuracy: {:.2f}%'.format(100 * train_accuracy, 100 * test_accuracy))

    utils.save_model(model, base_fp)

  else:
    print("Model already exists!")

    # Load the saved model
    model, _ = utils.load_model(base_fp)
    alpha = 0.085  # You can experiment with different values
    train_x_adv = generate_adversarial_examples(model, x_train, y_train, alpha)

    # Add code for adversarail training 
    model.fit(train_x_adv, y_train, epochs=10, batch_size=16)

    _, train_accuracy, _, test_accuracy = nets.evaluate_model(model, x_train, y_train, x_test, y_test)
    print('[Raw Model] Train accuracy: {:.2f}% --- Test accuracy: {:.2f}%'.format(100 * train_accuracy, 100 * test_accuracy))

  # ------- Using the Prediction Function -------
  
  # let's wrap the model prediction function so it could be replaced to implement a defense
  predict_fn = lambda x: defense_predict(model.predict, x) # TODO: Add defense here
  
  # Reshape image to desired size and add extra dimension for channels
  x_train_pred, x_test_pred = x_train[..., np.newaxis], x_test[..., np.newaxis]
  
  # now let's evaluate the model with this prediction function
  y_pred = predict_fn(x_train_pred)
  train_acc = np.mean(np.argmax(y_train, axis=-1) == np.argmax(y_pred, axis=-1))
  
  y_pred = predict_fn(x_test_pred)
  test_acc = np.mean(np.argmax(y_test, axis=-1) == np.argmax(y_pred, axis=-1))
  print('[Model] Train accuracy: {:.2f}% --- Test accuracy: {:.2f}%'.format(100*train_acc, 100*test_acc))

  # ---------- Adversarial Examples -------------

  # Adversarial examples are to be constructed using basic_predict
  predict_fn = lambda x: basic_predict(model, x)
  samples_fp = 'adv/fgsmk_samples_eps{}.npz'.format(eps)

  do_samples_exist = utils.check_if_samples_exists(samples_fp)
  if do_samples_exist == False:
    x_benign, correct_labels, x_adv_samples, avg_dist = attacks.craft_adversarial_fgsmk(model, x_test, y_test, num_adv_samples, eps)
    np.savez_compressed(samples_fp, x_benign=x_benign, correct_labels=correct_labels, x_adv_samples=x_adv_samples, avg_dist=avg_dist)
  else:
    data = np.load(samples_fp)
    x_benign, correct_labels, x_adv_samples, avg_dist = data['x_benign'], data['correct_labels'], data['x_adv_samples'], data['avg_dist']

  # ------------- Attack Evaluation -------------

  # # Adversarial examples are to be evaluated first using basic_predict
  predict_fn = lambda x: basic_predict(model, x)  # TODO: Add defense here
  benign_acc, adv_acc, misclassified_benign, misclassified_adv = attacks.evaluate_attack(predict_fn, x_benign, correct_labels, x_adv_samples)
  print('[Raw Model] Untargeted FGSM attack eval --- benign acc: {:.1f}%, adv acc: {:.1f}% [eps={:.1f}, mean distortion: {:.3f}]\n'.format(benign_acc*100.0, adv_acc*100.0, eps, avg_dist))
  utils.plot_adversarial_example(predict_fn, misclassified_benign[:3], misclassified_adv[:3], show=True, fname='images/Raw Model - Untargeted FSGM.png')

  # Adversarial examples are to be evaluated using defense_predict
  predict_fn = lambda x: defense_predict(model, x)  # TODO: Add defense here
  benign_acc, adv_acc, misclassified_benign, misclassified_adv = attacks.evaluate_attack(predict_fn, x_benign, correct_labels, x_adv_samples)
  print('[Model] Untargeted FGSM attack eval --- benign acc: {:.1f}%, adv acc: {:.1f}% [eps={:.1f}, mean distortion: {:.3f}]\n'.format(benign_acc*100.0, adv_acc*100.0, eps, avg_dist))
  utils.plot_adversarial_example(predict_fn, misclassified_benign[:3], misclassified_adv[:3], show=True, fname='images/With Defense - Untargeted FSGM.png')

  # ------------- Membership Inference Attack -------------

  mia_eval_size = 2000
  mia_eval_data_x = np.r_[x_train[0:mia_eval_size], x_test[0:mia_eval_size]]
  mia_eval_data_in_out = np.r_[np.ones((mia_eval_size, 1)), np.zeros((mia_eval_size, 1))]
  assert mia_eval_data_x.shape[0] == mia_eval_data_in_out.shape[0]
    
  mia_attack_fns = []
  mia_attack_fns.append(('Simple MIA Attack', simple_conf_threshold_mia))
  
  
  for i, tup in enumerate(mia_attack_fns):
    attack_str, attack_fn = tup
    
    in_out_preds = attack_fn(predict_fn, mia_eval_data_x).reshape(-1,1)
    assert in_out_preds.shape == mia_eval_data_in_out.shape, 'Invalid attack output format'
    
    cm = confusion_matrix(mia_eval_data_in_out.flatten(), in_out_preds.flatten(), labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    attack_acc = np.trace(cm) / np.sum(np.sum(cm))
    attack_adv = tp / (tp + fn) - fp / (fp + tn)
    attack_precision = tp / (tp + fp)
    attack_recall = tp / (tp + fn)
    attack_f1 = tp / (tp + 0.5*(fp + fn))
    print('{} --- Attack accuracy: {:.2f}%; advantage: {:.3f}; precision: {:.3f}; recall: {:.3f}; f1: {:.3f}'.format(attack_str, attack_acc*100, attack_adv, attack_precision, attack_recall, attack_f1))
    
  

if __name__ ==  "__main__": 
  main()
