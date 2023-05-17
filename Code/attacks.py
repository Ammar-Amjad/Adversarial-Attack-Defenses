#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" CIS6261TML -- Project Option 1 -- attacks.py

This file contains attacks for Part 2 of the project

It comprises of the following attacks:
1. FGSM Attack
2. MIA
3. Shokri Attack (Maybe)
"""
import numpy as np
import tensorflow as tf
import sys
import utils


def gradient_of_loss_wrt_input(model, x, y):
    # Create a context for computing gradients
    with tf.GradientTape() as tape:
        # Watch the input tensor
        tape.watch(x)
        # Compute the model's predictions
        y_pred = model(x)[0]
        # Compute the loss
        loss = tf.keras.losses.categorical_crossentropy(y_true=y, y_pred=y_pred)
    
    # Calculate the gradient of the loss with respect to the input
    grad = tape.gradient(loss, x)

    return grad




def do_untargeted_fgsm(model, in_x, y, alpha):
	'''Fast Gradient Sign Method (FGSM) for untargetted perturbations'''
	grad_vec = gradient_of_loss_wrt_input(model, in_x, y)
	perturb = alpha * tf.sign(grad_vec)  
	adv_x = in_x + perturb 
	adv_x = tf.clip_by_value(adv_x, 0, 255.0)
	return adv_x




"""
## Iterative Fast Gradient Sign Method (FGSM) for untargetted perturbations
## 'stop_fn' is a caller-defined function that returns True when the attack should terminate
## The output is the adversarial example and the number of iterations performed
"""
def iterative_fgsm(model, in_x, y, eps, terminate_fn, num_classes=10):
	adv_x =  tf.convert_to_tensor(in_x, dtype=tf.float32)
	y_onehot = tf.keras.utils.to_categorical(y, num_classes)
	
	minv = np.maximum(in_x.astype(float) - eps, 0.0)
	maxv = np.minimum(in_x.astype(float) + eps, 255.0)
	
	alpha = 8 # step magnitude
	
	i = 0
	while True:
		# print('> check', tf.shape(adv_x))
		# do one step of FGSM
		adv_x = do_untargeted_fgsm(model, adv_x, y_onehot, alpha)
		
		# clip to ensure we stay within an epsilon radius of the input
		adv_x = tf.clip_by_value(adv_x, clip_value_min=minv, clip_value_max=maxv)

		# check if predicted label is the target
		adv_label, adv_conf = utils.pred_label_and_conf_fgsm(model, adv_x)
		
		i += 1

		# call the stop function and exit if needed
		if terminate_fn(model, in_x, adv_x, y, i):
			break
			
	return adv_x.numpy().astype(int), i


"""
## Runs FGSMk
"""
def run_iterative_fgsm(model, x_in, y, eps, max_iter=100, conf=0.8):
	def done_fn_untargeted(model, x_in, x_adv, t, i, max_iter, conf):
		if i >= max_iter:
			return True
		y_v = model.predict(x_adv, verbose=0).reshape(-1)
		return np.argmax(y_v) != t and np.amax(y_v) >= conf

	terminate_fn = lambda m, x, xa, t, i: done_fn_untargeted(m, x, xa, t, i, max_iter, conf)
	x_adv, iters = iterative_fgsm(model, x_in, y, eps, terminate_fn)

	return x_adv


"""
## Computes the distortion between x_in and x_adv
"""
def distortion(x_in, x_adv):
	return np.mean(np.square(x_in - x_adv))


### Craft 'num_adv_samples' adversarial examples
def craft_adversarial_fgsmk(model, x_aux, y_aux, num_adv_samples, eps):
	# Initialize an array for storing those adversarial examples and also their correct labels
	x_adv_samples = np.zeros((num_adv_samples, x_aux[0].shape[0], x_aux[0].shape[1], 1)) # (num_adv_samples, 28, 28, 1)
	correct_labels = np.zeros((num_adv_samples,)) # (num_adv_samples,) These are the correct labels

	avg_dist = 0.0
	sys.stdout.write('Crafting {} adversarial examples (untargeted FGSMk -- eps: {})\n'.format(num_adv_samples, eps))
	x_benign = None

	for i in range(0, num_adv_samples):
		# Select a random benign example to use for generating the adversarial example
		ridx = np.random.randint(low=0, high=x_aux.shape[0])

		x_input = x_aux[ridx].reshape((1, 28, 28, 1)) # A single input that would be read by the model
		y_input = y_aux[ridx, :] # The correct label

		# keep track of the benign examples
		x_benign = x_input if x_benign is None else np.r_[x_benign, x_input]
		
		correct_labels[i] = np.argmax(y_input, axis=-1)
		
		x_adv = run_iterative_fgsm(model, x_input, correct_labels[i], eps)
		x_adv_samples[i,:] = x_adv
		
		avg_dist += distortion(x_input, x_adv)
		
		sys.stdout.write('.')
		sys.stdout.flush()
	print('Done.')
	
	avg_dist /= num_adv_samples

	return x_benign, correct_labels, x_adv_samples, avg_dist


"""
## Evaluates an adversarial example attack 
## 'model_predict_fn': a prediction function for the target model
## 'x_benign': benign samples with associate labels 'y_true_labels'
## 'y_true_labels': the labels of benign samples
## 'x_adv_samples': the adversarial examples produced for each of the benign samples
## 
## The output is a tuple of (benign accuracy, adversarial accuracy)
"""
def evaluate_attack(model_predict_fn, x_benign, y_true_labels, x_adv_samples):
	assert x_benign.shape[0] == y_true_labels.shape[0]
	assert x_benign.shape == x_adv_samples.shape
	
	y_true_labels = y_true_labels.astype(int)
			
	benign_preds = model_predict_fn(x_benign)
	y_preds = np.argmax(benign_preds, axis=-1)
	
	benign_accuracy = np.mean((y_true_labels == y_preds).astype(int))
	
	adv_preds = model_predict_fn(x_adv_samples)
	y_preds = np.argmax(adv_preds, axis=-1)
	adv_accuracy = np.mean((y_true_labels == y_preds).astype(int))
	
	misclassified_benign = x_benign[y_preds != y_true_labels]
	misclassified_adv = x_adv_samples[y_preds != y_true_labels]
	return benign_accuracy, adv_accuracy, misclassified_benign, misclassified_adv
