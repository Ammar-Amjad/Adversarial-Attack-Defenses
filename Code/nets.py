#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" CIS6261TML -- Project Option 1 -- attacks.py

# This file contains the model arch and related functionality
"""
import tensorflow as tf 
from functools import reduce

# Some global constants
keras = tf.keras


def create_fashion_mnist_cnn_model(verbose=True):
  layers = [
    keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.5),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dropout(0.5),
  ]

  inputs = keras.layers.Input(shape=(28,28,1))
  nets = reduce(lambda result, func: func(result), layers, inputs)
  outputs = keras.layers.Dense(10, activation='softmax')(nets)

  model = keras.models.Model(inputs=inputs, outputs=outputs, name='fashion_mnistclassification_model')
  if verbose: model.summary()

  return model


def evaluate_model(model, x_train, y_train, x_test, y_test, verbose=True):
  train_loss, train_accuracy = model.evaluate(x_train, y_train, verbose=verbose)
  test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=verbose)
  return train_loss, train_accuracy, test_loss, test_accuracy


def train_model(model, x_train, y_train, x_test, y_test, x_aux, y_aux, num_epochs, batch_size = 128, verbose=True):
  model.compile(optimizer=keras.optimizers.Nadam(learning_rate=0.001), loss='categorical_crossentropy',metrics=[keras.metrics.categorical_accuracy])
  earlyStop = keras.callbacks.EarlyStopping(
    monitor='val_categorical_accuracy', min_delta=0, patience=10, verbose=verbose, mode='auto', baseline=None, restore_best_weights=True
  )

  model.fit(
    x_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=verbose, validation_data=(x_aux, y_aux), callbacks=[earlyStop]
  )

  return evaluate_model(model, x_train, y_train, x_test, y_test, verbose)
