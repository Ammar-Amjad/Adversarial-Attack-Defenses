import tensorflow as tf
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter, generic_filter, maximum_filter, minimum_filter, uniform_filter, fourier_ellipsoid, fourier_gaussian, fourier_uniform, spline_filter, gaussian_laplace, laplace, rotate, gaussian_gradient_magnitude
import cv2

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

def apply_generic_filtering(images, size=3):
  filtered_images = []
  for image in images:
    filtered_image = generic_filter(image, size=size)
    filtered_images.append(filtered_image)
  return np.array(filtered_images)

def apply_maximum_filtering(images, size=3):
  filtered_images = []
  for image in images:
    filtered_image = maximum_filter(image, size=size)
    filtered_images.append(filtered_image)
  return np.array(filtered_images)

def apply_minimum_filtering(images, size=3):
  filtered_images = []
  for image in images:
    filtered_image = minimum_filter(image, size=size)
    filtered_images.append(filtered_image)
  return np.array(filtered_images)

def apply_uniform_filtering(images, size=3):
  filtered_images = []
  for image in images:
    filtered_image = uniform_filter(image, size=size)
    filtered_images.append(filtered_image)
  return np.array(filtered_images)

def apply_fe_filtering(images, size=3):
  filtered_images = []
  for image in images:
    filtered_image = fourier_ellipsoid(image, size=size)
    filtered_images.append(filtered_image)
  return np.array(filtered_images)

def apply_fg_filtering(images, sigma=1):
  filtered_images = []
  for image in images:
    filtered_image = fourier_gaussian(image, sigma=sigma)
    filtered_images.append(filtered_image)
  return np.array(filtered_images)

def apply_fu_filtering(images, size=3):
  filtered_images = []
  for image in images:
    filtered_image = fourier_uniform(image, size=size)
    filtered_images.append(filtered_image)
  return np.array(filtered_images)

def apply_spline_filtering(images, size=3):
  filtered_images = []
  for image in images:
    filtered_image = spline_filter(image)
    filtered_images.append(filtered_image)
  return np.array(filtered_images)

def apply_gl_smoothing(images, sigma=1):
  filtered_images = []
  for image in images:
    filtered_image = gaussian_laplace(image, sigma=sigma)
    filtered_images.append(filtered_image)
  return np.array(filtered_images)

def apply_laplace_smoothing(images):
  filtered_images = []
  for image in images:
    filtered_image = laplace(image)
    filtered_images.append(filtered_image)
  return np.array(filtered_images)

def apply_rotate(images):
  filtered_images = []
  for image in images:
    filtered_image = rotate(image, angle=10)
    filtered_images.append(filtered_image)
  return np.array(filtered_images)

def apply_gaussianGM_smoothing(images, sigma=1.0):
  smoothed_images = []
  for image in images:
    smoothed_image = gaussian_gradient_magnitude(image, sigma=sigma)
    smoothed_images.append(smoothed_image)
  return np.array(smoothed_images)



# Rotate the filtered images before classification
def rotate(images, angle):
  smoothed_images = []
  for image in images:
    # image = cv2.convertScaleAbs(image)
    smoothed_image = tf.keras.preprocessing.image.apply_affine_transform(image, theta=angle)
    smoothed_images.append(smoothed_image)
  return np.array(smoothed_images)

import cv2
from skimage.restoration import denoise_tv_chambolle

def jpeg_compression(image, quality=75):
  encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
  _, encoded_image = cv2.imencode('.jpg', image, encode_param)
  decoded_image = cv2.imdecode(encoded_image, 1)
  return decoded_image / 255.0

def JPEG_Compression(x, jpeg_quality=90):
  x_transformed = []
  for image in x:
    image = jpeg_compression(image, quality=jpeg_quality) 
    x_transformed.append(image)
  return np.array(x_transformed)

def total_variation_minimization(image, weight=0.1):
  return denoise_tv_chambolle(image, weight=weight)

def TVM_filtering(x, tvm_weight=0.1):
  x_transformed = []
  for image in x: 
    image = total_variation_minimization(image, weight=tvm_weight)
    x_transformed.append(image)
  return np.array(x_transformed)

def merge(imagez, x):
  x_transformed = []
  for i in range(len(x)):
    image = np.mean([x[i], imagez[i]], axis=0)
    x_transformed.append(image)
  return np.array(x_transformed)

# Define spatial smoothing function
def spatial_smoothing(images, sigma=1):
  blurred_images = np.zeros_like(images)
  for i in range(images.shape[0]):
    blurred_images[i] = cv2.GaussianBlur(images[i], (0, 0), sigmaX=sigma, sigmaY=sigma)
  return blurred_images

def gaussian_smoothing(images, sigma=1.0):
  smoothed_images = []
  for image in images:
    smoothed_image = gaussian_filter(image, sigma=sigma)
    smoothed_images.append(smoothed_image)
  return np.array(smoothed_images)

def median_filtering(images, size=2):
  filtered_images = []
  for image in images:
      filtered_image = median_filter(image, size=size)
      filtered_images.append(filtered_image)
  return np.array(filtered_images)


def jpeg_compression_with_original(x, jpeg_quality=90, blend_factor=0.5):
  x_transformed = []

  def compress(image, quality):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded_image = cv2.imencode('.jpg', image, encode_param)
    decoded_image = cv2.imdecode(encoded_image, 1)
    return decoded_image / 255.0

  for image in x:
    compressed_image = compress(image, quality=jpeg_quality)
    blended_image = blend_factor * compressed_image + (1 - blend_factor) * image
    x_transformed.append(blended_image)

  return np.array(x_transformed)

def defense_predict(model, x_defense):
  x_defense = median_filtering(x_defense, size=2)
  x_defense = median_filtering(x_defense, size=2)
  return model(x_defense)