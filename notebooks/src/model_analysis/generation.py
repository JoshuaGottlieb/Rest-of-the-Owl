import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import re
import os
from .. import utils

def split_pair(pair):
    '''
    Splits a sketch/image pair into two separate numpy arrays. Returns a tuple of arrays (sketch, image).
    
    pair: PIL.Image.Image, sketch/image pair to split.
    '''
        
    pair_arr = np.array(pair)
    width = pair_arr.shape[1] // 2
    
    # Split along halfway point of vertical axis.
    sketch = pair_arr[:, width:]
    image = pair_arr[:, :width] 
    
    return sketch, image

def rescale(image, factor = 255):
    '''
    Rescales image pixel values to lie between [0, 1]. Returns a numpy array with rescaled values.
    
    image: np.array, image array to rescale.
    factor: int, factor by which to divide to rescale image. Default 255.
    '''
    
    # Convert to float and divide values.
    image = image.astype(float)
    image = image / factor
    
    return image

def convert_to_tensor(image):
    '''
    Converts a numpy array representation of an image to a 4-dimensional tensorflow tensor.
    Returns a 4D tensor representation of the image.
    
    image: np.array, image array to convert.
    '''
    
    # Convert to tensor, then add dimension of size 1 to convert from (width, height) to (batch, width, height, channels)
    tensor = tf.convert_to_tensor(image)
    tensor = tf.expand_dims(tensor, axis = -1)
    tensor = tf.expand_dims(tensor, axis = 0)
    tensor = tf.cast(tensor, tf.float32)
    
    return tensor

def generate_image(generator, sketch_tensor):
    '''
    Generates an image using a trained generator and a 4D tensor representation of a sketch.
    Returns the predicted image as a PIL.Image.Image.
    
    generator: tf.keras.Model, a trained generator model to use to generate predictions.
    sketch_tensor: 4D tensor, a sketch to input into the generator to create a predicted image.
    '''
    
    # Generate prediction tensor.
    prediction = generator(sketch_tensor, training = True)
    
    # Convert from 4D tensor to 2D numpy array, rescaled to [0, 255].
    prediction_numpy = (prediction[0, :, :, 0] * 255).numpy()
    
    # Prevent underflow issues by pushing negative numbers to 0.
    prediction_numpy[prediction_numpy < 0] = 0
    
    # Create an image from the prediction array.
    prediction_image = Image.fromarray(prediction_numpy.astype(np.uint8))
    
    return prediction_image

def save_generated_image(image, end_dir, image_name):
    '''
    Save an image to a directory, creating the directory if needed.
    
    image: PIL.Image.Image, image to save.
    end_dir: str, denoting the directory to write the saved image to.
    image_name: str, denoting the name of the file to save the image as.
    '''
    
    # Make the directory if it does not exist.
    if not os.path.exists(end_dir):
        os.mkdir(end_dir)
    
    # Save the image
    image.save(end_dir + '/' + image_name)
    
    return

def generate_and_save_image(image_path, generator, end_dir, epoch, model, save = True):
    '''
    Takes in a sketch/image pair and a generator and returns a predicted image, optionally saving the predicted image.
    
    image_path: str, denoting the file location of the sketch/image pair to use for generation.
    generator: tf.keras.Model, trained generator model to use for generating a predicted image.
    end_dir: str, denoting the directory to write the saved image to.
    epoch: int, denoting the epoch used for generation, to be used for formatting the end image name.
    model: str, denoting the model used for generation, to be used for formatting the end image name.
    save: bool, whether to save the image. Default True.
    '''
    
    # Load the sketch/image pair.
    im = Image.open(image_path)
    
    # Convert to format usable by generator.
    print('Converting image.')
    
    sketch, image = split_pair(im)
    sketch = rescale(sketch)
    sketch = convert_to_tensor(sketch)
    
    # Generate the predicted image.
    print('Generating prediction.')
    
    prediction_image = generate_image(generator, sketch)

    # Save the generated image.
    if save:
        image_name = f'Generated_Owl_Model_{model.title()}_Epoch_{epoch:03d}_{image_path[-8:-4]}.jpg'
        
        print('Saving image.')

        save_generated_image(prediction_image, end_dir, image_name)

        print(f'{image_name} saved to {end_dir}')
    
    return prediction_image

def generate_images_from_dataset_epochs(datasets, models, epochs, base_path = '..', save = False):
    '''
    Generates images from a dataset directory using a variety of generators. Returns a list of predicted images.
    
    datasets: list of str, denoting the datasets to use, generally "test" or "train".
    models: list of str, denoting the models to use.
    epochs: list of int, denoting the epochs to use.
    base_path: str, denoting the base path of repository, for use in navigating directory structure. Default '..'.
    save: bool, whether the save the image. Default False.
    '''
    
    prediction_images = []
    
    # For each dataset, fetch the sketch/image pairs from the directory.
    for dataset in datasets:
        print(f'Generating images from {dataset}.')
        data = os.listdir(f'{base_path}/data/{dataset}')
        
        # For each model, iterate across epochs, using the trained generator for that model and epoch.
        for model in models:
            print(f'Generating images using {model}.')
            for epoch in epochs:
                print(f'Generating images using epoch {epoch}.')
                
                # Load generator.
                epoch_dir = f'{base_path}/models/{model}/epoch_{epoch:03d}'
                generator = utils.reload_model_from_epoch(epoch_dir, model, base_path)[0]

                # Generate image using generator.
                for image in data:
                    print(f'Generating image for {image}.')
                    end_dir = f'{base_path}/data/generated/{dataset}/{model}/epoch_{epoch}'
                    image_path = f'{base_path}/data/{dataset}/{image}'
                    prediction_images.append(generate_and_save_image(image_path, generator,
                                                                     end_dir, epoch, model, save = save))

    return prediction_images

def generate_images_from_epochs(images, models, epochs, base_path = '..', save = False):
    '''
    Generates images using generators from model/epoch pairs for non-test set data. Returns a list of predicted images.
    
    images: list of str, denoting sketch/image pairs to use of the form ".+_pair.jpg".
    models: list of str, denoting the models to use.
    epochs: list of int, denoting the epochs to use.
    base_path: str, denoting the base path of repository, for use in navigating directory structure. Default '..'.
    save: bool, whether the save the image. Default False.
    '''
    
    generated_images = []
    
    # For each model, iterate across epochs, using the trained generator for that model and epoch.
    for model in models:
        print(f'Generating images using {model}.')
        for epoch in epochs:
            print(f'Generating images using epoch {epoch}.')
            
            # Load generator.
            generator = utils.reload_model_from_epoch(f'../models/{model}/epoch_{epoch:03d}/', model, '..')[0]
            
            # Generate image using generator.
            for image in images:
                generated_images.append((f'{image[:-9]}_{model}_{epoch:03d}.jpg',
                                         generate_and_save_image(f'../visualizations/non_owls/{image}',
                                                                 generator, '../visualizations/non_owls/',
                                                                 '', model, save = False)))
                
    return generated_images