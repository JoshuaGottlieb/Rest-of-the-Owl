import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import re
import os
from .. import utils

def split_pair(pair):
    pair_arr = np.array(pair)
    width = pair_arr.shape[1] // 2
    sketch = pair_arr[:, width:]
    image = pair_arr[:, :width]
    
    
    
    return sketch, image

def rescale(image, factor = 255):
    image = image.astype(float)
    image = image / factor
    
    return image

def convert_to_tensor(image):
    tensor = tf.convert_to_tensor(image)
    tensor = tf.expand_dims(tensor, axis = -1)
    tensor = tf.expand_dims(tensor, axis = 0)
    tensor = tf.cast(tensor, tf.float32)
    
    return tensor

def generate_image(generator, sketch_tensor):
    prediction = generator(sketch_tensor, training = True)
    
    prediction_numpy = (prediction[0, :, :, 0] * 255).numpy()
    prediction_numpy[prediction_numpy < 0] = 0
    
    prediction_image = Image.fromarray(prediction_numpy.astype(np.uint8))
    
    return prediction_image

def save_generated_image(image, end_dir, image_name):
    if not os.path.exists(end_dir):
        os.mkdir(end_dir)
    image.save(end_dir + '/' + image_name)
    
    return

def generate_and_save_image(image_path, generator, end_dir, epoch, model, save = True):
    im = Image.open(image_path)
    
    print('Converting image.')
    
    sketch, image = split_pair(im)
    sketch = rescale(sketch)
    sketch = convert_to_tensor(sketch)
    
    print('Generating prediction.')
    
    prediction_image = generate_image(generator, sketch)

    if save:
        image_name = f'Generated_Owl_Model_{model.title()}_Epoch_{epoch:03d}_{image_path[-8:-4]}.jpg'
        
        print('Saving image.')

        save_generated_image(prediction_image, end_dir, image_name)

        print(f'{image_name} saved to {end_dir}')
    
    return prediction_image

def generate_images_from_dataset_epochs(datasets, models, epochs, base_path = '..', save = False):
    
    prediction_images = []
    
    for dataset in datasets:
        print(f'Generating images from {dataset}.')
        data = os.listdir(f'{base_path}/data/{dataset}')
        for model in models:
            print(f'Generating images using {model}.')
            for epoch in epochs:
                print(f'Generating images using epoch {epoch}.')
                epoch_dir = f'{base_path}/models/{model}/epoch_{epoch:03d}'
                generator = utils.reload_model_from_epoch(epoch_dir, model, base_path)[0]

                for image in data:
                    print(f'Generating image for {image}.')
                    end_dir = f'{base_path}/data/generated/{dataset}/{model}/epoch_{epoch}'
                    image_path = f'{base_path}/data/{dataset}/{image}'
                    prediction_images.append(generate_and_save_image(image_path, generator,
                                                                     end_dir, epoch, model, save = save))

    return prediction_images

def generate_images_from_epochs(images, models, epochs, base_path = '..', save = False):

    generated_images = []
    
    for epoch in epochs:
        for model in models:
            generator = utils.reload_model_from_epoch(f'../models/{model}/epoch_{epoch:03d}/', model, '..')[0]
            for image in images:
                generated_images.append((f'{image[:-9]}_{model}_{epoch:03d}.jpg',
                                         generate_and_save_image(f'../visualizations/non_owls/{image}',
                                                                 generator, '../visualizations/non_owls/',
                                                                 '', model, save = False)))
                
    return generated_images