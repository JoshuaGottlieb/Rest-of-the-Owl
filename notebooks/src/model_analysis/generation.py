import numpy as np
import tensorflow as tf
from PIL import Image
import re
import pandas as pd

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