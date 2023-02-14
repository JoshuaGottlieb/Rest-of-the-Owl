import tensorflow as tf
import numpy as np
import pandas as pd
import scipy.linalg
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
import matplotlib.pyplot as plt
import seaborn as sns
import re
from ..modeling import preprocess as model_prep

def compute_embeddings(dataloader, count):
    '''
    Calculates the Inception embeddings for each image in a dataset.
    Returns a numpy array containing the vector embeddings of each image.
    
    Copied from
    wandb.ai/ayush-thakur/gan-evaluation/reports/How-to-Evaluate-GANs-using-Frechet-Inception-Distance-FID---Vmlldzo0MTAxOTI
    
    dataloader: tf.data.Dataset, containing images from a set to comput the embeddings on.
    count: int, number of images to calculate embeddings for, should be no more than the number of images in the dataset.
    '''
    
    image_embeddings = []
    
    # Instantiate the Inception model.
    inception_model = InceptionV3(include_top = False, weights = "imagenet", pooling = 'avg')

    # Use the Inception model to create embeddings for each image.
    for _ in range(count):
        images = next(iter(dataloader))
        embeddings = inception_model.predict(images)


        image_embeddings.extend(embeddings)


    return np.array(image_embeddings)

def calculate_fid(real_embeddings, generated_embeddings):
    '''
    Calculates the Frechet Inception Distance (FID) between the real and generated embeddings.
    
    Copied from
    wandb.ai/ayush-thakur/gan-evaluation/reports/How-to-Evaluate-GANs-using-Frechet-Inception-Distance-FID---Vmlldzo0MTAxOTI
    
    Originally adapted from the paper "Rethinking the Inception Architecture for Computer Vision"
    https://arxiv.org/pdf/1512.00567.pdf
    
    real_embeddings: np.array, containing the vector embeddings of the ground-truth images.
    generated_embeddings: np.array, containing the vector embeddings of the generated images.
    '''
    
    # calculate mean and covariance statistics
    mu1, sigma1 = real_embeddings.mean(axis = 0), np.cov(real_embeddings, rowvar = False)
    mu2, sigma2 = generated_embeddings.mean(axis = 0), np.cov(generated_embeddings, rowvar = False)
    
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    
    # calculate sqrt of product between cov
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - (2.0 * covmean))
    
    return fid

def preprocess_for_inception(image):
    '''
    Preprocesses an image for input into the Inception v3 model. Returns the preprocessed image.
    
    image: 3D tensor of form (width, height, 1), image to process.
    '''
    
    # Convert from grayscale to RGB
    image = tf.image.grayscale_to_rgb(image)
    
    # Resize to correct shape.
    image = tf.image.resize(image, (299, 299))
    
    # Apply Inception custom preprocessing.
    image = preprocess_input(image)
    
    return image

def create_inception_dataset(image_dir, split = False):
    '''
    Creates a tf.data.Dataset formatted for use in the Inception v3 model. Returns a formatted dataset object.
    
    image_dir: str, directory to extract images from.
    split: bool, whether to split the images as in the case of the sketch/image pairs. Default False.
    '''
    
    # Creates a dataset from the image directory.
    dataset = tf.data.Dataset.list_files(image_dir)
    
    # Applies preliminary preprocessing to convert images to tensors.
    if split:
        dataset = dataset.map(model_prep.load, num_parallel_calls = tf.data.AUTOTUNE)
        dataset = dataset.map(lambda x, y: y, num_parallel_calls = tf.data.AUTOTUNE)
    else:
        dataset = dataset.map(lambda x: tf.cast(tf.io.decode_jpeg(tf.io.read_file(x)), tf.float32),
                                              num_parallel_calls = tf.data.AUTOTUNE)

    # Processes images for inception.
    dataset = dataset.map(preprocess_for_inception, num_parallel_calls = tf.data.AUTOTUNE)
    dataset = dataset.batch(1)
    
    return dataset

def calculate_fid_for_epoch_and_model(train_image_dir, models, epochs, train_size, base_path = '..', save = False):
    '''
    Calculates the FID scores for each model in models and each epoch in epochs. Returns a list of FID scores.
    
    train_image_dir: str, denoting the directory containing the training images.
    models: list of str, denoting the models to use for calculating the FID scores.
    epochs: list of int, denoting the epochs to use for calculating the FID scores.
    train_size: int, denoting the size of the training set.
    base_path: str, denoting the base path of repository, for use in specifying log and generated data locations.
               Default ".."
    save: bool, denoting whether to save the results to a log file. Default False.
    '''
    
    fids = []
    
    # Calculate the embeddings for the ground-truth images.
    print('Creating train dataset.')
    train_images = create_inception_dataset(train_image_dir)
    
    print('Getting inception embeddings for real images.')
    real_image_embeddings = compute_embeddings(train_images, train_size)
    
    # Each model and epoch, calculate the embeddings for the generated images, then calculate the FID score.
    # Write the score to log.
    for model in models:
        print(f'Using model {model}.')
        for epoch in epochs:
            print(f'Using epoch {epoch}.')
            generated_image_dir = f'{base_path}/data/generated/train/{model}/epoch_{epoch:03d}/*.jpg'

            print('Creating generated dataset.')
            train_generated = create_inception_dataset(generated_image_dir)

            print('Getting inception embeddings for generated images.')
            generated_image_embeddings = compute_embeddings(train_generated, train_size)

            print('Calculating FID.')
            fid = calculate_fid(real_image_embeddings, generated_image_embeddings)
            
            # Write FID score to log file.
            if save:
                log_location = f'{base_path}/logs/{model}/fid_scores.csv'
                print(f'Writing to {log_location}.')
                with open(log_location, 'a') as f:
                    f.write(f'Epoch: {epoch:03d}, FID: {fid:.03f}')
                    
            fids.append((model, epoch, fid))
            
    return fids