import tensorflow as tf
import numpy as np
import pandas as pd
import scipy.linalg
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
import matplotlib.pyplot as plt
import seaborn as sns
import re
from .. import visualize

def compute_embeddings(dataloader, count):
    image_embeddings = []
    
    inception_model = InceptionV3(include_top = False, weights = "imagenet", pooling = 'avg')

    for _ in range(count):
        images = next(iter(dataloader))
        embeddings = inception_model.predict(images)


        image_embeddings.extend(embeddings)


    return np.array(image_embeddings)

def calculate_fid(real_embeddings, generated_embeddings):
    # calculate mean and covariance statistics
    mu1, sigma1 = real_embeddings.mean(axis = 0), np.cov(real_embeddings, rowvar = False)
    mu2, sigma2 = generated_embeddings.mean(axis = 0), np.cov(generated_embeddings, rowvar = False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
     # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    
    return fid

def preprocess_for_inception(image):
    image = tf.image.grayscale_to_rgb(image)
    image = tf.image.resize(image, (299, 299))
    image = preprocess_input(image)
    
    return image

def create_inception_generated_dataset(image_dir):
    dataset = tf.data.Dataset.list_files(image_dir)
    dataset = dataset.map(lambda x: tf.cast(tf.io.decode_jpeg(tf.io.read_file(x)), tf.float32),
                                          num_parallel_calls = tf.data.AUTOTUNE)
    dataset = dataset.map(preprocess_for_inception, num_parallel_calls = tf.data.AUTOTUNE)
    dataset = dataset.batch(1)
    
    return dataset

def calculate_fid_for_epoch_and_model(train_images, models, epochs, train_size, base_path = '..', save = False):
    fids = []
    
    print('Getting inception embeddings for real images.')
    real_image_embeddings = compute_embeddings(train_images, train_size)
    
    for model in models:
        print(f'Using model {model}.')
        for epoch in epochs:
            print(f'Using epoch {epoch}.')
            generated_image_dir = f'{base_path}/data/generated/train/{model}/epoch_{epoch:03d}/*.jpg'

            print('Creating generated dataset.')
            train_generated = create_inception_generated_dataset(generated_image_dir)

            print('Getting inception embeddings for generated images.')
            generated_image_embeddings = compute_embeddings(train_generated, train_size)

            print('Calculating FID.')
            fid = calculate_fid(real_image_embeddings, generated_image_embeddings)

            if save:
                log_location = f'{base_path}/logs/{model}/fid_scores.csv'
                print(f'Writing to {log_location}.')
                with open(log_location, 'a') as f:
                    f.write(f'Epoch: {epoch:03d}, FID: {fid:.03f}')
                    
            fids.append(model, epoch, fid)
            
    return fids

def parse_fid_logs(fid_file, plot = False):
    with open(fid_file, 'r') as f:
        fids = f.read()
    
    fids_parsed = re.sub(r'(\d)(E)', '\g<1>\n\g<2>', fids).split('\n')
    fids_cols = [x.split(',') for x in fids_parsed]
    fids_parsed_numerical = [[re.search(r' (\d+\.?(\d+)?)', y)[1] for y in x] for x in fids_cols]
    
    df = pd.DataFrame(fids_parsed_numerical, columns = ['epoch', 'fid'])
    df = df.astype(float)
    df.epoch = df.epoch.astype(int).values
    
    if plot:
        fig, ax = plt.subplots(1, 1, figsize = (16, 8))      
        sns.lineplot(x = df.epoch, y = df.fid, ax = ax);
        ax.set_xlabel(f'FID Score', fontsize = 16)
            
        plt.xlim([0, 200])
            
    return df

def plot_fid_scores(merged_fid_df):
    fig, ax = plt.subplots(figsize = (16, 8))
    
    non_epoch_col = [x for x in merged_fid_df.columns if 'epoch' != x.lower()]
    
    for col in non_epoch_col:
        label = re.search('fid_(\w+)_?', col)[1].title()
        sns.lineplot(x = merged_fid_df.epoch, y = merged_fid_df[col], label = label, ax = ax);

    visualize.configure_axislabels_and_title('Epochs', 'FID Score',
                                             'Frechet Inception Distance (FID) Across Epochs', ax = ax);
    visualize.configure_ticklabels_and_params(ax = ax);
    visualize.configure_legend(ax = ax, fancybox = True, frameon = True, fontsize = 16);
    
    return fig, ax