import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import time
from IPython import display
import os
from tensorflow.keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from .. import utils

def sum_tv_loss(image):
    '''
    Calculates the total variance loss.
    Adapted from https://www.tensorflow.org/tutorials/generative/pix2pix
             and https://github.com/irfanICMLL/Auto_painter

    image: 4D tensor, image to calculate loss on.
    '''
    
    loss_y = tf.nn.l2_loss(image[:, 1:, :, :] - image[:, :-1, :, :])
    loss_x = tf.nn.l2_loss(image[:, :, 1:, :] - image[:, :, :-1, :])
    loss = 2 * (loss_y + loss_x)
    loss = tf.cast(loss, tf.float32)
    return loss

def feature_loss(image, vgg):
    '''
    Calculates the feature loss based of feature maps extracted from a Visual Geometry Group (VGG) net.
    Adapted from https://www.tensorflow.org/tutorials/generative/pix2pix
             and https://github.com/irfanICMLL/Auto_painter
             
    image: 4D tensor, image to calculate loss on.
    vgg: tf.keras.Model, sub-model using VGG net output at correct layer (in this case, conv3_3)
    '''
    
#     # Instantiate a model with output at the correct layer (in this case, block3, conv2D 3)
#     model = Model(inputs = vgg.inputs, outputs = vgg.layers[9].output)
    
    # Preprocess image to conform to VGG16 input requirements
    img = tf.reshape(image, [image.shape[-3], image.shape[-2], image.shape[-1]])
    img = tf.image.grayscale_to_rgb(img)
    img = tf.image.resize(img, (224, 224))
    img = tf.expand_dims(img, axis = 0)
    img = preprocess_input(img)
    
    # Extract feature maps
    feature_maps = vgg(img)
    
    return feature_maps

@tf.function
def generator_loss_autopainter(disc_generated_output, gen_output, target, net):
    '''
    Calculates the generator loss for the autopainter model.
    Returns the total loss, the GAN loss, the L1 loss, the total-variation loss, and the feature loss.
    Adapted from https://www.tensorflow.org/tutorials/generative/pix2pix
             and https://github.com/irfanICMLL/Auto_painter
    
    disc_generated_output: 3D or 4D tensor, the output of the discriminator from the generated image(s).
    gen_output: 3D or 4D tensor, the output of the generator (the generated image(s)).
    target: 3D or 4D tensor, the ground-truth image(s).
    net: A Visual Geometry Group (VGG) net.
    '''
    
    # GAN loss.
    loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits = True)
    gen_loss_GAN = loss_obj(tf.ones_like(disc_generated_output), disc_generated_output)
    
    # L1 loss.
    gen_loss_L1 = tf.reduce_mean(tf.abs(target - gen_output))
    
    # L2 of total Variation loss.
    gen_loss_tv = tf.reduce_mean(tf.sqrt(tf.nn.l2_loss(sum_tv_loss(gen_output))))
    
    # L2 of feature loss.
    gen_loss_f = tf.reduce_mean(tf.sqrt(tf.nn.l2_loss(feature_loss(target, net) - feature_loss(gen_output, net))))
    
    # Total loss.
    gen_total_loss = gen_loss_GAN + (gen_loss_L1 * 10) + (gen_loss_tv * 1e-5) + (gen_loss_f * 1e-4)
    
    return gen_total_loss, gen_loss_GAN, gen_loss_L1, gen_loss_tv, gen_loss_f

@tf.function
def discriminator_loss_autopainter(disc_real_output, disc_generated_output):
    '''
    Calculates the discriminator loss for the pix2pix model. Returns the total discriminator loss.
    Adapted from https://www.tensorflow.org/tutorials/generative/pix2pix
             and https://github.com/irfanICMLL/Auto_painter
    
    disc_real_output: 3D or 4D tensor, the output of the discriminator from the ground-truth image(s).
    disc_generated_output: 3D or 4D tensor, the output of the discriminator from the generated image(s).
    '''
    
    loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits = True)
    real_loss = loss_obj(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_obj(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

@tf.function
def train_step_autopainter(input_image, target, generator, discriminator, gen_optimizer, discrim_optimizer, net):
    '''
    Train step for the pix2pix model. Returns losses.
    Adapted from https://www.tensorflow.org/tutorials/generative/pix2pix
             and https://github.com/irfanICMLL/Auto_painter
    
    input_image: 3D or 4D tensor, sketch to input into generator.
    target: 3D or 4D tensor, ground-truth image.
    generator: tf.Model, generator model to use and update.
    discriminator: tf.Model, discriminator model to use and update.
    gen_optimizer: tf.keras.optimizers, optimizer to use to update generator model.
    discrim_optimizer: tf.keras.optimizers, optimizer to use to update discriminator model.
    net: A Visual Geometry Group (VGG) net.
    '''

    # Run sketches and ground-truth images through generator and discriminator and calculate losses.
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training = True)

        disc_real_output = discriminator([input_image, target], training = True)
        disc_generated_output = discriminator([input_image, gen_output], training = True)

        gen_total_loss, gen_loss_GAN, gen_loss_L1, gen_loss_tv, gen_loss_f =\
            generator_loss_autopainter(disc_generated_output, gen_output, target, net)
        disc_loss = discriminator_loss_autopainter(disc_real_output, disc_generated_output)

    # Calculate gradients.
    generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

    # Apply gradients.
    gen_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
    discrim_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))
    
    return gen_total_loss, gen_loss_GAN, gen_loss_L1, gen_loss_tv, gen_loss_f, disc_loss

def fit_autopainter(train_ds, test_ds, epochs, generator, discriminator, gen_optimizer,
                    discrim_optimizer, net, log_file, model_dir, starting_epoch = 0, save = True):

    '''
    Fits a pix2pix model.
    Adapted from https://www.tensorflow.org/tutorials/generative/pix2pix
             and https://github.com/irfanICMLL/Auto_painter
    train_ds: tf.data.Dataset, the training dataset.
    test_ds: tf.data.Dataset, the test dataset.
    epochs: int, the number of epochs (passes over the entire training dataset) to run the model for.
    generator: tf.Model, generator model to use and update.
    discriminator: tf.Model, discriminator model to use and update.
    gen_optimizer: tf.keras.optimizers, optimizer to use to update generator model.
    discrim_optimizer: tf.keras.optimizers, optimizer to use to update discriminator model.
    net: A Visual Geometry Group (VGG) net.
    log_file: str, a path denoting the file to write the losses to. Directory must exist at path.
    model_dir: str, a path denoting the directory to save the model weights and optimizer configurations to.
               Directory must exist at path, but subdirectories do not need to exist at path.
    starting_epoch: int, the epoch to start at. Needed for logging and model-saving purposes. Default 0.
    save: bool, denotes whether to log the losses during training and save models. Default True.
    '''
    
    # Pull random image, sketch pair from test set to show model progress during training.
    example_input, example_target = next(iter(test_ds.take(1)))
    
    start = time.time()
    epoch_range = range(starting_epoch, starting_epoch + epochs)
    
    # Loop through each epoch.
    for epoch, _ in enumerate(epoch_range):
        display.clear_output(wait = True)

        # Print time to process previous epoch.
        if (epoch + starting_epoch) != starting_epoch:
            print(f'Time taken for epoch: {time.time() - start:.2f} sec\n')

        # Restart timer and display current results of training.    
        start = time.time()
        utils.generate_images(generator, example_input, example_target)
        print(f'Epoch: {epoch + starting_epoch + 1}')
        
        # Initialize empty array for losses.
        train_losses = np.zeros(6)
        batch_size = 0
        
        # Train step.
        for (sketch, target) in train_ds:
            train_losses += train_step_autopainter(sketch, target, generator, discriminator,
                                                   gen_optimizer, discrim_optimizer, net)
            batch_size += 1
            
        train_losses = train_losses / batch_size
        
        if save:
            # Log losses
            with open(log_file, 'a') as f:
                f.write(f'Epoch: {epoch + starting_epoch + 1}, gen_total_loss: {train_losses[0]:0.3f}, '
                        + f'gen_gan_loss: {train_losses[1]:0.3f}, ' + f'gen_l1_loss: {train_losses[2]:0.3f}, '
                        + f'gen_tv_loss: {train_losses[3]:0.3f}, gen_f_loss: {train_losses[4]:0.3f}, '
                        + f'disc_loss: {train_losses[5]:0.3f}\n')

            # Save models every 10 epochs        
            if ((epoch + starting_epoch + 1) % 10) == 0:
                epoch_dir = f'{model_dir}/epoch_{epoch + starting_epoch + 1:03d}'
                if not os.path.exists(epoch_dir):
                    os.mkdir(epoch_dir)
                generator.save(f'{epoch_dir}/generator.h5')
                discriminator.save(f'{epoch_dir}/discriminator.h5')
                utils.pickle_obj(gen_optimizer.get_config(), f'{epoch_dir}/gen_optim_config.pickle')
                utils.pickle_obj(discrim_optimizer.get_config(), f'{epoch_dir}/discrim_optim_config.pickle')

    return