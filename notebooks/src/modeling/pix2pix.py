import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import time
from IPython import display
import numpy as np
from tf.data import Dataset
from .utils import generate_images

def generator_loss_pix2pix(disc_generated_output, gen_output, target, loss_object):
    '''
    Calculates the generator loss for the pix2pix model.
    Returns the total generator loss, the generator GAN loss, and the generator L1 loss.
    Adapted from https://www.tensorflow.org/tutorials/generative/pix2pix
    
    disc_generated_output: 3D or 4D tensor, the output of the discriminator from the generated image(s).
    gen_output: 3D or 4D tensor, the output of the generator (the generated image(s)).
    target: 3D or 4D tensor, the ground-truth image(s).
    loss_object: keras.losses.BinaryCrossEntropy object.
    '''
    
    # Calculate GAN loss using binary cross-entropy
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # L1 loss - mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    # Total gen loss
    total_gen_loss = gan_loss + (100 * l1_loss)

    return total_gen_loss, gan_loss, l1_loss

def discriminator_loss_pix2pix(disc_real_output, disc_generated_output, loss_object):
    '''
    Calculates the discriminator loss for the pix2pix model. Returns the total discriminator loss.
    Adapted from https://www.tensorflow.org/tutorials/generative/pix2pix
    
    disc_real_output: 3D or 4D tensor, the output of the discriminator from the ground-truth image(s).
    disc_generated_output: 3D or 4D tensor, the output of the discriminator from the generated image(s).
    loss_object: keras.losses.BinaryCrossEntropy object.
    '''
    
    # Real loss given by binary cross-entropy of ones-like pixels in the real image
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    # Generated loss given by binary cross-entropy of zeros-like pixels in the generated image
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    # Total discriminator loss
    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

def train_step_pix2pix(input_image, target, generator, discriminator, gen_optimizer, discrim_optimizer, loss_obj):
    '''
    Train step for the pix2pix model. Returns losses.
    Adapted from https://www.tensorflow.org/tutorials/generative/pix2pix
    
    input_image: 3D or 4D tensor, sketch to input into generator.
    target: 3D or 4D tensor, ground-truth image.
    generator: tf.Model, generator model to use and update.
    discriminator: tf.Model, discriminator model to use and update.
    gen_optimizer: tf.keras.optimizers, optimizer to use to update generator model.
    discrim_optimizer: tf.keras.optimizers, optimizer to use to update discriminator model.
    loss_object: keras.losses.BinaryCrossEntropy object.
    '''
    
    # Run sketches and ground-truth images through generator and discriminator and calculate losses.
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss_pix2pix(disc_generated_output, gen_output,
                                                                           target, loss_obj)
        disc_loss = discriminator_loss_pix2pix(disc_real_output, disc_generated_output, loss_obj)

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

    return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss

def fit_pix2pix(train_ds, test_ds, epochs, generator, discriminator,
                gen_optimizer, discrim_optimizer, loss_obj, log_file, model_dir,
                starting_epoch = 0, save = True):
    '''
    Fits a pix2pix model. Adapted from https://www.tensorflow.org/tutorials/generative/pix2pix
    
    train_ds: tf.data.Dataset, the training dataset.
    test_ds: tf.data.Dataset, the test dataset.
    epochs: int, the number of epochs (passes over the entire training dataset) to run the model for.
    generator: tf.Model, generator model to use and update.
    discriminator: tf.Model, discriminator model to use and update.
    gen_optimizer: tf.keras.optimizers, optimizer to use to update generator model.
    discrim_optimizer: tf.keras.optimizers, optimizer to use to update discriminator model.
    loss_object: keras.losses.BinaryCrossEntropy object.
    log_file: str, a path denoting the file to write the losses to. Directory must exist at path.
    model_dir: str, a path denoting the directory to save the model weights and optimizer configurations to.
               Directory must exist at path, but subdirectories do not need to exist at path.
    starting_epoch: int, the epoch to start at. Needed for logging and model-saving purposes. Default 0.
    save: bool, denotes whether to log the losses during training and save models. Default True.
    '''

    # Pull random image, sketch pair from test set to show model progress during training.
    example_target, example_input = next(iter(test_ds.take(1)))
    
    start = time.time()
    epoch_range = range(starting_epoch, starting_epoch + epochs)
    
    # Loop through each epoch.
    for epoch, _ in enumerate(epoch_range):
        display.clear_output(wait=True)

        # Print time to process previous epoch.
        if (epoch + starting_epoch) != starting_epoch:
            print(f'Time taken for epoch: {time.time() - start:.2f} sec\n')

        # Restart timer and display current results of training.    
        start = time.time()
        generate_images(generator, example_input, example_target)
        print(f'Epoch: {epoch + starting_epoch + 1}')
        
        # Initialize empty array for losses.
        train_losses = np.zeros(4)
        batch_size = 0
        
        # Train step.
        for (target, sketch) in train_ds:
            train_losses += train_step_pix2pix(sketch, target,
                                               generator, discriminator,
                                               gen_optimizer, discrim_optimizer, loss_obj)
            batch_size += 1
            
        train_losses = train_losses / batch_size

        if save:
            # Log losses.
            with open(log_file, 'a') as f:
                f.write(f'Epoch: {epoch + starting_epoch + 1}, '
                        + f'gen_total_loss: {train_losses[0]:0.3f}, '
                        + f'gen_gan_loss: {train_losses[1]:0.3f}, '
                        + f'gen_l1_loss: {train_losses[2]:0.3f}, '
                        + f'disc_loss: {train_losses[3]:0.3f}\n')

            # Save models every 10 epochs.        
            if ((epoch + 1) % 10) == 0:
                epoch_dir = f'{model_dir}/epoch_{epoch + starting_epoch + 1:03d}'
                if not os.path.exists(epoch_dir):
                    os.mkdir(epoch_dir)
                generator.save(f'{epoch_dir}/generator.h5')
                discriminator.save(f'{epoch_dir}/discriminator.h5')
                with open(f'{epoch_dir}/gen_optim_config.pickle', 'wb') as f:
                    pickle.dump(gen_optimizer.get_config(), f)
                with open(f'{epoch_dir}/discrim_optim_config.pickle', 'wb') as f:
                    pickle.dump(discrim_optimizer.get_config(), f)

    return