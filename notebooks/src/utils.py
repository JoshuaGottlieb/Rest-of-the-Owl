import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossEntropy
from tensorflow.keras.applications.vgg16 import VGG16
import pickle

def generate_images(model, test_input, tar):
    '''
    Generates an image from a model given a test input.
    Displays the input, the ground-truth image, and generated image side-by-side.
    
    model: tf.Model, a trained model to use to generate an image.
    test_input: 4D Tensor, containing a sketch.
    tar: 4D Tensor, containing a sketch.
    '''
    
    # Generate predicted image in training mode.
    prediction = model(test_input, training = True)
    
    # Create figure.
    plt.figure(figsize = (15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    # Plot images.
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i][:, : , 0], cmap = 'gray')
        plt.axis('off')
    plt.show()
    
    return

def generate_and_save_images(model, epoch, test_input, model_name, save = False, base_path = '..'):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training = False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    if save:
        plt.savefig(f'{base_path}/logs/{model_name}/images/image_at_epoch_{epoch:04d}.png')
    
    plt.show()
    
    return

def reload_model_from_epoch(epoch_dir, model_type):
    components = []
    
    generator = load_model(f'{epoch_dir}/generator.h5')
    discriminator = load_model(f'{epoch_dir}/discriminator.h5')
    
    components.append(generator)
    components.append(discriminator)

    with open(f'{epoch_dir}/gen_optim_config.pickle', 'rb') as f:
        gen_config = pickle.load(f)
    with open(f'{epoch_dir}/discrim_optim_config.pickle', 'rb') as f:
        discrim_config = pickle.load(f)

    generator_optimizer = Adam(1e-4).from_config(gen_config)
    discriminator_optimizer = Adam(1e-4).from_config(discrim_config)
    components.append(generator_optimizer)
    components.append(discriminator_optimizer)

    if model_type == 'pix2pix':
        loss_obj = BinaryCrossentropy(from_logits = True)
        components.append(loss_obj)
    elif model_type == 'autopainter':
        net = VGG16()
        components.append(net)

    log_file = f'{base_path}/logs/{model_type}/epoch_data.csv'
    model_dir = f'{base_path}/models/{model_type}'
    
    components.append(log_file)
    components.append(model_dir)

    return components