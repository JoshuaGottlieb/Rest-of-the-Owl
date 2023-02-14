import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.applications.vgg16 import VGG16
import pickle

def pickle_obj(obj, file):
    '''
    Pickles an object.
    
    obj: Object, object to to be pickled.
    file: str, file path denoting the destination of pickling.
    '''
    
    with open(file, 'wb') as f:
        pickle.dump(obj, f)
        
    return

def unpickle(file):
    '''
    Unpickles an object. Returns the unpickled object.
    
    file: str, file path of pickled object.
    '''
    
    with open(file, 'rb') as f:
        obj = pickle.load(f)
        
    return 

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

def reload_model_from_epoch(epoch_dir, model_type, base_path = '..'):
    '''
    Reloads a model and all of its components from a given epoch. Returns a list of components.
    
    epoch_dir: str, denoting the directory containing the saved model information.
    model_type: "autopainter" or "pix2pix", denoting the type of model to reload.
    base_path: str, denoting the base path of repository, for use in specifying log and model locations.
    '''
    
    components = []
    
    # Load the generator and discriminator.
    generator = keras.models.load_model(f'{epoch_dir}/generator.h5')
    discriminator = keras.models.load_model(f'{epoch_dir}/discriminator.h5')
    components.append(generator)
    components.append(discriminator)

    # Load optimizer configuration dictionaries.
    gen_config = unpickle(f'{epoch_dir}/gen_optim_config.pickle')
    discrim_config = unpickle(f'{epoch_dir}/discrim_optim_config.pickle')

    # Load the optimizers from the dictionaries.
    generator_optimizer = Adam(1e-4).from_config(gen_config)
    discriminator_optimizer = Adam(1e-4).from_config(discrim_config)
    components.append(generator_optimizer)
    components.append(discriminator_optimizer)

    # Load model specific components.
    if model_type == 'pix2pix':
        loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits = True)
        components.append(loss_obj)
    elif model_type == 'autopainter':
        net = VGG16()
        components.append(net)

    # Configure log file path and model directory path.
    log_file = f'{base_path}/logs/{model_type}/epoch_data.csv'
    model_dir = f'{base_path}/models/{model_type}'
    components.append(log_file)
    components.append(model_dir)

    return components