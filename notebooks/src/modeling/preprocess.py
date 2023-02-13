import tensorflow as tf

def load(image_file):
    '''
    Loads a sketch/image pair from a file, spliting the pair. Returns the sketch and image separated.
    
    image_file: str, path denoting location of sketch/image pair file.
    '''
    
    # Read and decode an image file to a uint8 tensor
    pair = tf.io.read_file(image_file)
    pair = tf.io.decode_jpeg(pair)
    
    w = tf.shape(pair)[1]
    w = w // 2
    
    # Split image and sketch
    sketch = pair[:, w:, :]
    image = pair[:, :w, :]

    # Convert both images to float32 tensors
    sketch = tf.cast(sketch, tf.float32)
    image = tf.cast(image, tf.float32)

    return sketch, image

def resize(image, height, width):
    '''
    Resizes sketch and image with padding to specified height and width.
    
    image: 3D tensor, representing the image information in tensor form.
    height: int, representing the desired height in pixels.
    width: int, representing the desired width in pixels.
    '''
    
    image_resized = tf.image.resize_with_pad(image, height, width, method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return image_resized

# Normalizing the images to [0, 1]
def normalize(image):
    '''
    Scales the images so that pixel values fall in [0, 1] instead of [0, 255].
    
    image: 3D tensor, representing the image information in tensor form.
    '''
    
    image_scaled = image / 255

    return image_scaled

def load_image_and_sketch(image_file, height = 256, width = 256):
    '''
    Loads a sketch/image pair from a file, performing resizing and scaling.
    
    image_file: str, path denoting location of sketch/image pair file.
    '''
    
    sketch, image = load(image_file)
    sketch = resize(sketch, height, width)
    sketch = normalize(sketch)
    image = resize(image, height, width)
    image = normalize(image)

    return sketch, image