from tensorflow.io import read_file, decode_jpeg
from tensorflow import cast, shape
from tensorflow.image import resize, resize_with_pad, ResizeMethod

def load(image_file):
    '''
    Loads a sketch/image pair from a file, spliting the pair. Returns the sketch and image separated.
    
    image_file: str, path denoting location of sketch/image pair file.
    '''
    
    # Read and decode an image file to a uint8 tensor
    pair = read_file(image_file)
    pair = decode_jpeg(pair)
    
    w = shape(pair)[1]
    w = w // 2
    
    # Split image and sketch
    image = pair[:, w:, :]
    sketch = pair[:, :w, :]

    # Convert both images to float32 tensors
    sketch = cast(sketch, tf.float32)
    image = cast(image, tf.float32)

    return sketch, image

def resize(sketch, image, height, width):
    '''
    Resizes sketch and image with padding to specified height and width.
    
    sketch: numpy array, representing the sketch information in array form.
    image: numpy array, representing the image information in array form.
    height: int, representing the desired height in pixels.
    width: int, representing the desired width in pixels.
    '''
    
    sketch_resized = resize_with_pad(sketch, height, width, method = ResizeMethod.NEAREST_NEIGHBOR)
    image_resized = resize_with_pad(image, height, width, method = ResizeMethod.NEAREST_NEIGHBOR)

    return sketch_resized, image_resized

# Normalizing the images to [0, 1]
def normalize(sketch, image):
    '''
    Scales the images so that pixel values fall in [0, 1] instead of [0, 255].
    
    sketch: numpy array, representing the sketch information in array form.
    image: numpy array, representing the image information in array form.
    '''
    
    sketch_scaled = sketch / 255
    image_scaled = image / 255

    return sketch_scaled, image_scaled

def load_image_and_sketch(image_file):
    '''
    Loads a sketch/image pair from a file, performing resizing and scaling.
    
    image_file: str, path denoting location of sketch/image pair file.
    '''
    
    sketch, image = load(image_file)
    sketch, image = resize(sketch, image, 256, 256)
    sketch, image = normalize(sketch, image)

    return sketch, image