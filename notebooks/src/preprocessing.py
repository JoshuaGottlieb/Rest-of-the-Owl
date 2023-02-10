import os
import shutil
from PIL import Image, ImageEnhance, ImageFilter
from PIL import ImageOps
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import cv2
from difPy import dif
import random

def remove_bad_pictures(directory, good = False):
    '''
    Selects good pictures from a directory, using text files present in the directory indicating good/bad pictures.
    Moves the good pictures to data/raw/actual_owls/named.
    
    directory: str, representing the folder path containing text files containing bad image indices
               and subfolder(s) with images.
    good: bool, denoting whether the text files contain good image indices or bad image indices.
          Default False, indicating that the text files contain bad image indices.
    '''
    
    # Extract all files and subdirectories from the directory.
    directory_files = os.listdir(directory)
    text_files = [x for x in directory_files if x.endswith('.txt')]
    subdir = [x for x in directory_files if not x.endswith('.txt')]
    
    # For each subdirectory, extract the image names.
    # Compare the image indices to the indices in each text file and assign good images accordingly.
    for folder in subdir:
       # Image names in subdirectory.
        folder_files = os.listdir(directory + '/' + folder)
        
        for text_file in text_files:
            with open(directory + '/' + text_file, 'r') as f:
                owls_list = f.readlines()
            
            # Convert each entry in text file to zero filled 3 digit entries (e.g. 001 - 999).
            owls_list_str = [f'{int(x[:-1]):03d}' for x in owls_list]
            
            # Extract the images in subdirectory with matching page number of text file.
            page_snip = text_file[0:-4]
            page_files_list = [x for x in folder_files if page_snip in x]
            
            # If good, matching indices are good images, else, they are bad images.
            if good:
                good_files_list = [x for x in page_files_list if any(f'Image_{y}' in x
                                                    for y in owls_list_str)]
            else:
                good_files_list = [x for x in page_files_list if not any(f'Image_{y}' in x
                                                                     for y in owls_list_str)]
            
            # Move good images from subdirectory to raw/actual_owls.
            for image in good_files_list:
                new_path = '../data/raw/actual_owls/named' + image
                shutil.move(directory + '/' + folder + '/' + image, new_path)
            
    return

def crop_fineart_watermarks(directory):
    '''
    Removes watermarks from images scraped from fineartamerica.com.
    Moves processed images to raw/actual_owls/unaltered.
    
    directory: str, representing the folder path containing images to crop.
    '''
    
    # Extract all images from directory, separate into fineartamerica images and other.
    owls_list = os.listdir(directory)
    fineart_owls = [x for x in owls_list if 'FineArtAmerica' in x]
    other_owls = [x for x in owls_list if x not in fineart_owls]
    
    end_dir = '../data/raw/actual_owls/unaltered'
    
    # For each fineart image, crop to 80% original shape (ratio selected as largest ratio that removes all watermarks).
    # Move cropped images and non-fineart images to new directory.
    for image in fineart_owls:
        img = Image.open(directory + '/' + image)
        img_array = np.array(img)
        img = img.crop((0, 0, img_array.shape[1], 0.8 * img_array.shape[0]))
        img.save(end_dir + '/' + image)
    for image in other_owls:
        shutil.copy(directory + '/' + image, end_dir + '/' + image)
    
    return

def rename_owl_files(directory):
    '''
    Renames all files in a directory to a uniform naming scheme. Renames in place.
    
    directory: str, representing the folder path containing images to rename.
    '''
    
    # Extract images, create names.
    unaltered_owls = os.listdir(directory)
    unaltered_owls_renamed = ['Owl_Unaltered_{:04d}.jpg'.format(x) for x in range(1, len(unaltered_owls) + 1)]
    
    # Rename files.
    for i in range(len(unaltered_owls)):
        os.rename(directory + unaltered_owls[i], directory + unaltered_owls_renamed[i])
        
    return

def convert_to_grayscale(init_dir, end_dir):
    '''
    Converts all files from init_dir into grayscale images and writes to end_dir.
    
    init_dir: str, representing the folder path containing images to convert to grayscale.
    end_dir: str, representing the folder path to write converted images to.
    '''
    
    # Get images and construct file names.
    init_dir_files = sorted(os.listdir(init_dir))
    end_dir_files = ['Owl_Grayscale_{}.jpg'.format(x[-8:-4]) for x in init_dir_files]
    
    # Convert and save images in end directory.
    for index, image in enumerate(init_dir_files):
        img = Image.open(init_dir + '/' + image).convert('L')
        img.save(end_dir + '/' + end_dir_files[index])
        
    return

def invert_images(init_dir, end_dir):
    '''
    Inverts color values for all files from init_dir into and writes to end_dir.
    
    init_dir: str, representing the folder path containing images to convert to grayscale.
    end_dir: str, representing the folder path to write converted images to.
    '''
    
    # Get images and construct file names
    images = sorted(os.listdir(init_dir))
    end_dir_files = ['Owl_Inverted_{}.jpg'.format(x[-8:-4]) for x in images]
    
    # Invert and save images in end directory
    for index, image in enumerate(images):
        im = ImageOps.invert(Image.open(directory + '/' + image))
        im.save(end_dir + '/' + end_dir_files[index])
        
    return

def dog(img, size = (0,0), k = 1.6, sigma = 0.5, gamma = 1):
    '''
    Applies difference of Gaussians (DoG) to an image. Copied from https://github.com/heitorrapela/xdog
    Returns difference of image arrays modified by cv2.GaussianBlur transformations.
    
    img: numpy array, array representation of image transform.
    size: 2-tuple, kernel size of matrix to pass to cv2.GaussianBlur. Default (0, 0).
    k: float, multiplier to apply to standard deviation of kernel along X-direction for second GaussianBlur. Default 1.6.
    sigma: float, standard deviation of kernel along X-direction to pass to cv2.GaussianBlur. Default 0.5.
    gamma: float, multiplier to apply to second GaussianBlur when differencing. Default 1.
    '''
    
    img1 = cv2.GaussianBlur(img, size, sigma)
    img2 = cv2.GaussianBlur(img, size, sigma * k)
    
    diff = img1 - (gamma * img2)
    
    return img1, img2, diff

def xdog_garygrossi(img, sigma = 0.5, k = 200, gamma = 0.98, epsilon = 0.1, phi = 10):
    '''
    Applies extended difference of Gaussians (XDoG) of an image. Copied from https://github.com/heitorrapela/xdog
    Returns array containing edge "sketch" of image.
    
    img: numpy array, array representation of image to transform.
    sigma: float, standard deviation of kernel along X-direction to pass to dog(). Default 0.5.
    k: float, multiplier to apply to standard deviation of kernel along X-direction to pass to dog(). Default 200.
    gamma: float, multiplier to apply to second GaussianBlur to pass to dog(). Default 0.98.
    epsilon: float, intensity threshold to determine white space. Default 0.1.
    phi: float, multiplier to apply to elements below epsilon threshold. Default 10.
    '''
    
    aux = dog(img, sigma = sigma, k = k, gamma = gamma)[-1] / 255
    
    for i in range(0, aux.shape[0]):
        for j in range(0, aux.shape[1]):
            if(aux[i,j] >= epsilon):
                aux[i,j] = 1
            else:
                ht = np.tanh(phi * (aux[i][j] - epsilon))
                aux[i][j] = 1 + ht

    return aux * 255

def find_percent_whitespace(sketch_arr, white_space_val = 245, fill_space_val = 10):
    '''
    Helper function to find the percentage of white space and fill space in an image.
    Returns 2-tuple (white_space_percentage, fill_space_percentage).
    
    sketch_arr: numpy array, array representation of sketch (or image) to process.
    white_space_val: int, minimum level of intensity required for a pixel to count as white space. Default 245.
    fill_space_val: int, maximum level of intensity required for a pixel to count as fill space. Default 10.
    '''
    
    # Apply masks and sum to obtain pixel counts for white space and fill space. Calculate image size.
    white_space = (sketch_arr > white_space_val).sum()
    fill_space = (sketch_arr < fill_space_val).sum()
    sketch_size = sketch_arr.shape[0] * sketch_arr.shape[1]
    
    # Calculate white space percent and fill space percent.
    white_space_per = white_space / sketch_size
    fill_space_per = fill_space / sketch_size
    
    return white_space_per, fill_space_per

def find_sketch_threshold(image, white_space_val = 245, fill_space_val = 10, fill_space_threshold = 0.03):
    '''
    Takes in an image and creates a sketch of detail level specified by fill_space_threshold.
    Returns None if no suitable sketch found, otherwise returns an image of a sketch.
    
    image: PIL.Image.Image, an image from which to create a sketch.
    white_space_val: int, minimum level of intensity required for a pixel to count as white space.
                     Passed to xdog_garygrossi(). Default 245.
    fill_space_val: int, maximum level of intensity required for a pixel to count as fill space.
                    Passed to xdog_garygrossi(). Default 10.
    fill_space_threshold: float, percentage of image as fill space indicating the detail threshold of the sketch.
                          Default 0.03.
    '''
    
    
    # Initialize an array of gamma thresholds from 0.2 to 0.8 inclusive.
    sketch_thresholds = np.array(list(range(20, 81))) / 100
    
    # Initialize variables.
    sketch_to_return = None
    thresh_to_return = 0
    white_space_per_to_return = 100
    fill_space_per_to_return = 0
    
    # For each threshold, create an XDoG sketch, calculate white space percentage and fill space percentage.
    # If the fill space percentage is above the input fill space threshold, then stop, image is at sufficient detail.
    for thresh in sketch_thresholds:
        # Test gamma level.
        print(f'Testing gamma {thresh:0.2f}')
        sketch_arr = np.uint8(xdog_garygrossi(np.array(image), sigma = 0.5, k = 200,
                                              gamma = thresh, epsilon = 0.1, phi = 10))
        white_space_per, fill_space_per = find_percent_whitespace(sketch_arr, white_space_val, fill_space_val)
        # If the fill space is too high, break, closest sketch found.
        if fill_space_per > fill_space_threshold:
            print('Too much fill space, terminating.')
            break

        # Update values for sketch at current thresh.
        sketch_to_return = sketch_arr
        white_space_per_to_return = white_space_per
        fill_space_per_to_return = fill_space_per
        thresh_to_return = thresh
        
        print(f'White space: {white_space_per_to_return:02f}, Fill space: {fill_space_per_to_return:02f}')
        
    # If no sketch is found (because the lowest gamma threshold created too detailed of a sketch), return None for sketch.
    # Else, return the image version of the sketch.
    if sketch_to_return is not None:    
        sketch = Image.fromarray(sketch_to_return)
    else:
        sketch = None
    
    return sketch, thresh_to_return, white_space_per_to_return, fill_space_per_to_return

def create_sketches(directory, dest_tail, white_space_val = 245, fill_space_val = 10, fill_space_threshold = 0.03):
    '''
    Create sketches for all images in a given directory. Writes the sketches to data/raw/actual_owls/sketched/dest_tail.
    
    directory: str, folder path to directory containing images to create sketches of.
    dest_tail: str, denotes the subdirectory of data/raw/actual_owls/sketched/ to write to.
    white_space_val: int, minimum level of intensity required for a pixel to count as white space.
                     Passed to find_sketch_threshold(). Default 245.
    fill_space_val: int, maximum level of intensity required for a pixel to count as fill space.
                    Passed to find_sketch_threshold(). Default 10.
    fill_space_threshold: float, percentage of image as fill space indicating the detail threshold of the sketch.
                          Passed to find_sketch_threshold(). Default 0.03.
    '''
    
    # Retrieve images, create end directory string, create sketch image names.
    images = sorted(os.listdir(directory))
    end_dir = f'../data/raw/actual_owls/sketched/{dest_tail}'
    end_dir_files = [f'Owl_Sketched_{dest_tail.title()}_{x[-8:-4]}.jpg' for x in images]
    
    # For each image, find the best sketch and write sketch information to a log.
    # If a sketch is found, save the sketch to the end directory.
    for index, image in enumerate(images):
        # Open the image.
        im = Image.open(directory + '/' + image)
        image_num = image[-8:-4]
        
        print(f'Finding best sketch for {image_num}')
        
        # Find the best sketch for image.
        sketch, thresh, white_space_per, fill_space_per = \
                find_sketch_threshold(im, white_space_val = white_space_val, fill_space_val = fill_space_val,
                                      fill_space_threshold = fill_space_threshold)
        
        # Create log string.
        info_string = f'Image: {image_num}, Threshold: {thresh:0.2f}, '\
                      + f'White Space: %{white_space_per:0.2f}, Fill Space: %{fill_space_per:0.2f}'
        
        # Save sketch if found.
        if sketch is not None:
            print(f'Saving sketch for {image_num}')
            sketch.save(end_dir + '/' + end_dir_files[index])
        else:
            print(f'No suitable sketch found for {image_num}')
        
        # Write log info.
        with open(f'../data/raw/actual_owls/sketched/{dest_tail}_sketch_info.txt', 'a') as f:
                f.write(info_string)

    return

def get_black_border_percentage(image, border = 0.2, threshold = 40):
    '''
    Helper function to obtain how much of an image's border is black.
    Returns a float representing the black border percentage.
    
    image: PIL.Image.Image, image to process.
    border: float, fraction of image to use as border. Default 0.2.
    threshold: int, maximum intensity level of a pixel to be considered a "black" pixel. Default 40.
    '''
    
    # Get border boundaries.
    left, top, right, bottom = ImageOps._border((image.width * border, image.height * border))
    left_bound, top_bound, right_bound, bot_bound =\
                    (int(left), int(top), int(image.size[0] - right), int(image.size[1] - bottom))
    
    # Initialize empty list.
    pixel_val = []
    
    # Iterate across border boundaries, getting pixel values.
    for x in range(0, left_bound):
        for y in range(top_bound, bot_bound):
            pixel_val.append(image.getpixel((x,y)))
    for x in range(right_bound, image.width):
        for y in range(top_bound, bot_bound):
            pixel_val.append(image.getpixel((x,y)))
    for y in range(0, top_bound):
        for x in range(left_bound, right_bound):
            pixel_val.append(image.getpixel((x,y)))
    for y in range(bot_bound, image.height):
        for x in range(left_bound, right_bound):
            pixel_val.append(image.getpixel((x,y)))

    # Convert to numpy array for masking.
    pixel_val_arr = np.array(pixel_val)
    
    return (pixel_val_arr < threshold).sum() / pixel_val_arr.size

def get_bad_sketches_by_border(directory, border = 0.2, intensity_threshold = 40, percentage_threshold = 0.4):
    '''
    Calculates black border percentage of all images in directory.
    Returns a list of str representing image indices with too much black border.
    
    border: float, fraction of image to use as border. Passed to get_black_border_percentage(). Default 0.2.
    intensity_threshold: int, maximum intensity level of a pixel to be considered a "black" pixel.
                         Passed to get_black_border_percentage(). Default 40.
    percentage_threshold: float, threshold denoting fraction of border percentage as "black" to be considered bad.
                          Default 0.4.
    '''
    
    # Get images.
    images = sorted(os.listdir(directory))
    
    bad_images = []
    
    # For each image, calculate the black border percentage. If above threshold, append to list.
    for image in images:
        im = Image.open(directory + '/' + image)
        im_border_percentage = get_black_border_percentage(im, border, intensity_threshold)
        
        if im_border_percentage >= percentage_threshold:
            bad_images.append(image)
    
    # Extract only indices from bad image names.
    bad_images_int = [int(x[-8:-4]) for x in bad_images]
    
    return bad_images_int

def get_bad_sketches_by_df(log_file):
    '''
    Extracts a list of bad image indices from a log file based on preset criteria.
    
    log_file: str, file path of log_file to process.
    '''
    
    # Read file
    with open(log_file) as f:
        log = f.read()

    # Perform some basic string cleaning to separate log entries.
    log = re.sub(r'(\d)(I)', r'\g<1>\n\g<2>', log).split('\n')
    log_split = [x.split(',') for x in log]
    log_split_float = [[float(re.search(r'\d+\.?\d*', x)[0]) for x in row] for row in log_split]
    
    # Create dataframe of log entries.
    log_df = pd.DataFrame(log_split_float, columns = ['image', 'gamma', 'white_space', 'fill_space'])
    
    # Extract images with a gamma threshold of 0 (no sketch found)
    # with a fill space less than 0.02 (too low detail), or with a white space less than 0.5 (too much black space)
    bad_images = log_df.loc[(log_df.gamma == 0) | (log_df.fill_space < 0.02)
                            | (log_df.white_space < 0.5)].image.values.tolist()
    
    # Extract indices from string names
    bad_images_int = [int(x) for x in bad_images]
    
    return bad_images_int

def select_images_and_sketches(sketch_directories, image_directories, log_files, border = 0.2,
                               intensity_threshold = 40, percentage_threshold = 0.4):
    '''
    Given a list of sketch directories, determines which sketch/image pair should be used (regular vs. inverted) and
    moves the good sketch/image pairs to a new directory, renaming them in preparation for stitching.
    
    sketch_directories: list of str, denoting the folder paths for the regular and inverted sketches.
                        Should have length 2.
    image_directories: list of str, denoting the folder paths for the regular and inverted images.
                       Should have the same length and order as the sketch_directories to ensure proper matching.
    log_files: list of str, denoting the file paths for the log files for the regular and inverted sketches.
               Should have the same length and order as the sketch_directories to ensure proper matching.
    border: float, fraction of image to use as border. Passed to get_bad_sketches_by_border(). Default 0.2.
    intensity_threshold: int, maximum intensity level of a pixel to be considered a "black" pixel.
                         Passed to get_bad_sketches_by_border(). Default 40.
    percentage_threshold: float, threshold denoting fraction of border percentage as "black" to be considered bad.
                          Passed to get_bad_sketches_by_border(). Default 0.4.
    '''
    
    # Initialize a dictionary of bad images.
    bad_image_dict = {}
    
    # Extract the directory tails for each directory (regular vs. inverted).
    directory_tails = [re.search('/[A-Za-z]+$', directory)[0] for directory in sketch_directories]
    
    # Find bad images by border for each of the sketch directories.
    for index, directory in enumerate(sketch_directories):
        print(f'Getting bad sketches for {directory} by border')
        
        # Convert to set for later use in set operations.
        bad_images_by_border = set(get_bad_sketches_by_border(directory,
                                                          border = border,
                                                          intensity_threshold = intensity_threshold,
                                                          percentage_threshold = percentage_threshold))
        # Add to dictionary.
        bad_image_dict[directory_tails[index]] = bad_images_by_border
    
    # Find bad images by log file for each of the sketch directories.
    for index, log_file in enumerate(log_files):
        print(f'Getting bad sketches for {log_file} by log file')
        
        # Convert to set for later use in set operations.
        bad_images_by_sketch = set(get_bad_sketches_by_df(log_file))
        
        # Append to dictionary at proper key value using union.
        bad_image_dict[directory_tails[index]] = bad_image_dict[directory_tails[index]].union(bad_images_by_sketch)

    # Take the intersection of both dictionary entries so that images which are bad in both variants are discarded.
    bad_both = bad_image_dict[directory_tails[0]].intersection(bad_image_dict[directory_tails[1]])
    
    # Update the dictionary entries so that each is disjoint.
    bad_image_dict[directory_tails[0]] -= bad_both
    bad_image_dict[directory_tails[1]] -= bad_both
    
    # For each directory, copy the image files which are not labeled "bad" from that directory.
    for index, directory in enumerate(image_directories):
        print(f'Copying images for {directory}')
        
        # Get images from directory.
        images = sorted(os.listdir(directory))
        
        # Select the good images.
        good_images = [x for x in images if int(x[-8:-4]) not in bad_image_dict[directory_tails[index]]\
                                            and int(x[-8:-4]) not in bad_both]
        
        # Create the destination names.
        end_dir_images = '../data/raw/actual_owls/selected'
        end_dir_image_names = [f'Owl_Selected_{x[-8:-4]}.jpg' for x in good_images]
        
        # Copy the good images to the final destination.
        for ix, image in enumerate(good_images):
            shutil.copy(directory + '/' + image, end_dir_images + '/' +  end_dir_image_names[ix])

    # For each directory, copy the sketch files which are not labeled "bad" from that directory.    
    for index, directory in enumerate(sketch_directories):
        print(f'Copying sketches for {directory}')
        
        # Get sketches from directory.
        sketches = sorted(os.listdir(directory))
        
        # Select the good sketches.
        good_sketches = [x for x in sketches if int(x[-8:-4]) not in bad_image_dict[directory_tails[index]]\
                                             and int(x[-8:-4]) not in bad_both]
        
        # Create the destination names.
        end_dir_sketches = '../data/raw/actual_owls/sketched/selected'
        end_dir_sketch_names = [f'Owl_Sketched_Selected_{x[-8:-4]}.jpg' for x in good_sketches]
        
        # Copy the good sketches to the final destination.
        for ix, sketch in enumerate(good_sketches):
            shutil.copy(directory + '/' + sketch, end_dir_sketches + '/' +  end_dir_sketch_names[ix])
            
    return

def remove_duplicates(image_dir, sketch_dir, duplicate_dictionary = None, recheck = False):
    '''
    Checks for duplicate image files using difPy, copying unique entries to new folders.
    https://github.com/elisemercury/Duplicate-Image-Finder
    
    image_dir: str, denoting the folder path containing the images.
    sketch_dir: str, denoting the folder path containing the sketches.
    duplicate_dictionary: OrderedDictionary or None, previous results of a difPy search.
                          If None, difPy is run to generate dictionary. Default None.
    recheck: bool, denotes whether to rerun the difPy process a second time, as sometimes additional duplicates
             are found after first pass. Default False.
    '''
    
    # If no duplicate dictionary is provided, run difPy to generate duplicate dictionary.
    # difPy is ran with default parameters.
    if duplicate_dictionary is None:
        print('Checking for duplicates.')
        duplicate_dict = dif(image_dir).result
    else:
        duplicate_dict = duplicate_dictionary
    
    duplicates = []
    
    # Extract all names of duplicates.
    for key in duplicate_dict.keys():
        for img in duplicate_dict[key]['duplicates']:
            duplicates.append(img)

    # Extract indices of duplicate names.
    duplicate_num = [x[-8:-4] for x in duplicates]
    
    print('Copying images.')
    
    # Extract names of files from image directory.
    images = sorted(os.listdir(image_dir))
    
    # Select unique entries.
    non_duplicate_images = [x for x in images if (x[-8:-4]) not in duplicate_num]

    # Create destination file names.
    end_dir_images = '../data/raw/actual_owls/no_duplicates'
    end_dir_image_names = [f'Owl_NoDuplicates_{x[-8:-4]}.jpg' for x in non_duplicate_images]

    # Copy to end directory.
    for ix, image in enumerate(non_duplicate_images):
        shutil.copy(image_dir + '/' + image, end_dir_images + '/' +  end_dir_image_names[ix])
        
    print('Copying sketches.')
    
    # Extract names of sketches from sketch directory.
    sketches = sorted(os.listdir(sketch_dir))
    
    # Select unique entries.
    non_duplicate_sketches = [x for x in sketches if (x[-8:-4]) not in duplicate_num]

    # Create destination files names.
    end_dir_sketches = '../data/raw/actual_owls/sketched/no_duplicates'
    end_dir_sketch_names = [f'Owl_Sketched_NoDuplicates_{x[-8:-4]}.jpg' for x in non_duplicate_sketches]

    # Copy to end directory.
    for ix, sketch in enumerate(non_duplicate_sketches):
        shutil.copy(directory + '/' + sketch, end_dir_sketches + '/' +  end_dir_sketch_names[ix])
    
    # If recheck is true, rerun difPy with default parameters on the images again, with previously found duplicates removed.
    # Remove any further duplicates in place.
    if recheck:
        print('Checking for more duplicates.')
        
        # Generate duplicate dictionary from difPy.
        duplicate_dict = dif(end_dir_images).result
        duplicates = []
    
        # Extract all names of duplicates
        for key in duplicate_dict.keys():
            for img in duplicate_dict[key]['duplicates']:
                duplicates.append(img)

        # Extract indices of duplicate names.
        duplicate_num = [x[-8:-4] for x in duplicates]
        
        # Remove duplicates found in place.
        print('Removing rechecked duplicates.')
        images = sorted(os.listdir(end_dir_images))
        sketches = sorted(os.listdir(end_dir_sketches))
        rechecked_duplicate_images = [x for x in images if (x[-8:-4]) in duplicate_num]
        rechecked_duplicate_sketches = [x for x in sketches if (x[-8:-4]) in duplicate_num]
        
        for ix, image in enumerate(rechecked_duplicate_images):
            os.remove(end_dir_images + '/' +  rechecked_duplicate_images[ix])
        for ix, sketch in enumerate(rechecked_duplicate_sketches):
            os.remove(end_dir_sketches + '/' +  rechecked_duplicate_sketches[ix])
        
    return

def resize_image(image_path, size):
    '''
    Resizes image to desired size ratio with zeros padding. Returns a padded image of the specified size.
    Adapted from https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
    
    image_path: str, denoting the path to the image file to resize.
    size: 2-tuple, denoting the (width, height) dimensions to resize the image to.
    '''
    
    im = Image.open(image_path)
    old_size = im.size
    
    ratio = float(max(size))/max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    
    im = im.resize(new_size, Image.LANCZOS)
    
    new_im = Image.new("L", (size))
    new_im.paste(im, ((size[0] - new_size[0]) // 2, (size[1] - new_size[1]) // 2))
    
    delta_w = size[0] - new_size[0]
    delta_h = size[1] - new_size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    new_im = ImageOps.expand(im, padding)
    
    return new_im

def resize_images(directory, size):
    '''
    Resizes all images in a directory to the specified size.
    
    directory: str, denoting the folder path of the directory to process.
    size: 2-tuple, denoting the (width, height) dimensions to resize each image to. Passed to resize_image().
    '''
    
    # Extract images from directory
    images = sorted(os.listdir(directory))
    
    # Create destination files names.
    end_dir = '../data/raw/actual_owls/resized'
    end_dir_files = [f'Owl_Resized_{x[-8:-4]}.jpg' for x in images]
    
    # Save resized images to destination.
    for index, image in enumerate(images):
        im = resize_image(directory + '/' + image, size)
        im.save(end_dir + '/' + end_dir_files[index])
        
    return

def resize_sketches(directory, size):
    '''
    Resizes all sketches in a directory to the specified size.
    
    directory: str, denoting the folder path of the directory to process.
    size: 2-tuple, denoting the (width, height) dimensions to resize each sketch to. Passed to resize_image().
    '''
    
    # Extract sketches from directory
    sketches = sorted(os.listdir(directory))
    
    # Create destination files names.
    end_dir = '../data/raw/actual_owls/sketched/resized'
    end_dir_files = [f'Owl_Sketched_Resized_{x[-8:-4]}.jpg' for x in sketches]
    
    # Save resized sketches to destination.
    for index, sketch in enumerate(sketches):
        im = resize_image(directory + '/' + sketch, size)
        im.save(end_dir + '/' + end_dir_files[index])
        
    return

def get_concat_h_cut(im1, im2):
    '''
    Takes in two images and creates a horizontally concatenated image of the two. Returns the concatenated image.
    Adapted from https://note.nkmk.me/en/python-pillow-concat-images/
    
    im1: PIL.Image.Image, image to put on left side of the horizontal concatenated image.
    im2: PIL.Image.Image, image to put on right side of the horizontal concatenated image.
    '''
    
    dst = Image.new('L', (im1.width + im2.width, min(im1.height, im2.height)))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    
    return dst

def create_train_test_split(test_size = 0.25, random_state = 42):
    '''
    Creates a train-test split of the sketches and images, concatenates the sketches and images together, and
    copies the result into train and test folders.
    
    test_size: float, fraction of dataset to use as test data. Default 0.25.
    random_state: int, random state to use for initializing seed. Needed for reproducibility. Default 42.
    '''
    
    # Set seed for random state.
    random.seed(random_state)
    
    # Extract all indices.
    indices = sorted([x[-8:-4] for x in os.listdir('../data/raw/actual_owls/resized')])
    
    # Randomly select test and train indices.
    test_indices = random.sample(indices, int(test_size * len(indices)))
    train_indices = [x for x in indices if x not in test_indices]
    
    # For each index in test indices, extract the image and sketch, concatenate, and save to test folder.
    for index in test_indices:
        im1 = Image.open(f'../data/raw/actual_owls/resized/Owl_Resized_{index}.jpg')
        im2 = Image.open(f'../data/raw/actual_owls/sketched/resized/Owl_Sketched_Resized_{index}.jpg')
        im_concat = get_concat_h_cut(im1, im2)
        im_concat.save(f'../data/test/Owl_Pair_{index}.jpg')
        
    # For each index in train indices, extract the image and sketch, concatenate, and save to train folder.
    for index in train_indices:
        im1 = Image.open(f'../data/raw/actual_owls/resized/Owl_Resized_{index}.jpg')
        im2 = Image.open(f'../data/raw/actual_owls/sketched/resized/Owl_Sketched_Resized_{index}.jpg')
        im_concat = get_concat_h_cut(im1, im2)
        im_concat.save(f'../data/train/Owl_Pair_{index}.jpg')
    
    return