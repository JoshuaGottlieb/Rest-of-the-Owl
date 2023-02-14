import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as pltticker
import seaborn as sns
from PIL import Image, ImageOps
from . import preprocessing as prep
from .model_analysis import generation as gen
# import dataframe_image as dfi

def configure_axislabels_and_title(x_label, y_label, title, ax,
                                   axis_size = 24, title_size = 32,
                                   axis_pad = 10, title_pad = 10, font_name = 'Arial'):
    '''
    Configure the axis and title labels of a matplotlib axis object.
    
    x_label: str or None, to use for labeling the x-axis.
    y_label: str or None, to use for labeling the y-axis.
    title: str or None, to use for the title of the ax object.
    ax: The matplotlib axis object to modify.
    axis_size: float, representing the font size of the axis labels. Default 24.0.
    title_size: float, representing the font size of the title. Default 32.0.
    axis_pad: float, representing the padding between graph and axis labels. Default 10.0.
    title_pad: float, representing the padding between graph and title. Default 10.0.
    font_name: str, representing the name of the font to use for all labels. Default Arial.
    '''
    
    ax.set_xlabel(x_label, fontfamily = font_name, fontsize = axis_size, labelpad = axis_pad)
    ax.set_ylabel(y_label, fontfamily = font_name, fontsize = axis_size, labelpad = axis_pad)
    ax.set_title(title, fontfamily = font_name, fontsize = title_size, pad = title_pad)
    
    return



# Configures ticklabels and tick parameters
def configure_ticklabels_and_params(ax, label_size = 16, length = 8, width = 1, font_name = 'Arial',
                                    x_label_size = None, x_length = None, x_width = None,
                                    y_label_size = None, y_length = None, y_width = None,
                                    format_xticks = False, x_ticks_rounding = 1, x_ticks_rotation = 0,
                                    format_yticks = False, y_ticks_rounding = 1, y_ticks_rotation = 0):
    '''
    Configures the ticklabels and tick sizes of a matplotlib axis object.
    
    ax: Matplotlib axis object to format.
    label_size: float, the font size of the major tick labels. Default 16.
    length: float, the length of the major tick marks. Default 8.
    width: float, the width of the major tick marks. Default 1.
    font_name: str, the font to use for tick labels. Default Arial.
    x_label_size: float, the font size of the major tick labels on the x-axis. By default inherits from label_size.
    x_length: float, the length of the major tick marks on the x-axis. By default inherits from length.
    x_width: float, the width of the major tick marks on the x-axis. By default inherits from width.
    y_label_size: float, the font size of the major tick labels on the y-axis. By default inherits from label_size.
    y_length: float, the length of the major tick marks on the y-axis. By default inherits from length.
    y_width: float, the width of the major tick marks on the y-axis. By default inherits from width.
    format_xticks: bool, indicates whether to format numerical x-tick labels. Default False.
    x_ticks_rounding: nonzero int, the number by which to divide the x-tick labels (e.g. to round to 10s, set to 10).
                      Default 1.
    x_ticks_rotation: int, the number in degrees by which to rotate the x-tick labels counter-clockwise. Default 0.
    format_yticks: bool, indicates whether to format numerical y-tick labels. Default False.
    y_ticks_rounding: int, the number by which to divide the y-tick labels (e.g. to round to 10s, set to 10). Default 1.
    y_ticks_rotation: int, the number in degrees by which to rotate the y-tick labels counter-clockwise. Default 0.
    '''
    
    # Set individual axis attributes
    if x_label_size is None:
        x_label_size = label_size
    if y_label_size is None:
        y_label_size = label_size
    if x_length is None:
        x_length = length
    if y_length is None:
        y_length = length
    if x_width is None:
        x_width = width
    if y_width is None:
        y_width = width
    
    # Set label sizes and tick lengths
    ax.tick_params(axis = 'x', which = 'major', labelsize = x_label_size, length = x_length, width = x_width)
    ax.tick_params(axis = 'y', which = 'major', labelsize = y_label_size, length = y_length, width = y_width)

    # Set font for tick labels on both axes, format tick labels if numerical, rotate if needed
    for tick in ax.get_xticklabels():
        tick.set_fontname("Arial")
        tick.set_rotation(x_ticks_rotation)
        # Rounds tick values and adds commas where appropriate
        if format_xticks:
            ax.get_yaxis().set_major_formatter(pltticker.FuncFormatter(lambda x, p: format(int(x / x_ticks_rounding),',')))

    for tick in ax.get_yticklabels():
        tick.set_fontname("Arial")
        tick.set_rotation(y_ticks_rotation)
        # Rounds tick values and adds commas where appropriate
        if format_yticks:
            ax.get_yaxis().set_major_formatter(pltticker.FuncFormatter(lambda x, p: format(int(x / y_ticks_rounding),',')))
        
    return

def configure_legend(ax, labels = None, loc = 0, bbox_to_anchor = None, labelcolor = 'black', fontsize = 12,
                     frameon = False, facecolor = 'inherit', fancybox = False, framealpha = 0.8,
                     edgecolor = 'inherit', borderpad = 0.4, labelspacing = 0.5):
    '''
    Configures the legend of a matplotlib axis object.
    
    ax: Matplotlib axis object to format.
    labels: list of str, to manually specify the labels to use for each artist. Default None uses automatic labeling.
    loc: int, str, or 2-tuple of float, specifies the location of the legend according to matplotlib documentation.
         Default 0, equivalent to 'best'.
    bbox_to_anchor: 2-tuple or 4-tuple of float, specifies the location of the bounding box of the legend. Default None.
    labelcolor: str or list of str, specifies the color(s) of the labels. Default 'black'.
    fontsize: int, specifies the fontsize of the labels. Default 12.
    frameon: bool, denotes whether the legend should be drawn on a patch (frame). Default False.
    facecolor: 'inherit' or str, denotes the background color of the legend. If 'inherit', match the color of the axis.
               Default 'inherit'.
    fancybox: bool, denotes whether the legend's background should have rounded edges. Default False.
    framealpha: float, denotes the alpha transparency of the legend's background. Default 0.8.
    edgecolor: 'inherit' or str, denotes the legend's background edge color. If 'inherit', match the color of the axis.
               Default 'inherit'.
    borderpad: float, denotes the fractional whitespace inside the legend border, in font-size units. Default 0.4.
    labelspacing: float, denotes the vertical space between the legend entries, in font-size units. Default 0.5.
    '''
    
    # Extract handles for artists
    handles = [i for i in ax.get_legend_handles_labels()][0]
    
    # If there is a bounding box, anchor to bounding box. Else draw legend using input parameters.
    if bbox_to_anchor is not None:
        ax.legend(handles = handles, labels = labels, loc = loc, bbox_to_anchor = None, labelcolor = labelcolor,
                  frameon = frameon, facecolor = facecolor, fontsize = fontsize, fancybox = fancybox,
                  framealpha = framealpha, borderpad = borderpad, labelspacing = labelspacing);
    else:
        ax.legend(handles = handles, labels = labels, loc = loc, labelcolor = labelcolor,
                  frameon = frameon, facecolor = facecolor, fontsize = fontsize, fancybox = fancybox,
                  framealpha = framealpha, borderpad = borderpad, labelspacing = labelspacing);
    
    return

def remove_ticks(ax, x_ticks = True, y_ticks = True):
    '''
    Removes ticks from the axis.
    
    ax: Matplotlib axis object to format.
    x_ticks: bool, denotes whether to remove ticks on the x-axis. Default True.
    y_ticks: bool, denotes whether to remove ticks on the y-axis. Default True.
    '''
    
    if x_ticks:
        ax.get_xaxis().set_ticks([]);
    
    if y_ticks:
        ax.get_yaxis().set_ticks([]);
        
    return

def plot_single_image(image, x_label = None, y_label = None, title = None, title_size = 24, title_pad = 10,
                      axis_pad = 10, axis_size = 16, figsize = (4, 4), axis = None):
    '''
    Plots a single image.
    
    image: PIL.Image.Image, np.array, or other format compatible with plt.imshow, image to plot.
    x_label: str, denoting the label to use along the x-axis. Default None.
    x_label: str, denoting the label to use along the y-axis. Default None.
    title: str, denoting the label to use for the title. Default None.
    axis_size: float, representing the font size of the axis labels. Default 16.0.
    title_size: float, representing the font size of the title. Default 24.0.
    axis_pad: float, representing the padding between graph and axis labels. Default 10.0.
    title_pad: float, representing the padding between graph and title. Default 10.0.
    figsize: 2-tuple of float, representing the size of the figure to create if no axis is passed. Default (4, 4).
    axis: Matplotlib axis object to use. Default None, thus drawing its own Matplotlib figure.
    '''
    
    # If an axis is passed, use axis. Else, create a fig, ax pair to draw on.
    if axis is None:
        fig, ax = plt.subplots(figsize = figsize)
    else:
        ax = axis
    
    # Draw image, configure labels, and remove ticks.
    ax.imshow(image, cmap = 'gray', vmin = 0, vmax = 255);
    configure_axislabels_and_title(x_label, y_label, title, ax = ax,
                                   axis_size = fontsize, axis_pad = labelpad,
                                   title_size = title_size, title_pad = title_pad)
    remove_ticks(ax);

    # Return the fig, ax pair if no axis was passed, otherwise, return nothing.
    if axis is None:
        return fig, ax
    else:
        return
    
    return

def get_example_sketches_at_thresholds(image_file, fill_space_thresholds):
    '''
    Gets example sketches at different fill space thresholds for an image.
    Returns a list of images containing the sketches at varying levels of detail.
    
    image_file: str, denoting the path of the image to create sketches from.
    fill_space_thresholds: list of float in [0, 1], denoting the fill space thresholds to pass to
                           prep.find_sketch_threshold().
    '''
    
    # Load image.
    image = Image.open(image_file)
    
    sketches = []
    
    # Create sketches based on thresholds.
    for threshold in fill_space_thresholds:
        sketches.append(prep.find_sketch_threshold(image, fill_space_threshold = threshold))
        
    return sketches

def display_sketch_thresholds(image_file, fill_space_thresholds, sketch_list):
    '''
    Displays an image along with sketches at various thresholds. Returns fig, ax of plotted images.
    
    image_file: str, denoting the path to the file to use.
    fill_space_thresholds: list of float, to use in labeling various sketches.
                           Should be of the same size and in the same order as sketch_list.
    sketch_list: list of tuples of the form (sketch, gamma, ...). Sketch is a PIL.Image.Image, gamma is a float.
                 Sketches to be plotted. Should be of the same length and order as sketch_list for proper labeling.
    '''
    
    # Draw figure and axis.
    num_images = len(fill_space_thresholds) + 1
    
    fig, ax = plt.subplots(1, num_images, figsize = (3.6 * num_images, 10));

    # Plot sketches.
    for i in range(len(fill_space_thresholds)):
        plot_single_image(sketches[i][0],
                          x_label = f'Threshold {fill_space_thresholds[i]}\nGamma: {sketches[i][1]}',
                          axis_pad = 10, axis_size = 16, axis = ax[i])

    # Plot image.
    plot_single_image(sketches[i][0], x_label = 'Ground-Truth', axis_pad = 10, axis_size = 16, axis = ax[-1])
                   
    # Adjust subplot spacing.
    plt.subplots_adjust(wspace = 0, hspace = 0);
    
    return fig, ax

def display_dog_mixtures(gauss_mixtures, gammas):
    '''
    Plots a set of gaussian mixtures produced by DoG algorithm labeled by their gamma values.
    Returns fig, ax objects of plotted images.
    
    gauss_mixtures: list of np.array, images to plot.
    gammas: list of float, floats to use to label images. Should be of the same size and order as gauss_mixtures.
    '''

    # Draw fig, ax objects.
    fig, ax = plt.subplots(1, 4, figsize = (14, 10));

    # Plot images, removing ticks.
    for i in range(len(gammas)):
        plot_single_image(np.uint8(gauss_mixtures[i]), x_label = f'Gamma: {gammas[i]}',
                                   axis_pad = 10, axis_size = 16, axis = ax[i])

    # Adjust subplot spacing.
    plt.subplots_adjust(wspace = 0, hspace = 0);
    
    return fig, ax

def create_images_with_border(inverted_image, regular_image, border = 0.2):
    '''
    Plots the inverted and regular versions of an image, with colored rectangles indicating the border of the image
    used during preprocessing to determine which version of an image was used. Returns fig, ax of plotted images.
    
    inverted_image: PIL.Image.Image, inverted version of image to plot.
    regular_image: PIL.Image.Image, regular version of image to plot.
    border: float in [0, 0.5], fraction of image to use as border. Passed to get_black_border_percentage().
            Default 0.2, as used in preprocessing for model training.
    '''

    # Calculate border offsets and border boundaries for inverted image.
    left_inv, top_inv, right_inv, bottom_inv = ImageOps._border((inverted_image.width * border,
                                                                 inverted_image.height * border))
    left_bound_inv, top_bound_inv, right_bound_inv, bot_bound_inv =\
                    (int(left_inv), int(top_inv),
                     int(inverted_image.size[0] - right_inv), int(inverted_image.size[1] - bottom_inv))
    
    # Calculate border offsets and border boundaries for regular image.
    left_reg, top_reg, right_reg, bottom_reg = ImageOps._border((regular_image.width * border,
                                                                 regular_image.height * border))
    left_bound_reg, top_bound_reg, right_bound_reg, bot_bound_reg =\
                    (int(left_reg), int(top_reg),
                     int(regular_image.size[0] - right_reg), int(regular_image.size[1] - bottom_reg))
    
    # Get black border percentage for both images.
    black_border_percentage_inv = prep.get_black_border_percentage(inverted_image, border = border)
    black_border_percentage_reg = prep.get_black_border_percentage(regular_image, border = border)
    
    # Create fig, ax objects.
    fig, ax = plt.subplots(1, 2, figsize = (12, 12))
    
    # Plot images.
    plot_single_image(inverted_image, x_label = f'BB%: {black_border_percentage_inv:0.4f}',
                      title = 'Inverted', title_pad = 10, title_size = 24,
                      axis_pad = 10, axis_size = 16, axis = ax[0])
    plot_single_image(regular_image, x_label = f'BB%: {black_border_percentage_reg:0.4f}',
                      title = 'Regular', title_pad = 10, title_size = 24,
                      axis_pad = 10, axis_size = 16, axis = ax[0])
    
    # Determine border color - green means that version of the image is good, red means that version is bad.
    if black_border_percentage_inv < 0.4:
        color_inv = 'green'
        color_reg = 'red'
    else:
        color_inv = 'red'
        color_reg = 'green'
    
    # Create colored rectangles to represent image border boundaries.
    rect_inv = patches.Rectangle((left_bound_inv, bot_bound_inv),
                             right_bound_inv - left_bound_inv,
                             top_bound_inv - bot_bound_inv,
                             linewidth = 2,
                             edgecolor = color_inv,
                             facecolor = 'none')
    rect_reg = patches.Rectangle((left_bound_reg, bot_bound_reg),
                         right_bound_reg - left_bound_reg,
                         top_bound_reg - bot_bound_reg,
                         linewidth = 2,
                         edgecolor = color_reg,
                         facecolor = 'none')
    
    # Apply rectangles to axes.
    ax[0].add_patch(rect_inv)
    ax[1].add_patch(rect_reg)
    
    # Adjust subplot spacing.
    plt.subplots_adjust(wspace = 0, hspace = 0);
    
    return fig, ax

def image_generations_per_epoch(image_indices, epochs, model_names, base_path = '..', test_set = True,
                                image_dir = None, gen_dir = None, title_size = 32, label_size = 24,
                                width_multi = 4, height_multi = 3):
    '''
    Plots sketch input, generated versions of image at different epochs, and ground-truth image in a vertical grid for
    each image in image_indices using each of the generators in generator_names.
    Returns fig, ax objects with plotted images.
    
    image_indices: list of int or list of str, denoting either the indices to use when searching for images in the
                   test set, or the names of images to use when searching for images using image_dir and gen_dir.
    epochs: list of int, denoting which epochs to use for plotting generated images.
    model_names: list of str, denoting which models to use for selecting generated images.
    base_path: str, denoting the base path of repository, for use in extracting image locations.
    test_set: bool, whether the images are being drawn from the test set. If True, image_indices should be a list of int.
              If False, image_indices should be a list of str, and image_dir and gen_dir must not be None. Default True.
    image_dir: str or None, denoting the location of the sketch/image pair, only used if test_set is False. Default None.
    gen_dir: str or None, denoting the location of the generated images, only used if test_set is False. Default None.
    title_size: float, denoting the size of the title labels. Default 32.
    label_size: float, denoting the size of the axis labels. Default 24.
    width_multi: float, denoting the multiplier to apply to the figure size's width, used for controlling spacing
                 between images. Default 4.
    height_multi: float, denoting the multiplier to apply to the figure size's height, used for controlling spacing
                 between images. Default 3.
    '''
    
    # Draw fig, ax objects.
    rows = len(epochs) + 2
    columns = len(image_indices) * len(model_names)
    
    fig, ax = plt.subplots(rows, columns, figsize = (width_multi * rows, height_multi * columns))
    
    # Plot images.
    for i, index in enumerate(image_indices):
        # If pulling from a test set, use image indices as indices, not as image names.
        # Else, use the image directory given and use image indices as image names.
        if test_set:
            sketch, image = gen.split_pair(Image.open(f'{base_path}/data/test/Owl_Pair_{index:04d}.jpg'));
        else:
            sketch, image = gen.split_pair(Image.open(f'{image_dir}/{index}_pair.jpg'))
        
        # Convert from arrays to images.
        sketch = Image.fromarray(sketch);
        image = Image.fromarray(image);

        for j, name in enumerate(model_names):
            # Plot sketch and ground-truth image.
            plot_single_image(sketch, axis = ax[0][(i * len(model_names)) + j])
            plot_single_image(image, axis = ax[-1][(i * len(model_names)) + j])
#             ax[0][(i * 2) + j].imshow(sketch, cmap = 'gray', vmin = 0, vmax = 255,
#                                       extent = [-4, 4, -1, 1], aspect = 4);
#             ax[-1][(i * 2) + j].imshow(image, cmap = 'gray', vmin = 0, vmax = 255,
#                                        extent = [-4, 4, -1, 1], aspect = 4);
            # Set specific labeling at top-most images denoting which model used.
            ax[0][(i * len(model_names)) + j].set_title(f'{name}', pad = 10,
                                                        fontsize = title_size, rotation = 'horizontal');
        
            for k, epoch in enumerate(epochs):
                # If pulling from a test set, use image indices as indices, not as image names.
                # Else, use the generated directory given and use image indices as image names.
                if test_set:
                    generated = Image.open(f'{base_path}/data/generated/test/{name.lower()}/epoch_{epoch:03d}/'
                                           + f'Generated_Owl_Model_{name}_Epoch_{epoch:03d}_{index:04d}.jpg');
                else:
                    generated = Image.open(f'{gen_dir}/{index}_{name.lower()}_{epoch:03d}.jpg');
                
                # Plot generated image at given epoch.
                plot_single_image(generated, axis = ax[k + 1][(i * len(model_names)) + j])
#                 ax[k + 1][(i * 2) + j].imshow(generated, cmap = 'gray', vmin = 0, vmax = 255,
#                                               extent = [-4, 4, -1, 1], aspect = 4);
    
#     for j in range(len(image_indices) * len(generator_names)):
#         for i in range(len(epochs) + 2):
#             ax[i][j].get_xaxis().set_ticks([], []);
#             ax[i][j].get_yaxis().set_ticks([], []);
    
    

    # Set specific labeling at left-most images denoting sketch row and ground-truth row.
    ax[0][0].set_ylabel('Sketch', labelpad = 10, fontsize = label_size, rotation = 'vertical');
    ax[-1][0].set_ylabel('Ground-Truth', labelpad = 10, fontsize = label_size, rotation = 'vertical');
    
    # Set specific labeling at left-most images denoting epoch per row.
    for i, epoch in enumerate(epochs):
        ax[i + 1][0].set_ylabel(f'Epoch {epoch}', labelpad = 10, fontsize = label_size, rotation = 'vertical');
    
    # Adjust subplot spacing.
    plt.subplots_adjust(wspace = 0, hspace = 0);
    
    return fig, ax

def plot_gen_and_disc_losses(df, model_name):
    '''
    Plots the losses for a model. Returns a fig, ax object containing the plots of the losses functions.
    
    df: Pandas DataFrame, containing the columns "epoch", "gen_total_loss", and "disc_loss", used to plot the losses.
    model_name: str, denoting the model name. Used to format the titles of the graphs.
    '''
    
    # Draw fig, ax objects.
    fig, ax = plt.subplots(2, 1, figsize = (28, 18))
    
    # Draw loss graphs on the same graph.
    sns.lineplot(x = df.epoch, y = df.gen_total_loss, label = 'Generator Total Loss', ax = ax[0]);
    sns.lineplot(x = df.epoch, y = df.disc_loss, label = 'Discriminator Loss', ax = ax[1]);

    # Configure labels and titles.
    configure_axislabels_and_title('Epochs', 'Generator Total Loss',
                                   f'{model_name} Generator Loss Across Epochs', ax = ax[0]);
    configure_axislabels_and_title('Epochs', 'Discriminator Total Loss',
                                   f'{model_name} Discriminator Loss Across Epochs', ax = ax[1]);
    
    # Add a red horizontal line denoting the ideal discriminator loss at ln(2) - 50/50 uncertainty.
    ax[1].hlines(np.log(2), 0, 200, colors = 'red', label = 'Ideal Discriminator Loss')
    
    # Configure ticklabels and legend.
    for axis in ax:
        configure_ticklabels_and_params(ax = axis);
        configure_legend(ax = axis, fancybox = True, frameon = True, fontsize = 16);
        
    # Set tight layout.
    plt.tight_layout(pad = 3.0)

    return fig, ax

def plot_fid_scores(merged_fid_df):
    '''
    Plots the FID scores for both models. Returns fig, ax objects containing plotted scores.
    
    merged_fid_df: Pandas DataFrame, containing columns "epoch", and any number of columns of the form
                   "fid_\w+", used for plotting the FID scores across models.
    '''
    
    # Create fig, ax objects.
    fig, ax = plt.subplots(figsize = (16, 8))
    
    # Select non-epoch columns.
    non_epoch_col = [x for x in merged_fid_df.columns if 'epoch' != x.lower()]
    
    # For each non-epoch column, draw the lineplot of the FID scores across epochs.
    for col in non_epoch_col:
        # Extract the title of the model using regex.
        label = re.search('fid_(\w+)_?', col)[1].title()
        sns.lineplot(x = merged_fid_df.epoch, y = merged_fid_df[col], label = label, ax = ax);

    # Configure axis labels, title, ticklabels, and legend.
    configure_axislabels_and_title('Epochs', 'FID Score',
                                             'Frechet Inception Distance (FID) Across Epochs', ax = ax);
    configure_ticklabels_and_params(ax = ax);
    configure_legend(ax = ax, fancybox = True, frameon = True, fontsize = 16);
    
    return fig, ax