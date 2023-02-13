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
    
    x_label: String to use for labeling the x-axis.
    y_label: String to use for labeling the y-axis.
    title: String to use for the title of the ax object.
    ax: The matplotlib axis object to modify.
    axis_size: Float representing the font size of the axis labels. Default 24.0.
    title_size: Float representing def image_generations_per_epoch(image_indices, epochs, generator_names, base_path = '..', test_set = True,
                                image_dir = None, gen_dir = None, title_size = 32, label_size = 24,
                                width_multi = 4, height_multi = 3):
    fig, ax = plt.subplots(len(epochs) + 2, len(image_indices) * len(generator_names),
                           figsize = (width_multi * (len(epochs) + 2),
                                      height_multi * len(image_indices) * len(generator_names)))
    
    for i, index in enumerate(image_indices):
        if test_set:
            sketch, image = split_pair(Image.open(f'{base_path}/data/test/Owl_Pair_{index:04d}.jpg'));
        else:
            sketch, image = split_pair(Image.open(f'{image_dir}/{index}_pair.jpg'))
        sketch = Image.fromarray(sketch);
        image = Image.fromarray(image);

        for j, name in enumerate(generator_names):
            ax[0][(i * 2) + j].imshow(sketch, cmap = 'gray', vmin = 0, vmax = 255,
                                      extent = [-4, 4, -1, 1], aspect = 4);
            ax[-1][(i * 2) + j].imshow(image, cmap = 'gray', vmin = 0, vmax = 255,
                                       extent = [-4, 4, -1, 1], aspect = 4);
            ax[0][(i * 2) + j].set_title(f'{name}', pad = 10, fontsize = title_size, rotation = 'horizontal');
            ax[0][0].set_ylabel('Sketch', labelpad = 10, fontsize = label_size, rotation = 'vertical');
            ax[-1][0].set_ylabel('Ground-Truth', labelpad = 10, fontsize = label_size, rotation = 'vertical');
            
            for k, epoch in enumerate(epochs):
                if test_set:
                    generated = Image.open(f'{base_path}/data/generated/test/{name.lower()}/epoch_{epoch:03d}/'
                                           + f'Generated_Owl_Model_{name}_Epoch_{epoch:03d}_{index:04d}.jpg');
                else:
                    generated = Image.open(f'{gen_dir}/{index}_{name.lower()}_{epoch:03d}.jpg');
                ax[k + 1][(i * 2) + j].imshow(generated, cmap = 'gray', vmin = 0, vmax = 255,
                                              extent = [-4, 4, -1, 1], aspect = 4);
    
    for j in range(len(image_indices) * len(generator_names)):
        for i in range(len(epochs) + 2):
            ax[i][j].get_xaxis().set_ticks([], []);
            ax[i][j].get_yaxis().set_ticks([], []);
    
    for i, epoch in enumerate(epochs):
        ax[i + 1][0].set_ylabel(f'Epoch {epoch}', labelpad = 10, fontsize = label_size, rotation = 'vertical');
    
#     plt.subplots(layout = 'constrained')
    plt.subplots_adjust(wspace = 0, hspace = 0);
    
    return fig, axthe font size of the title. Default 32.0.
    axis_pad: Float representing the padding between graph and axis labels. Default 10.0.
    title_pad: Float representing the padding between graph and title. Default 10.0.
    font_name: String representing the name of the font to use for all labels. Default Arial.
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
    label_size: Float, the font size of the major tick labels. Default 16.
    length: Float, the length of the major tick marks. Default 8.
    width: Float, the width of the major tick marks. Default 1.
    font_name: String, the font to use for tick labels. Default Arial.
    x_label_size: Float, the font size of the major tick labels on the x-axis. By default inherits from label_size.
    x_length: Float, the length of the major tick marks on the x-axis. By default inherits from length.
    x_width: Float, the width of the major tick marks on the x-axis. By default inherits from width.
    y_label_size: Float, the font size of the major tick labels on the y-axis. By default inherits from label_size.
    y_length: Float, the length of the major tick marks on the y-axis. By default inherits from length.
    y_width: Float, the width of the major tick marks on the y-axis. By default inherits from width.
    format_xticks: Bool, indicates whether to format numerical x-tick labels. Default False.
    x_ticks_rounding: Integer, the number by which to divide the x-tick labels (e.g. to round to 10s, set to 10). Default 1.
    x_ticks_rotation: Integer, the number in degrees by which to rotate the x-tick labels counter-clockwise. Default 0.
    format_yticks: Bool, indicates whether to format numerical y-tick labels. Default False.
    y_ticks_rounding: Integer, the number by which to divide the y-tick labels (e.g. to round to 10s, set to 10). Default 1.
    y_ticks_rotation: Integer, the number in degrees by which to rotate the y-tick labels counter-clockwise. Default 0.
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
    labels: List of strings, to manually specify the labels to use for each artist. Default None uses automatic labeling.
    loc: Integer, string, or pair of floats, specifies the location of the legend according to matplotlib documentation.
         Default 0, equivalent to 'best'.
    bbox_to_anchor: 2-tuple or 4-tuple of floats, specifies the location of the bounding box of the legend. Default None.
    labelcolor: String or list of strings, specifies the color(s) of the labels. Default 'black'.
    fontsize: Integer, specifies the fontsize of the labels. Default 12.
    frameon: Bool, denotes whether the legend should be drawn on a patch (frame). Default False.
    facecolor: 'inherit' or color, denotes the background color of the legend. If 'inherit', match the color of the axis.
               Default 'inherit'.
    fancybox: Bool, denotes whether the legend's background should have rounded edges. Default False.
    framealpha: Float, denotes the alpha transparency of the legend's background. Default 0.8.
    edgecolor: 'inherit' or color, denotes the legend's background edge color. If 'inherit', match the color of the axis.
               Default 'inherit'.
    borderpad: Float, denotes the fractional whitespace inside the legend border, in font-size units. Default 0.4.
    labelspacing: Float, denotes the vertical space between the legend entries, in font-size units. Default 0.5.
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

def get_example_sketches_at_thresholds(image_file, fill_space_thresholds):
    image = Image.open(image_file)
    
    sketches = []
    
    for threshold in fill_space_thresholds:
        sketches.append(prep.find_sketch_threshold(image, fill_space_threshold = threshold))
        
    return sketches

def display_sketch_thresholds(image_file, fill_space_thresholds, sketches):
    fig, ax = plt.subplots(1, 5, figsize = (18, 10));

    for i in range(4):
        ax[i].imshow(sketches[i][0], cmap = 'gray');
        ax[i].set_xlabel(f'Threshold {fill_space_thresholds[i]}\nGamma: {sketches[i][1]}',
                         labelpad = 10, fontsize = 16);
        ax[i].get_xaxis().set_ticks([]);
        ax[i].get_yaxis().set_ticks([]);

    ax[4].imshow(Image.open(image_file), cmap = 'gray');
    ax[4].set_xlabel('Ground-Truth', labelpad = 10, fontsize = 16);
    ax[4].get_xaxis().set_ticks([]);
    ax[4].get_yaxis().set_ticks([]);
    plt.subplots_adjust(wspace = 0, hspace = 0);
    
    return fig, ax

def display_dog_mixtures(gauss_mixtures, gammas):
    fig, ax = plt.subplots(1, 4, figsize = (14, 10));

    for i in range(4):
        ax[i].imshow(np.uint8(gauss_mixtures[i]), cmap = 'gray', vmin = 0, vmax = 255);
        ax[i].set_xlabel(f'Gamma: {gammas[i]}', labelpad = 10, fontsize = 16);
        ax[i].get_xaxis().set_ticks([]);
        ax[i].get_yaxis().set_ticks([]);
    plt.subplots_adjust(wspace = 0, hspace = 0);
    
    return fig, ax

def plot_single_image(image, x_label, labelpad = 10, fontsize = 16, figsize = (4, 4)):
    fig, ax = plt.subplots(figsize = figsize)
    
    ax.imshow(image, cmap = 'gray', vmin = 0, vmax = 255);
    configure_axislabels_and_title(x_label, None, None, ax = ax,
                                   axis_size = fontsize, axis_pad = labelpad)
    
    ax.get_xaxis().set_ticks([]);
    ax.get_yaxis().set_ticks([]);
    
    return fig, ax

def create_images_with_border(inverted_image, regular_image, border):
    left_inv, top_inv, right_inv, bottom_inv = ImageOps._border((inverted_image.width * border,
                                                                 inverted_image.height * border))
    
    left_bound_inv, top_bound_inv, right_bound_inv, bot_bound_inv =\
                    (int(left_inv), int(top_inv),
                     int(inverted_image.size[0] - right_inv), int(inverted_image.size[1] - bottom_inv))
    
    left_reg, top_reg, right_reg, bottom_reg = ImageOps._border((regular_image.width * border,
                                                                 regular_image.height * border))
    
    left_bound_reg, top_bound_reg, right_bound_reg, bot_bound_reg =\
                    (int(left_reg), int(top_reg),
                     int(regular_image.size[0] - right_reg), int(regular_image.size[1] - bottom_reg))
    
    black_border_percentage_inv = prep.get_black_border_percentage(inverted_image, border = border)
    black_border_percentage_reg = prep.get_black_border_percentage(regular_image, border = border)
    
    fig, ax = plt.subplots(1, 2, figsize = (12, 12))
    
    ax[0].imshow(inverted_image, cmap = 'gray', vmin = 0, vmax = 255)
    ax[0].set_xlabel(f'BB%: {black_border_percentage_inv:0.4f}', labelpad = 10, fontsize = 16);
    ax[0].set_title('Inverted', pad = 10, fontsize = 24)
    ax[1].imshow(regular_image, cmap = 'gray', vmin = 0, vmax = 255)
    ax[1].set_title('Regular', pad = 10, fontsize = 24)
    ax[1].set_xlabel(f'BB%: {black_border_percentage_reg:0.4f}', labelpad = 10, fontsize = 16);
    
    for axis in ax:
        axis.get_xaxis().set_ticks([], []);
        axis.get_yaxis().set_ticks([], []);
    
    if black_border_percentage_inv < 0.4:
        color_inv = 'green'
        color_reg = 'red'
    else:
        color_inv = 'red'
        color_reg = 'green'
    
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
    
    ax[0].add_patch(rect_inv)
    ax[1].add_patch(rect_reg)
    
    plt.subplots_adjust(wspace = 0, hspace = 0);
    
    return fig, ax

def image_generations_per_epoch(image_indices, epochs, generator_names, base_path = '..', test_set = True,
                                image_dir = None, gen_dir = None, title_size = 32, label_size = 24,
                                width_multi = 4, height_multi = 3):
    
    fig, ax = plt.subplots(len(epochs) + 2, len(image_indices) * len(generator_names),
                           figsize = (width_multi * (len(epochs) + 2),
                                      height_multi * len(image_indices) * len(generator_names)))
    
    for i, index in enumerate(image_indices):
        if test_set:
            sketch, image = gen.split_pair(Image.open(f'{base_path}/data/test/Owl_Pair_{index:04d}.jpg'));
        else:
            sketch, image = gen.split_pair(Image.open(f'{image_dir}/{index}_pair.jpg'))
        sketch = Image.fromarray(sketch);
        image = Image.fromarray(image);

        for j, name in enumerate(generator_names):
            ax[0][(i * 2) + j].imshow(sketch, cmap = 'gray', vmin = 0, vmax = 255,
                                      extent = [-4, 4, -1, 1], aspect = 4);
            ax[-1][(i * 2) + j].imshow(image, cmap = 'gray', vmin = 0, vmax = 255,
                                       extent = [-4, 4, -1, 1], aspect = 4);
            ax[0][(i * 2) + j].set_title(f'{name}', pad = 10, fontsize = title_size, rotation = 'horizontal');
            ax[0][0].set_ylabel('Sketch', labelpad = 10, fontsize = label_size, rotation = 'vertical');
            ax[-1][0].set_ylabel('Ground-Truth', labelpad = 10, fontsize = label_size, rotation = 'vertical');
            
            for k, epoch in enumerate(epochs):
                if test_set:
                    generated = Image.open(f'{base_path}/data/generated/test/{name.lower()}/epoch_{epoch:03d}/'
                                           + f'Generated_Owl_Model_{name}_Epoch_{epoch:03d}_{index:04d}.jpg');
                else:
                    generated = Image.open(f'{gen_dir}/{index}_{name.lower()}_{epoch:03d}.jpg');
                ax[k + 1][(i * 2) + j].imshow(generated, cmap = 'gray', vmin = 0, vmax = 255,
                                              extent = [-4, 4, -1, 1], aspect = 4);
    
    for j in range(len(image_indices) * len(generator_names)):
        for i in range(len(epochs) + 2):
            ax[i][j].get_xaxis().set_ticks([], []);
            ax[i][j].get_yaxis().set_ticks([], []);
    
    for i, epoch in enumerate(epochs):
        ax[i + 1][0].set_ylabel(f'Epoch {epoch}', labelpad = 10, fontsize = label_size, rotation = 'vertical');
    
    plt.subplots_adjust(wspace = 0, hspace = 0);
    
    return fig, ax