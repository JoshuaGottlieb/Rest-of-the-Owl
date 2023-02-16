# Rest of the Owl

**Author**: Joshua Gottlieb

## Usage Information and Known Bugs

Non-Colab notebooks are visible in notebooks directory of this repository. All data, model weights, and colab notebooks can be found at this [Google Drive](https://drive.google.com/drive/folders/1G_lOUjNFyL0Vx2cLZQDXYWASLBHe4jL8?usp=share_link) for the time being. The current draft of the presentation can be found [here](https://docs.google.com/presentation/d/1bw1ivyi-PK-g4p3Tn5eHXXpfb9wErUO58FC-7fqH3A4/edit#slide=id.g207a982db38_0_58).

Things to do:
<ul>
  <li>Update this README.</li>
  <li>Curate the Google Drive to contain only essential information and move data out of personal Google Drive.</li>
</ul>

If any of the images in this README fail to load or are not legible due to Github dark mode, all of the visuals contained exist in the [visualizations](./visualizations) directory of this repository or in the [presentation](./presentation/Rest-of-the-Owl-Presentation.pdf).

## Overview and Research

| ![](./visualizations/misc/draw_the_rest_of_the_owl.jpg) |
| :---: |
| [Know Your Meme - Rest of the Owl](https://knowyourmeme.com/memes/how-to-draw-an-owl) |

This seemingly comical internet meme inspired an important research question for me. Is it actually possible to take a low-quality sketch of an image and produce a high-quality sketch using neural networks? Low-detail sketches contain very little information, so this seemed like it would be a difficult project, and while my intuition was not wrong about the difficulty of such a task, it turns out that sketch-to-image synthesis is a current domain of research, with papers published on the topic as recently as 2021.


In researching this topic, I stumbled upon the paper [Auto-painter: Cartoon Image Generation from Sketch by Using Conditional Generative Adversarial Networks](https://arxiv.org/pdf/1705.01908.pdf), which was subsequently based upon the "pix2pix" model published in [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1705.01908.pdf). These two papers detailed neural networks which were capable of image-to-image synthesis, which is exactly what I wanted to achieve with my sketch-to-sketch idea. Conveniently, both of these models share similar architectures, and so I adapted my model architecture and loss functions from these two papers and named my models "autopainter" and "pix2pix," respectively. Unfortunately, the [original autopainter model](https://github.com/irfanICMLL/Auto_painter) was written in Tensorflow v1, while the [original pix2pix model](https://github.com/phillipi/pix2pix) was written in PyTorch. Fortunately, there exists a [Tensorflow v2.x tutorial for pix2pix](https://www.tensorflow.org/tutorials/generative/pix2pix), which I was able to use to help translate the original models into a Tensorflow v2.x architecture.

The purpose of this project is to create and train a conditional generative adversarial network (cGAN) which can take in low-detail "sketches" and produce high-detail sketches.
  
## Data Collection

For data, I used [requests](https://requests.readthedocs.io/en/latest/) and [BeautifulSoup](https://beautiful-soup-4.readthedocs.io/en/latest/) to scrape [Adobe Stock](https://stock.adobe.com/), [VectorStock](https://www.vectorstock.com/), and [Fine Art America](https://fineartamerica.com/). I scraped around ~30,000 images; ~20,000 from Adobe Stock, ~4,000 from VectorStock, and ~6,000 from Fine Art America. In order to choose my images, I manually inspected ~17,000 images, ending with 2,880 images.

My criteria for selecting images was relatively simple. Images needed to have:
<ul>
  <li>Medium-High levels of detail in order to generate lower levels of sketches.</li>
  <li>Only a single owl present in the photo, with no other animals present.</li>
  <li>Realistic drawings/photos, and not look like cartoon images or metallic creations.</li>
  <li>No text visible in the image.</li>
  <li>Minimal background objects (the main ones which I allowed were braches/trees and moons).</li>
  <li>No glasses or hats.</li>
</ul>

For examples of images that failed to meet these criteria, see the [dropped subfolder](./visualizations/dropped) in this repository, or check out slide/page 4 of the [presentation](./presentation/Rest-of-the-Owl-Presentation.pdf).

## Data Cleaning and Sketch Generation

In order to train my model, I needed sketch/image pairs, but my scraping only provided me with images. Thus, I created my own sketches using a the eXtended Difference of Gaussians (XDoG) technique outlined in [XDoG: An eXtended difference-of-Gaussians compendium
including advanced image stylization](https://users.cs.northwestern.edu/~sco590/winnemoeller-cag2012.pdf). Specifically, I used Python implementation of the baseline XDoG model using a continuous ramp, [available here](https://github.com/heitorrapela/xdog). The DoG process works by applying two separate Gaussian Blurs to an image, one very weak, one very strong, and calculating a weighted difference between the two blurred images. The XDoG process furthers this by applying thresholding to each pixel and ramping each pixel from black to white to create a simple yet effect edge detection algorithm. Illustrated below is the process applied to an image.

| Ground-Truth | Weak Blur | - γ * Strong Blur |
| :--: | :--: | :--: |
| ![](./visualizations/DoG/ground_truth_owl.png) | ![](./visualizations/DoG/weak_gauss_blur.png) | ![](./visualizations/DoG/strong_gauss_blur.png) |

| Apply Differencing |
| :--: |
| ![](./visualizations/DoG/gauss_mixtures.png) |
| Apply Thresholding |
| ![](./visualizations/DoG/sketch_thresholds_01.png) |

The threshold statistic underneath each image indicates the percentage of the image that is covered in "black" pixels (value <= 40). This is a self-made measurement for the amount of detail each sketch has. Observe how different starting images require different gamma values to obtain the same fill-threshold.

|![](./visualizations/DoG/sketch_thresholds_02.png) |
| :--: |

In an attempt to normalize the detail levels of each of my sketches, I decided to create sketches based upon fill thresholds rather than by applying a single gamma value across my dataset. After inspecting ~50 images at different thresholds, I settled on a threshold 0.03 as having the best balance of low-detail while not losing general shapes and features that should be captured.

Some images consisted mainly of black images with white highlights. In an attempt to normalize my images as being black drawings on white backgrounds, I decided to invert some of my images. The decision on whether or not to invert an image was done by examining how much of the border was "black" (again, value <= 40). The theory is that images which are on black backgrounds will exhibit a high black-border percentage, while images which are on white backgrounds will exhibit a low black-border percentage.

| Image which should be inverted |
| :--: |
| ![](./visualizations/borders/pair_02.png) |
| Image which should not be inverted |
| ![](./visualizations/borders/pair_01.png) |

The area of the image that I used for the border was the outer 40% of the image. My threshold for determining if an image should be inverted is a 40% black-border percentage, which is a number I settled on through manual inspection of ~50 images. If a regular image has >40% black-border percentage, the inverted should be used, and vice-versa, as the inverted version of a regular image will have a high black-border percentage.

In order to remove duplicates, I used the [difPy package](https://github.com/elisemercury/Duplicate-Image-Finder/wiki/difPy-Usage-Documentation) to select the highest resolution versions of each image. Finally, after choosing which version of each image to use and generating sketches, the images and sketches were resized to 256x256 with zero-padding to preserve aspect ratio. The sketches and images were then concatenated together horizontally, as specified in the [Tensorflow v2.x tutorial for pix2pix](https://www.tensorflow.org/tutorials/generative/pix2pix), for use in preprocessing for modeling.

## Model Architecture

| Generator Training Loop | Discriminator Training Loop |
| :--: | :--: |
| ![](./visualizations/model_architecture/model_visualizations/generator_loop.png) | ![](./visualizations/model_architecture/model_visualizations/discriminator_loop.png) |
| [Generator Loop - Pix2Pix Tutorial](https://www.tensorflow.org/tutorials/generative/pix2pix#define_the_generator_loss) | [Discriminator Loop - Pix2Pix Tutorial](https://www.tensorflow.org/tutorials/generative/pix2pix#define_the_discriminator_loss) |

Both of my models follow a modified cGAN architecture, where the generator and discriminator take turns training each other. The generator creates a fake image from an input sketch. The discriminator takes in the sketch/ground-truth pair and makes a prediction on whether such a pair is probable, and then separately takes in the sketch/generated-image pair and makes a prediction on whether the pair is probable. The discriminator's job is to classify sketch/ground-truth pairs as probable and to classify sketch/generated-image pairs as improbable. The generator's job is to create generated-images that can fool the discriminator. Losses and gradients are then calculated, and weights are updated before training again.

While the pix2pix and autopainter models referenced in their respective papers differ slightly in architecture, my versions utilize the same architecture, differing only in their loss functions.

| pix2pix Losses |
| :--: |
| ![](./visualizations/model_architecture/losses/cgan_loss.png) |
| ![](./visualizations/model_architecture/losses/l1_loss.png) |
| ![](./visualizations/model_architecture/losses/pix2pix_loss.png) |

The pix2pix losses are relatively simple. `G` stands for the generator, `D` for the discriminator, `x` for the sketch, `y` for the ground-truth image, `z` for the noise, and `G(x,z)` for the generated image. The cGAN loss is defined as the log loss of the discriminator's ability to correctly identify sketch/ground-truth pairs and sketch/generated-image pairs. The discriminator seeks to minimize this loss, while the generator seeks to maximize this loss. In addition, an L1 pixel-level loss is calculated for the generator to ensure that the generated image does not vary drastically from the real image. The weights for each loss are the same as those used in the pix2pix Tensorflow tutorial.

| autopainter Losses |
| :--: |
| ![](./visualizations/model_architecture/losses/feature_loss.png) |
| ![](./visualizations/model_architecture/losses/tv_loss.png) |
| ![](./visualizations/model_architecture/losses/autopainter_loss.png) |

The autopainter losses build upon the pix2pix losses, with two additional loss functions defined as part of the generator loss. The feature loss uses the L2 distance between the feature maps for the real image and the feature maps for the generated image at a specific layer of a convolutional net in order to maintain overall feature and shape consistency in generated images. Following the autopainter paper, I used a VGG16 net and used the `Conv3_3` layer for the feature maps. The total variation loss consists of the root of the sum of squared errors between the generated image and shifted versions of itself. This ensures that the total variation of the generated images is dampened, preventing sharp changes in color (in the case of grayscale images, brightness) in the generated image and produces a smoothing effect on generated images. Both of these losses are added to the pix2pix losses to create the autopainer loss, with the weights for each loss identical to those used in the autopainter models available on GitHub.

## Results


## Conclusions


## Next Steps


## For More Information

Please look at my full analysis in [Jupyter Notebooks](./notebooks) and code in my [Python Modules](./notebooks/src), or my [presentation](./presentation/Rest-of-the-Owl-Presentation.pdf).

For any additional questions, please contact: **Joshua Gottlieb (joshuadavidgottlieb@gmail.com)**

## Repository Structure
