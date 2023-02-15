# Rest of the Owl

**Author**: Joshua Gottlieb

## Usage Information and Known Bugs

Non-Colab notebooks are visible in notebooks directory of this repository. All data, model weights, and colab notebooks can be found at this [Google Drive](https://drive.google.com/drive/folders/1G_lOUjNFyL0Vx2cLZQDXYWASLBHe4jL8?usp=share_link) for the time being. The current draft of the presentation can be found [here](https://docs.google.com/presentation/d/1bw1ivyi-PK-g4p3Tn5eHXXpfb9wErUO58FC-7fqH3A4/edit#slide=id.g207a982db38_0_58).

Things to do:
<ul>
  <li>Add documentation to all functions and test them in all notebooks to ensure they still work.</li>
  <li>Create PDF of presentation and upload to Github.</li>
  <li>Move Colab notebooks to Github.</li>
  <li>Update this README.</li>
  <li>Curate the Google Drive to contain only essential information and move data out of personal Google Drive.</li>
</ul>


## Overview and Research

| <img src="https://github.com/JoshuaGottlieb/Rest-of-the-Owl/blob/main/visualizations/misc/draw_the_rest_of_the_owl.jpg"></img> |
| :---: |
| <a href="https://knowyourmeme.com/memes/how-to-draw-an-owl">Know Your Meme - Rest of the Owl</a> |

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




## Results


## Conclusions


## Next Steps


## For More Information

Please look at my full analysis in [Jupyter Notebooks](./notebooks) and code in my [Python Modules](./notebooks/src), or my [presentation](./presentation/Rest-of-the-Owl-Presentation.pdf).

For any additional questions, please contact: **Joshua Gottlieb (joshuadavidgottlieb@gmail.com)**

## Repository Structure
