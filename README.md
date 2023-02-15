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



In researching this topic, I stumbled upon the paper [Auto-painter: Cartoon Image Generation from Sketch by Using Conditional Generative Adversarial Networks](https://arxiv.org/pdf/1705.01908.pdf), which was subsequently based upon the "pix2pix" model published in [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1705.01908.pdf). These two papers detailed neural networks which were capable of image-to-image synthesis, which is exactly what I wanted to achieve with my sketch-to-sketch idea. Conveniently, both of these models share similar architectures, and so I adapted my model architecture and loss functions from these two papers and named my models "autopainter" and "pix2pix," respectively. Unfortunately, the [original autopainter model](https://github.com/irfanICMLL/Auto_painter) was written in Tensorflow v1, while the [original pix2pix model](https://github.com/phillipi/pix2pix) was written in PyTorch. Fortunately, there exists a Tensorflow v2.x tutorial for pix2pix, which I was able to use to help translate the original models into a Tensorflow 2 architecture.




  
## Data Collection

## Data Cleaning




## Results


## Conclusions


## Next Steps


## For More Information

Please look at my full analysis in [Jupyter Notebooks](./notebooks) and code in my [Python Modules](./notebooks/src), or my [presentation](./presentation/Rest-of-the-Owl-Presentation.pdf).

For any additional questions, please contact: **Joshua Gottlieb (joshuadavidgottlieb@gmail.com)**

## Repository Structure
