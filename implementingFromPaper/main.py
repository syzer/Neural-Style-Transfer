import os
import torch.utils.data
import torchvision.datasets as datasets

from neural_net import *
from gram import *
from load_file import *

# CUDA Configurations (Macs typically have to run on CPU)
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

_dirname = os.path.abspath(os.path.dirname(__file__))
input_dir = os.path.join(_dirname, "../images/inputs/style/")
output_dir = os.path.join(_dirname, "../images/output/")

# Content and style
style = image_loader(input_dir + "starry_night.jpg").type(dtype)
content = image_loader(input_dir + "patterned_leaves.jpg").type(dtype)

# input
pastiche = image_loader(output_dir + "blue_moon_lake_1-0_2.jpg").type(dtype)
pastiche.data = torch.randn(pastiche.data.size()).type(dtype)

num_epochs = 1


def main(pastiche):
    pass
    style_cnn = StyleCNN(style, content, pastiche)

    for i in range(num_epochs):
        pastiche = style_cnn.train()

        if i % 10 == 0:
            print("Iteration: %d" % (i))

            path = "outputs/%d.png" % (i)
            pastiche.data.clamp_(0, 1)
            save_image(pastiche, path)



if __name__ == "__main__":
    main(pastiche)

