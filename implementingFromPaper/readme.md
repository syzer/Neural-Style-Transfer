# about paper

0. with neural net you can separate content and style
1. uses texture processing techniques, to grab superficial information about colors
2. encodes spacial information to gram matrix
3. down sampling is done via max-pooling
4. style is encoded as multiple layers of conv net
5. other techniques need to be used to separate content and style **completly**

6. neural system that uses biological vision learns to represent image: like a sum of style + content
7. 16 convolutional and 5 pooling layers of the 19 layer VGG- Network.
8. syntesis uses average pooling(smooth) n place of `max-pooling`

9. calculate loss function  as sare distance between genereted image and gram matrix
10. calculate gradient of loss between layers
11. start from white noice
12. Loss is sum of alpha * Loss of content and beta Loss of style 


also there are known implementations for torch , cafee and tf


# it follows pytorch state transfer tutorial

