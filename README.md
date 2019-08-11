# Neural Style Transfer #
Neural style transfer is an algorithm that takes two images and generates an image with the content 
of one image with the style of the other. The techniques can be found in [Leon A. Gatys, Alexander S. Ecker, and 
Matthias Bethge 's "A Neural Algorithm of Artistic Style"](https://arxiv.org/pdf/1508.06576.pdf).  
My implementation uses intermediate layers of [Keras' pre-trained VGG-19 model](https://keras.io/applications/#vgg19).
While the content cost is computed using the activation of layer 3, the weighted style cost is computed using the 
activations of layers 2, 3, 4 with weights (0.3, 0.4, 0.3). Optimization is done using scipy's L-BFGS implementation.   
The original and generated images are shown below.  
Content image:  
<img src="nature.jpg" width="400" height="300">  
Style Image:  
<img src="the_scream.jpg" width="300" height="400">  
Generated Images:  
<img src="nst_results/iteration_10.png" width="33%">
<img src="nst_results/iteration_20.png" width="33%">
<img src="nst_results/iteration_50.png" width="33%">