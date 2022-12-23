# Instructions

There are two python scripts in this resources.zip package.

## seam_carving.py:

This is the main python script to generate all replicated images. It can run directly in the CS6475 environment. No other packages are needed. Just make sure all the original images(before retargeting) from the authors are under the 'images/' folder in the run directory.

It takes about 20 minutes on my laptop to complete.

My laptop hardware configuration overview:

|Model Name				        |MacBook Pro           |
|-------------------------------|----------------------|
|Model Identifier				|MacBookPro15,1        |
|Processor Name				    |8-Core Intel Core i9  |
|Processor Speed				|2.3 GHz               |
|Number of Processors			|1                     |
|Total Number of Cores			|8                     |
|L2 Cache (per Core)			|256 KB                |
|L3 Cache				        |16 MB                 |
|Hyper-Threading Technology		|Enabled               |
|Memory				            |16 GB                 |

## MT_compare.py:

This the script to generate the MSE and SSIM report of each replicated image. It requires an extra package scikit-image to be installed in the CS6475 environment. To install it, type the following command in the CS6475 environment prompt:

conda install scikit-image

Then,

1. put the file names of all original images from the author(after retargeting) in the 'author_result' list in the script
2. put the file names of my replicated images in the 'my_result' list in the script
3. Assign correct directories to the 'my_path' and 'author_path' variable.

Save it and run the script. It will output the following table:
|Replica                      |        MSE   |   SSIM|
|-----------------------------|--------------|-------|
|fig5.png                     |     631.72   |   0.74|
|fig8d_07.png                 |      85.80   |   0.94|
|fig8f_07.png                 |     119.88   |   0.93|
|fig8Comp_backward_08.png     |     181.05   |   0.83|
|fig8Comp_forward_08.png      |     865.09   |   0.68|
|fig9Comp_backward_08.png     |      67.86   |   0.92|
|fig9Comp_forward_08.png      |      98.46   |   0.88|

## Demo and explanation

The link to my project video:
<https://drive.google.com/file/d/1tGe4Ixg8UZb4ayzXfiWFfKYAVZTe2gEu/view?usp=sharing>
