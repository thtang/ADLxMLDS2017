
### Package : 
`Pytorch` &nbsp; `Keras` &nbsp;` Numpy`  &nbsp;` pandas` &nbsp;

### Note :
Conditional GAN for text2image generation

In my .py script, I used the following script to assign the task running on GPU 0.<br>

```
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
```
![Alt Text](https://github.com/thtang/ADLxMLDS2017/blob/master/hw4/anime_cDCGAN_generation_animation_.gif)