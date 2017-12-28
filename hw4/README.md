
### Package : 
`Pytorch` &nbsp; `Keras` &nbsp;` Numpy`  &nbsp;` pandas` &nbsp;

### Note :
Conditional GAN for text2image generation

In my .py script, I used the following script to assign the task running on GPU 0.<br>

```
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
```

##Conditional DCGAN

test text:
```
1,blue hair blue eyes
2,blue hair green eyes
3,blue hair red eyes
4,green hair blue eyes
5,red hair
6,red hair green eyes
7,black eyes
8,aqua eyes purple hair
9,white hair pink eyes
10,brown eyes aqua hair
```
![Alt Text](https://github.com/thtang/ADLxMLDS2017/blob/master/hw4/anime_cDCGAN_generation_animation_.gif)
