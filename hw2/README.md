# Video Captioning
### Package : 
`Tensorflow` &nbsp; ` Numpy`  &nbsp;` pandas` &nbsp;

### Note :
In my .py script, I used the following script to assign the task running on GPU 0.<br>

```
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
```

Usage Example for special mission:<br>
```
bash hw2_special.sh MLDS_hw2_data/ special_output.txt
```

for complete submission:<br>
```
bash hw2_seq2seq.sh MLDS_hw2_data/ testset_output.txt peerreview_output.txt
```
### Model description
![Alt Text](https://github.com/thtang/ADLxMLDS2017/blob/master/hw2/S2VTarchitecture.png)
* reference:https://arxiv.org/abs/1505.00487
### Result:
Input a video:<br>

![Alt Text](https://github.com/thtang/ADLxMLDS2017/blob/master/hw2/video_2.gif)

Output text:
```
a group of men are dancing. 
```
