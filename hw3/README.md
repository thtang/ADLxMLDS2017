# Game Playing
### Techniques:
`Deep reinforce learning` &nbsp; `DQN` &nbsp;`Polocy network`  &nbsp;`DDQN` &nbsp;
### Package : 
`Tensorflow` &nbsp; `Keras` &nbsp;` Numpy`  &nbsp;` pandas` &nbsp;

### Environment 
| Breakout  | Pong  |  
|---|---|
|  <img src="https://github.com/thtang/ADLxMLDS2017/blob/master/hw3/picture/break_out.gif" width=280> |<img src="https://github.com/thtang/ADLxMLDS2017/blob/master/hw3/picture/pong2.gif" width=280>   |

### Experiment results:
Please refer to [the report](https://github.com/thtang/ADLxMLDS2017/blob/master/hw3/report.pdf).
### Note :
In my .py script, I used the following script to assign the task running on GPU 0.<br>

```
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
```
