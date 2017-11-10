
### Package : 
`Tensorflow` &nbsp; ` Numpy`  &nbsp;` pandas` &nbsp;

### Note :
In my .py script, I used the following script to assign the task running on GPU 0.<br>

```
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
```

Usage Example:<br>
```
bash hw2_special.sh MLDS_hw2_data/ special_output.txt
```