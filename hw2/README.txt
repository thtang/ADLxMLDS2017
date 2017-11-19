### Package : 
Tensorflow  Numpy  Pandas and other python standard liberary

### Note :
In my .py script, I used the following script to assign the task running on GPU 0.
```
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
```

Usage Example for special mission:
```
bash hw2_special.sh MLDS_hw2_data/ special_output.txt
```

for complete submission:
```
bash hw2_seq2seq.sh MLDS_hw2_data/ testset_output.txt peerreview_output.txt
```