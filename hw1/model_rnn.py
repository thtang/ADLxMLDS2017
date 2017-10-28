## RNN predict
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.callbacks import EarlyStopping
from keras.layers import Conv1D
from keras.layers import MaxPooling1D,GlobalAveragePooling1D
from keras.layers import Flatten
from keras.models import load_model
from keras.layers import Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.callbacks import History 
from keras.layers import Activation


with open("mfcc/train.ark",'r') as f:
    mfcc_train = f.readlines()
instance_ID_list = []
mfcc_feature_dict = {}
for line in mfcc_train:
    sp = line.split()
    instance_ID_list.append(sp[0])  #### create instance ID order list
    mfcc_feature_dict[sp[0]]= np.array(sp[1:],dtype=float) ### create mfcc feature
mfcc_feature_dict["None"] = np.zeros(39)

with open("fbank/train.ark",'r') as f:
    fbank_train = f.readlines()

fbank_feature_dict = {}
for line in fbank_train:
    sp = line.split()
    fbank_feature_dict[sp[0]] = np.array(sp[1:],dtype=float)
fbank_feature_dict["None"] = np.zeros(69)

from collections import defaultdict
from sklearn.preprocessing import LabelBinarizer

with open("train.lab","r") as f:
    labels = f.readlines()
    
label_dict = defaultdict()

for label in labels:
    sp = label.split(",")
    label_dict[sp[0]] = sp[1].strip()

# align label with instance id order
label_list = []
for i in instance_ID_list:
    label_list.append(label_dict[i])
# Now the order of label_list is same as features oreder




#add None phone for padding
label_list.append("None")

print("label list:", label_list[:80])



# transform to categorical one-hot format

encoder = LabelBinarizer()
transformed_label = encoder.fit_transform(label_list)

print("transformed label shape", transformed_label.shape)

transformed_label_dict = {}
for i, tr_label in zip(instance_ID_list+["None"],transformed_label):
    transformed_label_dict[i] = tr_label
print("transform label:", dict(list( transformed_label_dict.items())[0:2]))

audio = []
for ins in instance_ID_list:
    sp = ins.split("_")
    audio.append(sp[0]+"_"+sp[1])
print("共",len(np.unique(audio)),"個音檔")
audio_dict = defaultdict(int)
for ins in audio:
    audio_dict[ins] += 1
### create audio index for batch generation

first_index = [0]
for i in range(1,len(audio)):
    if audio[i] != audio[i-1]:
        first_index.append(i)
first_index.append(len(audio))

def cut_audio(audio, lookback = 200, overlap = 10):
    audio_len = len(audio)
    
    if audio_len < lookback:
        result = audio + ["None" for _ in range(lookback - audio_len)]
        return [result]
    

    else:
        # pad the sequence first
        start = 0
        result = []
        for i in range(audio_len//lookback+1):
            seq = audio[start:start+lookback]
            if len(seq) < lookback:
                result.append(seq + ["None" for _ in range(lookback - len(seq))])
            else:
                result.append(seq)
            start = start + lookback - overlap
            
        return(result)

batch=[]

frame_count = 200

for i in range(len(first_index)-1):
    batch += cut_audio(instance_ID_list[first_index[i]:first_index[i+1]],lookback=frame_count, overlap=30)

# feature V2 RNN

train_X = []
train_y = []

for b in batch:
    frame_feature = []
    frame_label = []
    for instance_id in b:
        frame_feature.append(np.hstack((mfcc_feature_dict[instance_id] , fbank_feature_dict[instance_id])))
        frame_label.append(transformed_label_dict[instance_id])
    train_X.append(frame_feature)
    train_y.append(frame_label)

print("feature V2 RNN train_X:", np.shape(train_X))
print("feature V2 RNN train_y:", np.shape(train_y))



model = Sequential()

model.add(LSTM(input_shape = (200,108),units=64,return_sequences= True))
model.add(Dropout(0.5))
model.add(LSTM(input_shape = (200,108),units=128,return_sequences= True))
model.add(Dense(512,activation="relu"))
model.add(Dense(49, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(train_X, train_y, epochs=90, batch_size=128)
model.save('RNN.h5')