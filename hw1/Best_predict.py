## RNN predict
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import LabelBinarizer
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
frame_count = 200
# load testing data
path = sys.argv[1]
output = sys.argv[2]
with open(path+"mfcc/test.ark",'r') as f:
    mfcc_test = f.readlines()

with open(path+"fbank/test.ark",'r') as f:
    fbank_test = f.readlines()

mfcc_test_dict = {}
test_ID_list = []

for line in mfcc_test:
    sp = line.split()
    test_ID_list.append(sp[0])  #### create instance ID order list
    mfcc_test_dict[sp[0]]= np.array(sp[1:],dtype=float) ### create mfcc feature

fbank_test_dict = {}

for line in fbank_test:
    sp = line.split()
    fbank_test_dict[sp[0]] = np.array(sp[1:],dtype=float)

mfcc_test_dict["None"] = np.zeros(39)   
fbank_test_dict["None"] = np.zeros(69)

with open(path + "mfcc/train.ark",'r') as f:
    mfcc_train = f.readlines()
instance_ID_list = []
for line in mfcc_train:
    sp = line.split()
    instance_ID_list.append(sp[0])  #### create instance ID order list

with open(path + "label/train.lab","r") as f:
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
# transform to categorical one-hot format
encoder = LabelBinarizer()
transformed_label = encoder.fit_transform(label_list)


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


# cut list for each audio
test_audio = []
for ins in test_ID_list:
    sp = ins.split("_")
    test_audio.append(sp[0]+"_"+sp[1])
    

test_first_index = [0]
for i in range(1,len(test_audio)):
    if test_audio[i] != test_audio[i-1]:
        test_first_index.append(i)
test_first_index.append(len(test_audio))

test_batch = []
for i in range(len(test_first_index)-1):
    test_batch += cut_audio(test_ID_list[test_first_index[i]:test_first_index[i+1]],lookback=frame_count, overlap=30)

# testing feature V2
test_X = []


for b in test_batch:
    frame_feature = []

    for instance_id in b:
        frame_feature.append(np.hstack((mfcc_test_dict[instance_id] , fbank_test_dict[instance_id])))
    test_X.append(frame_feature)
print("test_X shape:",np.shape(test_X))

batch_count = np.diff(np.array(test_first_index))//frame_count +1 # 200 是 frame count
total_batch = sum(batch_count)
batch_index = [0] + list(np.cumsum(batch_count))

model = load_model('LSTMx4.h5')
print(model.summary())
pred = model.predict(test_X)
print("predict shape",pred.shape)

pred_audio = []
for i in range(len(batch_index)-1):
    pred_audio.append(pred[batch_index[i]:batch_index[i+1]])

overlap = 30
lookback = frame_count

sample_audio = []
for audio in pred_audio:
    each_audio = []
    for ind, frames in enumerate(audio):
        if ind > 0:
            each_audio.append(frames[overlap:,])
        else:
            each_audio.append(frames)
    each_audio = np.concatenate(each_audio)
    sample_audio.append(each_audio)

pred_trans = [encoder.inverse_transform(aud) for aud in sample_audio]

with open( path+ "48phone_char.map","r") as f:
    phone_char = f.readlines()

phone_char_dict = {}
for i in phone_char:
    sp = i.split("\t")
    phone_char_dict[sp[0]] = sp[2].strip()

    
with open(path + "phones/48_39.map","r") as f:
    phone_phone = f.readlines()
phone_phone_dict = {}
for i in phone_phone:
    sp = i.split("\t")
    phone_phone_dict[sp[0]] = sp[1].strip()

pred_letter = []
for audio in pred_trans:
    audio_letter = []
    for phone in audio:
        if phone != "None":
            audio_letter.append(phone_char_dict[phone_phone_dict[phone]])
    pred_letter.append(audio_letter)



def pred_filter1(pred_letter):
    for index in range(1,len(pred_letter)-1):
        if pred_letter[index-1] == pred_letter[index+1] and pred_letter[index] != pred_letter[index+1]:
            pred_letter[index] = pred_letter[index+1]
        elif pred_letter[index] != pred_letter[index+1] and pred_letter[index] != pred_letter[index-1]:
            pred_letter[index] = pred_letter[index+1]
    return pred_letter

def pred_filter2(pred_letter):
    for index in range(1,len(pred_letter)-2):
        if (pred_letter[index] == pred_letter[index+1] and pred_letter[index] != pred_letter[index+2] 
                and pred_letter[index] != pred_letter[index-1]):
                pred_letter[index] = pred_letter[index+2]
                pred_letter[index+1] = pred_letter[index+2]
    return pred_letter

def trimmer(pred_y):
    char_seq = []
    for instance in pred_y:
        seq = []
        ele = ""
        for char in instance:
            if char != ele:
                seq.append(char)
                ele = char
        #trim "L" in the front or back
        if seq[0] =="L":
            seq = seq[1:]
        if seq[-1]=="L":
            seq = seq[:-1]
        
        char_seq.append("".join(seq))
    return(char_seq)


pred_y = [pred_filter1(aud) for aud in pred_letter]

pred_y = [pred_filter2(aud) for aud in pred_y]
print("pred_y_trimmed")

pred_y_trimmed = trimmer(pred_y)

submit_csv = pd.read_csv("sample.csv")

submit_csv["phone_sequence"] = pred_y_trimmed

submit_csv.to_csv(output,index=False)