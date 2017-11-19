import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
import time
import cv2
from keras.preprocessing import sequence
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from os import listdir
import json
import math
import operator
from functools import reduce
import random
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


dim_image = 4096
dim_hidden= 256
n_video_lstm_step = 80
n_caption_lstm_step = 15
n_frame_step = 80
schedule_sample_probability = 1
n_epochs = 1000
batch_size = 32
learning_rate = 0.001
decay_epoch = 30



class Video_Caption_Generator():
    def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_lstm_steps, n_video_lstm_step
                 ,n_caption_lstm_step,schedule_p, bias_init_vector=None):
        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps
        self.n_video_lstm_step=n_video_lstm_step
        self.n_caption_lstm_step=n_caption_lstm_step
        self.schedule_p = schedule_p

        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb') # (token_unique, 1000)
        
        self.lstm1 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False) # c_state, m_state are concatenated along the column axis 
        self.lstm2 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False)

        self.encode_image_W = tf.Variable( tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_W') # (4096, 1000)
        self.encode_image_b = tf.Variable( tf.zeros([dim_hidden]), name='encode_image_b')
        # variable for attention (hWz)
        self.attention_z = tf.Variable(tf.random_uniform([self.batch_size,self.lstm2.state_size],-0.1,0.1), name="attention_z")
        self.attention_W = tf.Variable(tf.random_uniform([self.lstm1.state_size,self.lstm2.state_size],-0.1,0.1),name="attention_W")
    
        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1,0.1), name='embed_word_W') # (1000, n_words)
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

    def build_model(self):
        video = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step, self.dim_image]) # (batch, 80, 4096)
        video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step])

        caption = tf.placeholder(tf.int32, [self.batch_size, self.n_caption_lstm_step+1]) # enclude <BOS>; store word ID; (batch, max_length)
        caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_caption_lstm_step+1]) # (batch_size, max_length+1)
        video_flat = tf.reshape(video, [-1, self.dim_image]) 
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b ) # (batch_size*n_lstm_steps, dim_hidden)
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_lstm_steps, self.dim_hidden])
        print("lstm1 sate size,",self.lstm1.state_size)
        print("lstm2 sate size,",self.lstm2.state_size) # 2*hidden size 
        state1 = tf.zeros([self.batch_size, self.lstm1.state_size]) # initial state
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size]) # initial state
        padding = tf.zeros([self.batch_size, self.dim_hidden]) # (batch, 1000)

        probs = []
        loss = 0.0
        
        ##############################  Encoding Stage ##################################
        context_padding = tf.zeros([self.batch_size, self.lstm2.state_size]) #(batch_size, 2000)
        h_list = []
        for i in range(0, self.n_video_lstm_step): # n_vedio_lstm_step = 80
            with tf.variable_scope("LSTM1", reuse= (i!=0)):
                output1, state1 = self.lstm1(image_emb[:,i,:], state1)
                h_list.append(state1)
            with tf.variable_scope("LSTM2", reuse=(i!=0)):
                output2, state2 = self.lstm2(tf.concat( [padding, output1, context_padding] ,1), state2)
        print(np.shape(h_list))
        h_list = tf.stack(h_list,axis=1) 
        print(np.shape(h_list)) # (64, 80, 2000)
        ############################# Decoding Stage ######################################
        for i in range(0, self.n_caption_lstm_step): ## Phase 2 => only generate captions
            if i==0:
                with tf.device("/cpu:0"):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:, i])
            else: # schedule sampling
                print(self.schedule_p)
                if(np.random.binomial(1,self.schedule_p)==1): # schedule_p 擲骰子值出來是1的機率
                    with tf.device("/cpu:0"):
                        current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:, i])
                else:
                    max_prob_index = tf.argmax(logit_words, 1)[0]
                    with tf.device("/cpu:0"):
                        current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)
            with tf.variable_scope("LSTM1",reuse= True):
                output1, state1 = self.lstm1(padding, state1)
            ##### attention ####
            context = []
            if i == 0:
                new_z = self.attention_z
            # h_list_flat = tf.reshape(h_list,[-1,self.lstm1.state_size])
            # print("h_list_flat shape, ", h_list_flat.shape) # 5120,2000
            
#             for sample in range(0, self.batch_size):
#                 alpha_list = [] # a list to store alpha"s" in each training sample
#                 for step_ in range(0,self.n_video_lstm_step):
#                     alpha =1 - tf.losses.cosine_distance(h_list[sample,step_,:], new_z[sample,:], dim=0)
#                     alpha_list.append(alpha)
#                 alpha_list = tf.expand_dims(alpha_list,1)
#                 ci = tf.reduce_sum(tf.multiply(alpha_list, h_list[sample,:,:]),axis = 0)
#                 context.append(ci)
#             context = tf.stack(context)
#             print("context shape", content.shape)
            h_list_flat = tf.reshape(h_list,[-1,self.lstm1.state_size])
            htmp = tf.matmul(h_list_flat,self.attention_W) # for matmul operation (5120,2000)
            hW = tf.reshape(htmp,[self.batch_size, self.n_video_lstm_step,self.lstm2.state_size])
            for x in range(0,self.batch_size):
                x_alpha = tf.reduce_sum(tf.multiply(hW[x,:,:],new_z[x,:]),axis=1)
                x_alpha = tf.nn.softmax(x_alpha)
                x_alpha = tf.expand_dims(x_alpha,1)
                x_new_z = tf.reduce_sum(tf.multiply(x_alpha,h_list[x,:,:]),axis=0)
                context.append(x_new_z) 
            context = tf.stack(context)
            print("context shape", context.shape)
            with tf.variable_scope("LSTM2", reuse= True):
                print(output1.shape) # (64,1000)
                output2, state2 = self.lstm2(tf.concat([current_embed, output1, context], 1), state2)
                new_z = state2
        
            labels = tf.expand_dims(caption[:, i+1], 1) # (batch, max_length, 1)
            indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1) # (batch_size, 1)
            concated = tf.concat([indices, labels], 1)
            onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0)

            logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b) #probability of each word
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels= onehot_labels)
            cross_entropy = cross_entropy * caption_mask[:,i]
            
            
                    
            probs.append(logit_words)

            current_loss = tf.reduce_sum(cross_entropy)/self.batch_size
            loss = loss + current_loss
            
        return loss, video, video_mask, caption, caption_mask, probs


    def build_generator(self):

        # batch_size = 1
        context_padding = tf.zeros([1, self.lstm2.state_size])
        h_list = []
        video = tf.placeholder(tf.float32, [1, self.n_video_lstm_step, self.dim_image]) # (80, 4096)
        video_mask = tf.placeholder(tf.float32, [1, self.n_video_lstm_step])

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)
        image_emb = tf.reshape(image_emb, [1, self.n_video_lstm_step, self.dim_hidden])

        state1 = tf.zeros([1, self.lstm1.state_size])
        state2 = tf.zeros([1, self.lstm2.state_size])
        padding = tf.zeros([1, self.dim_hidden])

        generated_words = []

        probs = []
        embeds = []

        for i in range(0, self.n_video_lstm_step):
            with tf.variable_scope("LSTM1", reuse=(i!=0)):
                output1, state1 = self.lstm1(image_emb[:, i, :], state1)
                h_list.append(state1)

            with tf.variable_scope("LSTM2", reuse=(i!=0)):
                output2, state2 = self.lstm2(tf.concat([padding, output1, context_padding], 1), state2)
        h_list = tf.stack(h_list,axis=1) 
        for i in range(0, self.n_caption_lstm_step):
            if i == 0:
                with tf.device('/cpu:0'):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, tf.ones([1], dtype=tf.int64))

            with tf.variable_scope("LSTM1", reuse=True):
                output1, state1 = self.lstm1(padding, state1)

            with tf.variable_scope("LSTM2", reuse=True):
                context = []
                if i == 0:
                    new_z = self.attention_z
                h_list_flat = tf.reshape(h_list,[-1,self.lstm1.state_size])
                htmp = tf.matmul(h_list_flat,self.attention_W)
                hW = tf.reshape(htmp, [1, self.n_video_lstm_step,self.lstm1.state_size])
                for x in range(0,1): # only one sample 
                    x_alpha = tf.reduce_sum(tf.multiply(hW[x,:,:],new_z[x,:]),axis=1)
                    x_alpha = tf.nn.softmax(x_alpha)
                    x_alpha = tf.expand_dims(x_alpha,1)
                    x_new_z = tf.reduce_sum(tf.multiply(x_alpha,h_list[x,:,:]),axis=0)
                    context.append(x_new_z)
                context = tf.stack(context)
                output2, state2 = self.lstm2(tf.concat([current_embed, output1,context],1), state2)
                new_z = state2
            logit_words = tf.nn.xw_plus_b( output2, self.embed_word_W, self.embed_word_b)
            max_prob_index = tf.argmax(logit_words, 1)[0]
            generated_words.append(max_prob_index)
            probs.append(logit_words)

            with tf.device("/cpu:0"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)
                current_embed = tf.expand_dims(current_embed, 0)

            embeds.append(current_embed)

        return video, video_mask, generated_words, probs, embeds


## bleu score
def count_ngram(candidate, references, n):
    clipped_count = 0
    count = 0
    r = 0
    c = 0
    for si in range(len(candidate)):
        # Calculate precision for each sentence
        ref_counts = []
        ref_lengths = []
        # Build dictionary of ngram counts
        for reference in references:
            ref_sentence = reference[si]
            ngram_d = {}
            words = ref_sentence.strip().split()
            ref_lengths.append(len(words))
            limits = len(words) - n + 1
            # loop through the sentance consider the ngram length
            for i in range(limits):
                ngram = ' '.join(words[i:i+n]).lower()
                if ngram in ngram_d.keys():
                    ngram_d[ngram] += 1
                else:
                    ngram_d[ngram] = 1
            ref_counts.append(ngram_d)
        # candidate
        cand_sentence = candidate[si]
        cand_dict = {}
        words = cand_sentence.strip().split()
        limits = len(words) - n + 1
        for i in range(0, limits):
            ngram = ' '.join(words[i:i + n]).lower()
            if ngram in cand_dict:
                cand_dict[ngram] += 1
            else:
                cand_dict[ngram] = 1
        clipped_count += clip_count(cand_dict, ref_counts)
        count += limits
        r += best_length_match(ref_lengths, len(words))
        c += len(words)
    if clipped_count == 0:
        pr = 0
    else:
        pr = float(clipped_count) / count
    bp = brevity_penalty(c, r)
    return pr, bp


def clip_count(cand_d, ref_ds):
    """Count the clip count for each ngram considering all references"""
    count = 0
    for m in cand_d.keys():
        m_w = cand_d[m]
        m_max = 0
        for ref in ref_ds:
            if m in ref:
                m_max = max(m_max, ref[m])
        m_w = min(m_w, m_max)
        count += m_w
    return count


def best_length_match(ref_l, cand_l):
    """Find the closest length of reference to that of candidate"""
    least_diff = abs(cand_l-ref_l[0])
    best = ref_l[0]
    for ref in ref_l:
        if abs(cand_l-ref) < least_diff:
            least_diff = abs(cand_l-ref)
            best = ref
    return best


def brevity_penalty(c, r):
    if c > r:
        bp = 1
    else:
        bp = math.exp(1-(float(r)/c))
    return bp


def geometric_mean(precisions):
    return (reduce(operator.mul, precisions)) ** (1.0 / len(precisions))


def BLEU(s,t,flag = False):

    score = 0.  
    count = 0
    candidate = [s.strip()]
    if flag:
        references = [[t[i].strip()] for i in range(len(t))]
    else:
        references = [[t.strip()]] 
    precisions = []
    pr, bp = count_ngram(candidate, references, 1)
    precisions.append(pr)
    score = geometric_mean(precisions) * bp
    return score


def preProBuildWordVocab(sentence_iterator, word_count_threshold=5):
    # borrowed this function from NeuralTalk
    print ('preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold))
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in sent.lower().split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print ('filtered words from %d to %d' % (len(word_counts), len(vocab)))

    ixtoword = {}
    ixtoword[0] = '<pad>'
    ixtoword[1] = '<bos>'
    ixtoword[2] = '<eos>'
    ixtoword[3] = '<unk>'

    wordtoix = {}
    wordtoix['<pad>'] = 0
    wordtoix['<bos>'] = 1
    wordtoix['<eos>'] = 2
    wordtoix['<unk>'] = 3

    for idx, w in enumerate(vocab):
        wordtoix[w] = idx+4
        ixtoword[idx+4] = w

    word_counts['<pad>'] = nsents
    word_counts['<bos>'] = nsents
    word_counts['<eos>'] = nsents
    word_counts['<unk>'] = nsents

    bias_init_vector = np.array([1.0 * word_counts[ ixtoword[i] ] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
    
    return wordtoix, ixtoword, bias_init_vector

train_feature_folder = "MLDS_hw2_data/training_data/feat/"
path_list = listdir(train_feature_folder)
train_feature_path = [train_feature_folder+path for path in path_list]
#### train_ID_list ####
train_ID_list = [ path[:-4] for path in path_list]


#### train_feature_dict ####
train_feature_dict = {}
for path in train_feature_path:
    feature = np.load(path)
    train_feature_dict[path[:-4].replace("MLDS_hw2_data/training_data/feat/","")] = feature

def clean_string(string):
    return string.replace('.', '').replace(',', '').replace('"', '').replace('\n', '').replace('?', '').replace('!', '').replace('\\', '').replace('/', '')

train_label_dict={}
with open('MLDS_hw2_data/training_label.json') as data_file:    
    train_label = json.load(data_file)
captions_corpus = []
for sample in train_label:
    cleaned_captions = [clean_string(sentence) for sentence in sample["caption"]]
    captions_corpus += cleaned_captions
    train_label_dict[sample["id"]] = cleaned_captions


wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions_corpus, word_count_threshold=1)
np.save("./wordtoix_pyf", wordtoix)
np.save('./ixtoword_pyf', ixtoword)
np.save("./bias_init_vector_pyf", bias_init_vector)


model = Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(wordtoix),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_lstm_steps=n_frame_step,
            n_video_lstm_step=n_video_lstm_step,
            n_caption_lstm_step=n_caption_lstm_step,
            bias_init_vector=bias_init_vector,schedule_p = schedule_sample_probability)

tf_loss, tf_video, tf_video_mask, tf_caption, tf_caption_mask, tf_probs= model.build_model()

## testing graph
video_tf, video_mask_tf, caption_tf, probs_tf, last_embed_tf = model.build_generator()
test_folder_path = "./MLDS_hw2_data/testing_data/feat/"
test_path = listdir(test_folder_path)
test_features = [ (file[:-4],np.load(test_folder_path + file)) for file in test_path]
test = json.load(open('MLDS_hw2_data/testing_label.json','r'))
ixtoword_series = pd.Series(np.load('./ixtoword_pyf.npy').tolist())


sess = tf.InteractiveSession()
saver = tf.train.Saver(max_to_keep=100)

train_op = tf.train.AdamOptimizer(0.001).minimize(tf_loss)
tf.global_variables_initializer().run()

loss_fd = open('loss3_pyf.txt', 'w')
loss_to_draw = []


X_y_pairs = []
target_texts = []

words_list = []
for ID in train_ID_list:
    
#     text = np.random.choice(train_label_dict[ID],1)[0]
    for text in train_label_dict[ID]:
        X_y_pairs.append((train_feature_dict[ID],text))
        words = text.split()
        target_texts.append(words)
        for word in words:
            words_list.append(word)
target_words_set = np.unique(words_list, return_counts=True)[0]
num_decoder_tokens = len(target_words_set)
max_decoder_seq_length = max([len(txt) for txt in target_texts])
print("sample counts", np.shape(X_y_pairs))
print("number of decoder tokens", num_decoder_tokens)
print("max length of label", max_decoder_seq_length)

print("tuple first element shape", X_y_pairs[1][0].shape)
print("tuple second element:", X_y_pairs[1][1])



print("tuple second element:", X_y_pairs[1][1])
loss_to_draw_epoch = []
model_path = './models/pyf'
sample_size = 1450
print("Total sample:", sample_size)
for epoch in range(0, n_epochs):
    random.shuffle(X_y_pairs)
    X_y_pairs_sub =  random.sample(X_y_pairs,sample_size)
    # modify schedule p
    if epoch<decay_epoch:
        model.schedule_p = 1
    else:
        #linear
        model.schedule_p = np.max([1-(epoch/decay_epoch-1), 0])
        
        #inversesigmoid decay
        #model.schedule_p = decay_epoch/(decay_epoch+np.exp(epoch/decay_epoch))
    for batch_start, batch_end in zip(range(0, sample_size, batch_size), range(batch_size, sample_size, batch_size)):
        start_time = time.time()
        current_batch = X_y_pairs_sub[batch_start:batch_end]
        current_feats = [ row[0] for row in current_batch]
        current_video_masks = np.zeros((batch_size, n_video_lstm_step))

        current_captions = np.array(["<bos> "+ row[1] for row in current_batch])
        for idx, single_caption in enumerate(current_captions):
            word = single_caption.lower().split(" ")
            if len(word) < n_caption_lstm_step:
                current_captions[idx] = current_captions[idx] + " <eos>"
            else:
                new_word = ""
                for i in range(n_caption_lstm_step-1):
                    new_word = new_word + word[i] + " "
                current_captions[idx] = new_word + "<eos>"
        current_caption_ind = []
        for cap in current_captions:
            current_word_ind = []
            for word in cap.lower().split(' '):
                if word in wordtoix:
                    current_word_ind.append(wordtoix[word])
                else:
                    current_word_ind.append(wordtoix['<unk>'])
            current_caption_ind.append(current_word_ind)

        current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=n_caption_lstm_step)
        current_caption_matrix = np.hstack( [current_caption_matrix, np.zeros( [len(current_caption_matrix), 1] ) ] ).astype(int)
        current_caption_masks = np.zeros( (current_caption_matrix.shape[0], current_caption_matrix.shape[1]) )
        nonzeros = np.sum((current_caption_matrix != 0),1) # 算每個row 有幾個字
        for ind, row in enumerate(current_caption_masks):
            row[:nonzeros[ind]] = 1 # 把前幾個有字的在mask上塗成1

        probs_val = sess.run(tf_probs, feed_dict={
                    tf_video:current_feats,
                    tf_caption: current_caption_matrix
                    })
        _, loss_val = sess.run(
                        [train_op, tf_loss],
                        feed_dict={
                            tf_video: current_feats,
                            tf_video_mask : current_video_masks,
                            tf_caption: current_caption_matrix,
                            tf_caption_mask: current_caption_masks
                            })
        loss_to_draw_epoch.append(loss_val)
    print (" Epoch: ", epoch, " loss: ", loss_val, ' Elapsed time: ', str((time.time() - start_time)))
    loss_fd.write('epoch ' + str(epoch) + ' loss ' + str(loss_val) + '\n')
    
    test_sentences = []
    id_list = []

    #validation
    for idx, video_feat in test_features:
        video_feat = video_feat.reshape(1,80,4096)
        if video_feat.shape[1] == n_frame_step:
            video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))
        generated_word_index = sess.run(caption_tf, feed_dict={video_tf:video_feat, video_mask_tf:video_mask})
        generated_words = ixtoword_series[generated_word_index]
        generated_sentence = ' '.join(generated_words)
        generated_sentence = generated_sentence.replace('<bos> ', '')
        generated_sentence = generated_sentence.replace(' <eos>', '')
        generated_sentence = generated_sentence.replace('<pad> ', '')
        generated_sentence = generated_sentence.replace(' <pad>', '')
        generated_sentence = generated_sentence.replace(' <unk>', '')
        id_list.append(idx)
        test_sentences.append(generated_sentence)
        if idx in ["klteYv1Uv9A_27_33.avi","UbmZAe5u5FI_132_141.avi","wkgGxsuNVSg_34_41.avi",
                   "JntMAcTlOF0_50_70.avi","tJHUH9tpqPg_113_118.avi"] and np.mod(epoch, 5) == 0:
            print(generated_sentence)
    submit = pd.DataFrame(np.array([id_list,test_sentences]).T)
    submit.to_csv("ADL_hw2_pyf.txt",index = False,  header=False)
    result = {}
    with open("ADL_hw2_pyf.txt",'r') as f:
        for line in f:
            line = line.rstrip()
            comma = line.index(',')
            test_id = line[:comma]
            caption = line[comma+1:]
            result[test_id] = caption
    #count by average
    bleu=[]
    for item in test:
        score_per_video = []
        for caption in item['caption']:
            caption = caption.rstrip('.')
            score_per_video.append(BLEU(result[item['id']],caption))
        bleu.append(sum(score_per_video)/len(score_per_video))
    average1 = sum(bleu) / len(bleu)
    print("Originally, average bleu score is " + str(average1))
    #count by the method described in the paper https://aclanthology.info/pdf/P/P02/P02-1040.pdf
    bleu=[]
    for item in test:
        score_per_video = []
        captions = [x.rstrip('.') for x in item['caption']]
        score_per_video.append(BLEU(result[item['id']],captions,True))
        bleu.append(score_per_video[0])
    average = sum(bleu) / len(bleu)
    print("By another method, average bleu score is " + str(average))
    
    loss_to_draw.append(np.mean(loss_to_draw_epoch))
    plt_save_dir = "./loss_imgs"
    plt_save_img_name = str(epoch) + '.png'
    plt.plot(range(len(loss_to_draw)), loss_to_draw, color='g')
    plt.grid(True)
    plt.savefig(os.path.join(plt_save_dir, plt_save_img_name))

    if np.mod(epoch, 10) == 0:
        print ("Epoch ", epoch, " is done. Saving the model ...")
        saver.save(sess, os.path.join(model_path, 'model'+str(average1)[2:6]), global_step=epoch)