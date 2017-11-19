import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
import time
from os import listdir
from keras.preprocessing import sequence

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

data_path = sys.argv[1]
output_file = sys.argv[2]


class Video_Caption_Generator():
    def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_lstm_steps, n_video_lstm_step, n_caption_lstm_step, bias_init_vector=None):
        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps
        self.n_video_lstm_step=n_video_lstm_step
        self.n_caption_lstm_step=n_caption_lstm_step

        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb') # (token_unique, 1000)
        
        self.lstm1 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False) # c_state, m_state are concatenated along the column axis 
        self.lstm2 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False)

        self.encode_image_W = tf.Variable( tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_W') # (4096, 1000)
        self.encode_image_b = tf.Variable( tf.zeros([dim_hidden]), name='encode_image_b')


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

        state1 = tf.zeros([self.batch_size, self.lstm1.state_size]) # initial state
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size]) # initial state
        padding = tf.zeros([self.batch_size, self.dim_hidden]) # (batch, 1000)

        probs = []
        loss = 0.0

        ##############################  Encoding Stage ##################################
        for i in range(0, self.n_video_lstm_step): # n_vedio_lstm_step = 80
            with tf.variable_scope("LSTM1", reuse= (i!=0)):
                output1, state1 = self.lstm1(image_emb[:,i,:], state1)

            with tf.variable_scope("LSTM2", reuse=(i!=0)):
                output2, state2 = self.lstm2(tf.concat( [padding, output1] ,1), state2)

        ############################# Decoding Stage ######################################
        for i in range(0, self.n_caption_lstm_step): ## Phase 2 => only generate captions
            #if i == 0:
            #    current_embed = tf.zeros([self.batch_size, self.dim_hidden])
            #else:
            with tf.device("/cpu:0"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:, i])

            

            with tf.variable_scope("LSTM1",reuse= True):
                output1, state1 = self.lstm1(padding, state1)

            with tf.variable_scope("LSTM2", reuse= True):
                output2, state2 = self.lstm2(tf.concat([current_embed, output1], 1), state2)

            labels = tf.expand_dims(caption[:, i+1], 1) # (batch, max_length, 1)
            indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1) # (batch_size, 1)
            concated = tf.concat([indices, labels], 1)
            onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0)

            logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels= onehot_labels)
            cross_entropy = cross_entropy * caption_mask[:,i]
            probs.append(logit_words)

            current_loss = tf.reduce_sum(cross_entropy)/self.batch_size
            loss = loss + current_loss
            
        return loss, video, video_mask, caption, caption_mask, probs


    def build_generator(self):

        # batch_size = 1
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

            with tf.variable_scope("LSTM2", reuse=(i!=0)):
                output2, state2 = self.lstm2(tf.concat([padding, output1], 1), state2)

        for i in range(0, self.n_caption_lstm_step):
            if i == 0:
                with tf.device('/cpu:0'):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, tf.ones([1], dtype=tf.int64))

            with tf.variable_scope("LSTM1", reuse=True):
                output1, state1 = self.lstm1(padding, state1)

            with tf.variable_scope("LSTM2", reuse=True):
                output2, state2 = self.lstm2(tf.concat([current_embed, output1],1), state2)

            logit_words = tf.nn.xw_plus_b( output2, self.embed_word_W, self.embed_word_b)
            max_prob_index = tf.argmax(logit_words, 1)[0]
            generated_words.append(max_prob_index)
            probs.append(logit_words)

            with tf.device("/cpu:0"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)
                current_embed = tf.expand_dims(current_embed, 0)

            embeds.append(current_embed)

        return video, video_mask, generated_words, probs, embeds



dim_image = 4096
dim_hidden= 256

n_video_lstm_step = 80
n_caption_lstm_step = 20
n_frame_step = 80

n_epochs = 1000
batch_size = 50
learning_rate = 0.0001




ixtoword = pd.Series(np.load('./ixtoword_special.npy').tolist())

bias_init_vector = np.load('./bias_init_vector_special.npy')
model = Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(ixtoword),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_lstm_steps=n_frame_step,
            n_video_lstm_step=n_video_lstm_step,
            n_caption_lstm_step=n_caption_lstm_step,
            bias_init_vector=bias_init_vector)
video_tf, video_mask_tf, caption_tf, probs_tf, last_embed_tf = model.build_generator()
sess = tf.InteractiveSession()
print("start to restore")
saver = tf.train.Saver()
saver.restore(sess, "./models/model-230")
print("restore success")

test_folder_path = data_path + "testing_data/feat/"
test_path = listdir(test_folder_path)
test_features = [ (file[:-4],np.load(test_folder_path + file)) for file in test_path]


test_sentences = []
id_list = []
for idx, video_feat in test_features:
	print(idx)
	video_feat = video_feat.reshape(1,80,4096)
	print(video_feat.shape)
	if video_feat.shape[1] == n_frame_step:
	    video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))

	generated_word_index = sess.run(caption_tf, feed_dict={video_tf:video_feat, video_mask_tf:video_mask})
	print(generated_word_index)
	generated_words = ixtoword[generated_word_index]
	generated_sentence = ' '.join(generated_words)
	generated_sentence = generated_sentence.replace('<bos> ', '')
	generated_sentence = generated_sentence.replace(' <eos>', '')
	generated_sentence = generated_sentence.replace('<pad> ', '')
	generated_sentence = generated_sentence.replace(' <pad>', '')
	print (generated_sentence,'\n')
	if idx in ["klteYv1Uv9A_27_33.avi","5YJaS2Eswg0_22_26.avi","UbmZAe5u5FI_132_141.avi","JntMAcTlOF0_50_70.avi","tJHUH9tpqPg_113_118.avi"]:
		id_list.append(idx)
		test_sentences.append(generated_sentence)

submit = pd.DataFrame(np.array([id_list,test_sentences]).T)
submit.to_csv(output_file,index = False,  header=False)