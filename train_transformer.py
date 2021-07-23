
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, TimeDistributed, Conv1D, MaxPooling1D
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint

import fasttext

import parameters
from os import path, mkdir
import pickle
import time
import json
import re
import glob
import os
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from flask import Flask, request, jsonify, render_template
from datetime import datetime
from gensim.models import FastText
from numpy import zeros, asarray, argmax
import numpy as np
import pandas as pd

start_time = time.clock()



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import shutil
from shutil import copyfile



class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output




class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.2):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim,embedding_matrix):
        super(TokenAndPositionEmbedding, self).__init__()
        # self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        # self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=maxlen, weights=[embedding_matrix], trainable=True)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions




def train_model(model_name):
    t1=time.time()
    os.popen(parameters.clean_local)
    pwd=parameters.model_path
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    mkdir(pwd+'/model_{}/'.format(now))
    save_foler_name = pwd+'/model_{}/'.format(now)

    file=parameters.training_data
    data = pd.read_excel(file)
    data=data.sample(frac=1)
    data["message"] = data["message"].str.lower()
    #data["message"] = data["message"].map(lambda x: re.sub('[!@#$?]',' ',x))
    print(len(data))

    message_col = 'message'
    intent_col = 'intent'


    texts = []
    intents = []
    for i, j in data.iterrows():
        texts.append(str(j[message_col]))
        intents.append(j[intent_col])
    distinct_intents = list(set(intents))
    
    with open('/home/ubuntu/projects/MUSE_embeddings/word2id_150000.pkl', 'rb') as file:
        word2id = pickle.load(file)
    with open('/home/ubuntu/projects/MUSE_embeddings/id2word_150000.pkl', 'rb') as file:    
        id2word = pickle.load(file)
    embeddings = np.load('/home/ubuntu/projects/MUSE_embeddings/multilingual_embedding_150000.npy')


    copyfile(parameters.training_data, save_foler_name+"training.xlsx")
    with open(parameters.file_for_tokenizer, "r") as word_list:
        text_for_tokenizer = word_list.read().split('\n')
    text_for_tokenizer.extend(texts)
    t = Tokenizer()
    # t.fit_on_texts(texts)
    t.fit_on_texts(text_for_tokenizer)
    vocab_size = len(t.word_index) + 1

    tokenizer_file_name = 'tokenizer.pkl'
    with open(save_foler_name + tokenizer_file_name, 'wb') as file:
        pickle.dump(t, file)



    vocab_intent = {}
    sentence_per_intent = {}
    for i, intent in enumerate(distinct_intents):
        sentence_per_intent[intent] = intents.count(intent)
        vocab_intent[intent] = i
    vocab_intent_size = len(vocab_intent)

    vocab_intent_rev = {v:k for k, v in vocab_intent.items()}

    vocab_intent_rev_file_name = 'vocab_intent_rev.pkl'
    with open(save_foler_name + vocab_intent_rev_file_name, 'wb') as file:
        pickle.dump(vocab_intent_rev, file)


    X = t.texts_to_sequences(texts)
    y = [] 
    for intent in intents:
        y.append(vocab_intent[intent])
    y = to_categorical(y, num_classes=len(vocab_intent))


    maxlen = 30
    maxlen_file_name = 'maxlen.pkl'
    with open(save_foler_name + maxlen_file_name, 'wb') as file:
        pickle.dump(maxlen, file)
   

    embed_dim = 100 # use from [100, 200, 300] if you're using glove

    X = pad_sequences(X, maxlen=maxlen, padding='post')
    
    
    word_index=t.word_index
    vocab_size=len(word_index)+1
    embedding_matrix = zeros((vocab_size, 100))
    for word, i in word_index.items():
        try:
            embedding_vector = embeddings[word2id[word]]
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        except:
            pass
    
    print("embedding_matrix",embedding_matrix.shape)
   
    num_heads=4
    ff_dim = 100
    embed_dim=100
         
    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim,embedding_matrix)
    x_ = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x_ = transformer_block(x_)
    x_ = layers.GlobalAveragePooling1D()(x_)
    x_ = layers.Dropout(0.2)(x_)
    # x_ = layers.Dense(20, activation="relu")(x_)
    # x_ = layers.Dropout(0.1)(x_)
    outputs = layers.Dense(vocab_intent_size, activation="softmax")(x_)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model_file_name = save_foler_name + '{}' .format(model_name)

    checkpoint = ModelCheckpoint(model_file_name, verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
    
    model.summary()
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])

    if len(data)<=50:
        batch_size=2
        epochs=5
    elif 50<len(data)<=100:
        batch_size=4
        epochs= 5
    elif 100<len(data)<=500:
        batch_size=4
        epochs =5
    elif 500<len(data)<=1500:
        batch_size=4
        epochs=5
    else:
        batch_size=6
        epochs=10

    print('Batch_size= '+ str(batch_size))
    print('Epochs= '+ str(epochs))  



    history = model.fit(X, y, epochs=epochs, verbose=2,callbacks=[checkpoint], batch_size=batch_size,validation_split=0.1)
  
    result = "Your bot is getting trained. Please wait for 10 minutes and restart your server."
    t2=time.time()
    print(t2-t1)
    print( os.popen(parameters.sync_local_to_s3).read())
    print('sync with s3 is done, restarting server')
    print(os.popen(parameters.clean_s3).read())
    os.popen('sudo systemctl restart gunicorn.service')
    print('restarting gunicorn services')
    return result   
    
    
    


# train_model('my_model')




