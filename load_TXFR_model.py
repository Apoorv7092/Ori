import pandas as pd
import re
import fasttext
import time
from tensorflow import keras
import pickle
from keras.preprocessing.sequence import pad_sequences
import numpy
import glob
import os
import shutil
import re
import parameters
from shutil import copyfile

def lang_s3_to_local():
    print(os.popen(parameters.lang_syncs3_to_local).read())
    model_ft = fasttext.load_model("/home/ubuntu/projects/MUSE_lang/langdetect.bin")
    print(os.popen(parameters.hinglish_syncs3_to_local).read())
    model_ft_dual = fasttext.load_model("/home/ubuntu/projects/MUSE_Dual/langdetect_hinglish.bin")
    return model_ft, model_ft_dual



t1 = time.time()
def predict_lang(sent,model_ft):
    prediction = model_ft.predict(sent)
    lang = prediction[0][0].split("__label__")[1]
    confidence = round(prediction[1][0],3)
    return (lang, confidence)
t2 = time.time()
print("Time taken for language prediction:", t2-t1)
print("done")



def sync_s3_to_models():
    if not os.path.exists(parameters.training_data_path):
        os.makedirs(parameters.training_data_path)
    print(os.popen(parameters.syncs3_to_local).read())
    print('sync s3 with models directory is done!!')

    models=parameters.model_path

    list_of_files = glob.glob(models+'/*')
    model_list=[]
    for file in list_of_files:
        head, tail = os.path.split(file)
        name=tail.split("_")
        model_list.append(int(name[-1]))
        
    latest_model=max(model_list)
    print(latest_model)

    latest_file= models+'/model_'+str(latest_model)
    print(latest_file)
    

    def load_components(folder_name):
        unique_label = '{}/vocab_intent_rev.pkl'.format(folder_name)
        max_length = '{}/maxlen.pkl'.format(folder_name)
        t = '{}/tokenizer.pkl'.format(folder_name)
        model_file = '{}/my_modelB'.format(folder_name)
        print(model_file)
        file_mode = 'rb'

        with open(unique_label, file_mode) as file:
            unique_label = pickle.load(file)
        with open(max_length, file_mode) as file:
            max_length = pickle.load(file)
        with open(t, file_mode) as file:
            t = pickle.load(file)
            
        try:
            copyfile(folder_name+"/training.xlsx",parameters.training_data)
        except:
            pass

        model = keras.models.load_model(model_file)
        Model_Name = latest_model
        return unique_label, max_length,t, model,Model_Name


    #unique_label, max_length,t,word_index,model,Model_Name = load_components(latest_file)
    while list_of_files:
        try:
            unique_label, max_length,t,model,Model_Name = load_components(latest_file)
            break
        except:
            list_of_files.remove(latest_file)
            latest_file = max(list_of_files, key=os.path.getctime)
   
    print(latest_file)
    print(model.summary())
    Model_Name = latest_model
    return unique_label, max_length,t,model,Model_Name
    

def predict(utterance,unique_label, max_length,t,model,Model_Name):
    if utterance.isdigit():
        return {"intent": {"confidence": 0.999,"message": utterance,"name": 'none', "model_type":'classifier',"model_name":Model_Name}}
    else:
        utter = [utterance]
        encoded_doc = t.texts_to_sequences(utter)
        padded_doc = pad_sequences(encoded_doc, maxlen=max_length, padding='post')
        pred = model.predict(padded_doc)
        target = numpy.argmax(pred)
        conf_score = round(pred[0][target],7)
        final_prediction = unique_label[target]

       
        return {"intent": {"confidence":(str(conf_score)),"message": utterance,"name": final_prediction, "model_type":'classifier',"model_name":Model_Name}}
        
# while True:
#     text= input("please enter input: ")
#     print(predict(text))
