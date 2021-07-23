import numpy as np
import pandas as pd
import parameters
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True
from train_transformer import train_model
from flask import Flask, request, jsonify, render_template
from load_TXFR_model import lang_s3_to_local, predict_lang, sync_s3_to_models, predict
import glob
#from train_NER import NER_output, NER_training,s3_to_local 
import pickle
import time
import json
from ast import literal_eval
import re
import glob
import os




start_time = time.clock()

app = Flask(__name__)

unique_label, max_length,t, model,Model_Name= sync_s3_to_models()
from spell_checker import spell_checker
print(len(unique_label))
model_ft, model_ft_dual = lang_s3_to_local()
#interpreter = s3_to_local()


@app.route('/test', methods=['POST'])
def test():
    #data = request.get_json(force=True)
    return jsonify("This is testing API")



@app.route('/results', methods=['POST'])
def language():
    t1=time.time()
    data = request.get_json(force=True)
    text=data["message"]
    text= spell_checker(text)
    text=text.strip()
    text=text.replace('\n','')
    #text=re.sub('[^A-Za-z0-9]+', ' ', text)
    text=re.sub('[!@#$?-]',' ',text)
    print(text)   
    if text.isdigit():
        language = 'en'
        language_confidence = 1.0
    else:
        language, language_confidence = predict_lang(text,model_ft)
        if language == "en":
            language, language_confidence = predict_lang(text,model_ft_dual)
    output = predict(text,unique_label, max_length,t,model,Model_Name)
    #output['entity']=NER_output(text,interpreter)
    output['intent']['language'] =language
    output['intent']['language_confidence'] = language_confidence

    return jsonify(output)    


@app.route('/train/<model>', methods=['POST'])
def train(model):
    data=request.data
    my_json = literal_eval(data.decode('utf8'))
    #print(my_json)
    my_json1=my_json['intentData']
    print(my_json1)
    df = json.dumps(my_json1, indent=4, sort_keys=True)
    c=pd.DataFrame(eval(df))
    c.to_excel(parameters.training_data)
    '''
    entity_data = my_json["entityData"]
    print(type(entity_data))
    print(entity_data[1])
    new_dict = {
  		"rasa_nlu_data": {
    		"common_examples": entity_data
  		}
	}
    with open(parameters.NER_train_data, 'w') as outfile:
        json.dump(new_dict, outfile)
    NER_training(parameters.NER_train_data, parameters.configs_file, parameters.NER_model_path)
    '''
    model_name = model.replace(" ","")
    print(model_name)
    print(type(model_name))
    result = {"result":train_model(model_name)}
    result={"result":"please wait"}
    return jsonify(result)

print (time.clock() - start_time, "Seconds")

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0",threaded=False)

