import os
import logging
import pprint
from rasa.shared.nlu.training_data.loading import load_data
from rasa.nlu import config
from rasa.nlu.model import Trainer
from rasa.nlu.model import Interpreter
from rasa.nlu.test import run_evaluation
import parameters
import os
#logfile = './models/nlu_model.log'


def NER_training(data_path, configs, model_path):
    #logging.basicConfig(filename=logfile, level=logging.DEBUG)
    training_data = load_data(data_path)
    pprint.pprint(training_data)
    trainer = Trainer(config.load(configs))
    trainer.train(training_data)
    model_directory = trainer.persist(model_path, fixed_model_name='nlu')# project_name='current',
    run_evaluation(data_path, model_directory)
    print( os.popen(parameters.sync_local_NER_to_s3).read())
    print('sync with s3 is done, NER_updated')

def s3_to_local():
    print(os.popen(parameters.sync_s3_NER_to_local).read())
    interpreter = Interpreter.load(parameters.NER_model_path+'/nlu')
    return interpreter


def NER_output(text, interpreter):
    #print(os.popen(parameters.sync_s3_NER_to_local).read())
    #logging.basicConfig(filename=logfile, level=logging.DEBUG)
    #interpreter = Interpreter.load(parameters.NER_model_path+'/current/nlu')
    return interpreter.parse(text)['entities']
    #pprint.pprint((interpreter.parse("What are color options in supernova"))['entities'])


# if __name__ == '__main__':
#     NER_training(parameters.NER_train_data, parameters.configs_file, parameters.NER_model_path)
#     NER_output(parameters.NER_model_path+'/current/nlu')