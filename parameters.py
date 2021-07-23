
### classifier params
model_path= "/home/ubuntu/projects/classifier_models"
training_data_path='/home/ubuntu/projects/training_data'
training_data='/home/ubuntu/projects/training_data/training.xlsx'
embedding_path='/home/ubuntu/projects/vodafone_emb_100.bin'
file_for_tokenizer='/home/ubuntu/projects/MUSE_embeddings/eng_voda_processed.txt'

### clean and s3 sync params
syncs3_to_local='aws s3 sync s3://oriserve-dev-nlp/bajaj-3wh/bajaj-3wh_Dev/classifier_models /home/ubuntu/projects/classifier_models'
sync_local_to_s3='aws s3 sync /home/ubuntu/projects/classifier_models  s3://oriserve-dev-nlp/bajaj-3wh/bajaj-3wh_Dev/classifier_models'
clean_s3="aws s3 ls s3://oriserve-dev-nlp/bajaj-3wh/bajaj-3wh_Dev/classifier_models/ --recursive | head -n -14 | awk '{print $4}' | xargs -I {} aws s3 rm s3://oriserve-dev-nlp/{}"
clean_local="ls -1 /home/ubuntu/projects/classifier_models | grep model | grep -v grep | head -n -2 | xargs -I {} -d '\n' rm -r /home/ubuntu/projects/classifier_models/{}"
sync_s3_NER_to_local= "aws s3://oriserve-dev-nlp/bajaj-3wh/bajaj-3wh_Dev/NER_models /home/ubuntu/projects/NER_models"
sync_local_NER_to_s3= " aws s3 sync /home/ubuntu/projects/NER_models  s3://oriserve-dev-nlp/bajaj-3wh/bajaj-3wh_Dev/NER_models"

# hinglish language model path
hinglish_syncs3_to_local='aws s3 sync s3://oriserve-dev-nlp/bajaj-3wh/bajaj-3wh_Dev/MUSE_dual /home/ubuntu/projects/MUSE_Dual'
#hinglish_sync_local_to_s3='aws s3 sync /home/ubuntu/projects/MUSE_Dual s3://oriserve-dev-nlp/bajaj-3wh/bajaj-3wh_Dev/MUSE_Dual'
#hinglish_clean_s3="aws s3 ls s3://oriserve-dev-nlp/bajaj-3wh/bajaj-3wh_Dev/MUSE_Dual/ --recursive | head -n -1 | awk '{print $4}' | xargs -I {} aws s3 rm s3://oriserve-dev-nlp/{}"
#hinglish_clean_local="ls -1 /home/ubuntu/projects/MUSE_Dual | grep model | grep -v grep | head -n -1 | xargs -I {} -d '\n' rm -r /home/ubuntu/projects/MUSE_Dual/{}"

lang_syncs3_to_local = "aws s3 sync s3://oriserve-vodafone/OCS/oriNLP/Environment/Development/staticModels/muse/ /home/ubuntu/projects/MUSE_lang"

### spellchecker params
spell_checker_file='/home/ubuntu/fastTextModelUpdated/spell_checker_file.txt'

### NER params
NER_train_data='/home/ubuntu/projects/training_data/NER_data.json'
configs_file='/home/ubuntu/fastTextModelUpdated/config.yml'
NER_model_path='/home/ubuntu/projects/NER_models'

