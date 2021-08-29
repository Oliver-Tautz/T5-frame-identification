from preprocessing import get_train_data,get_data
from model import get_model, get_tokenizer
import defaults
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
from rouge import Rouge
from os.path import isdir, join
from os import makedirs
import pickle
from sklearn.metrics import f1_score
from collections import defaultdict

import argparse

parser = argparse.ArgumentParser(description='train and evaluate a T5 on the Webis-Framing-19 dataset.')
parser.add_argument('--model-savename', help="Name the new, trained model will be saved as.", type=str,default=defaults.model_savename)
parser.add_argument('--pretrained-name', help="Name of the pretrained model to use. Use 't5-small' for smaller checkpoint. Checkpoint in 'trained_models' folder can also be used.",type=str,default=defaults.pretrained_name)
parser.add_argument('--epochs', help="Train for that many epochs", type=int, default=defaults.epochs)
parser.add_argument('--batchsize', help="Use batchsize for training and evaluation. 6 works for t5-base on colab. t5-small can use 24.", type=int, default=defaults.batchsize)
parser.add_argument('--val-split', help="portion of the data to be used for val set.", type=float, default=defaults.val_split)
parser.add_argument('--test-split', help="portion of the data to be used for test set.", type=float, default=defaults.val_split)
parser.add_argument('--data-file', help="absolute or relative path to Webis-Framing-19 dataset csv file.", type=str, default=defaults.data_csv_file)
parser.add_argument('--debug', help="Use small dataset and model for testing the code.", action='store_true')


args = parser.parse_args()

# make trained_models dir if not existing.
makedirs(defaults.models_path,exist_ok=True)

PRETRAINED_NAME = args.pretrained_name
MODEL_FILENAME = join(defaults.models_path,args.model_savename)
EPOCHS = args.epochs
BATCHSIZE = args.batchsize
DEBUG=args.debug
VAL_SPLIT = args.val_split
TEST_SPLIT=args.test_split
DATA_CSV_FILENAME = "data/Webis-argument-framing.csv"

def train_and_save(modelname,pretrained_name,X,X_val,epochs=5,batch_size=6,learning_rate=1e-4):
    train_size = len(X['input_ids'])
    val_size = len(X_val['input_ids'])
    print("Load Model.")
    model = get_model(pretrained_name)
    model.compile(optimizer=Adam(learning_rate=learning_rate),loss=SparseCategoricalCrossentropy)
    print(f"Starting training on {train_size} samples. Validating on {val_size} samples.")
    history = model.fit(X,validation_data=X_val,epochs=epochs,batch_size=batch_size,validation_split=0.2)
    model.save_pretrained(modelname)
    print("finished training and saved model.")
    return model, history

def build_int_mapping(args):

    # set default value to something not present.
    d = defaultdict(lambda : 100000)
    for a in args:
        d[a.frame] = a.frame_id

    return d

def evaluate(model, args_test,history , batch_size=BATCHSIZE):
    rouge = Rouge()
    print(f"Evaluate on {len(args_test)} samples.")

    X_test, y_test = get_train_data(args_test, return_text_labels=True)

    tokenizer = get_tokenizer()
    predictions = []
    correct = 0
    input_ids = X_test['input_ids']
    attention_masks = X_test['attention_mask']
    input_ids_batch = []
    attention_mask_batch = []

    total = len(input_ids)
    no_predicted = 0
    for x_input_ids, x_attention_masks, y in tqdm(zip(input_ids, attention_masks, y_test), total=total,
                                                  desc='predicting test set'):

        # if batch is full or last batch (maybe < batchsize) is reached
        if total - no_predicted - len(input_ids_batch) == 1:
            input_ids_batch.append(x_input_ids)
            attention_mask_batch.append(x_attention_masks)

        if len(input_ids_batch) == batch_size or total - no_predicted - len(input_ids_batch) == 0:
            # predict_batch
            input_ids_batch = tf.stack(input_ids_batch)
            attention_mask_batch = tf.stack(attention_mask_batch)
            prediction = model.generate(input_ids=input_ids_batch, attention_mask=attention_mask_batch)

            for out_ids in prediction:
                tokens = tokenizer.convert_ids_to_tokens(out_ids, skip_special_tokens=True)
                tokens = [t for t in tokens if t not in ['<pad>', '</s>']]
                strs = tokenizer.convert_tokens_to_string(tokens)
                predictions.append(strs)
            no_predicted += batch_size
            input_ids_batch = [x_input_ids]
            attention_mask_batch = [x_attention_masks]

        else:
            # build batch
            input_ids_batch.append(x_input_ids)
            attention_mask_batch.append(x_attention_masks)

    # compute accuracy
    correct = 0
    for i, (pred, label) in enumerate(zip(predictions, y_test)):
        if pred == label:
            correct += 1
        if pred == "":
            # commented out for cleaner output. Should be
            #print("WARNING:Empty prediction!")
            predictions[i] = "EMPTY"

    # compute rouge1 f1 mean
    r1_f1 = rouge.get_scores(predictions, y_test, avg=True)['rouge-1']['f']

    # compute f1 classification score
    class_int_mapping = build_int_mapping(args_test)

    pred_int = [class_int_mapping[p] for p in predictions]
    label_int = [class_int_mapping[l] for l in y_test]


    f1 = f1_score(label_int,pred_int,average='weighted')
    print('rouge-1 f-score: ',r1_f1)
    print('f1:', f1)
    print('acc= ', correct / len(y_test))


if __name__ == "__main__":

    print(MODEL_FILENAME)
    if isdir(MODEL_FILENAME):
        yn = input(f"WARNING! Model \"{MODEL_FILENAME}\" already exists. Training again will overwrite old checkpoint."
                   f"\nYou can define a new name in default.py or call with option --model-savename \n\tContinue?[y/N]")
        if yn != 'y':
            print('Exiting, user abort.')
            exit(0)

    print("Preprocessing data for training.")
    X_train,X_val, args_test = get_data(DATA_CSV_FILENAME, train_min_class_count=5, val_split=VAL_SPLIT, test_split=TEST_SPLIT)


    if DEBUG:
        # use small model and data for testing.
        X_train['input_ids'] = X_train['input_ids'][0:100]
        X_train['attention_mask'] = X_train['attention_mask'][0:100]
        X_val['input_ids'] = X_val['input_ids'][0:20]
        X_val['attention_mask'] = X_val['attention_mask'][0:20]
        args_test=args_test[0:20]
        EPOCHS = 2
        PRETRAINED_NAME = 't5-small'

    print("Finished preprocessing")
    trained_model, history = train_and_save(MODEL_FILENAME,'t5-small', X_train, X_val, batch_size=BATCHSIZE, epochs=EPOCHS)

    with open(join(MODEL_FILENAME,'history.pkl'),'wb') as f:
        pickle.dump(history.history,f,-1)

    evaluate(trained_model, args_test, history, batch_size=BATCHSIZE)
