import defaults
from model import get_model, get_tokenizer
from preprocessing import get_train_data, get_arguments
from random import randint
import tensorflow as tf

import argparse

parser = argparse.ArgumentParser(description='test a T5 on random examples from the Webis-Framing-19 dataset.')
parser.add_argument('--pretrained-name', help="Name of the pretrained model to test. Huggingface models and local models can be used. For relative paths you need to use \"./../modelname\"" ,type=str,default=defaults.model_savename)
parser.add_argument('--data-file', help="absolute or relative path to Webis-Framing-19 dataset csv file.", type=str, default=defaults.data_csv_file)
parser.add_argument('--no-tries', help="Test this many random examples from the dataset.", type=int, default=defaults.no_tries)



args = parser.parse_args()

MODELNAME = args.pretrained_name
CSV_FILE = args.data_file
NO_TRIES = args.no_tries

seperator_line = "---------------------------------------------------------------------------------"

if __name__ == "__main__":
    print("Loading Model.")
    model = get_model(MODELNAME)
    tokenizer = get_tokenizer()

    print("Preprocessing data.")
    args = get_arguments(CSV_FILE)
    X , Y = get_train_data(args,return_text_labels=True)
    
    print("Starting test.")
    for k in range(NO_TRIES):
        ix = randint(0,len(args)-1)
        x_text = args[ix].x_text
        
        x = tf.expand_dims(X.input_ids[ix],0)
        y = Y[ix]
        
        out_ids = tf.squeeze(model.generate(x))
        tokens = tokenizer.convert_ids_to_tokens(out_ids,skip_special_tokens=True)
        tokens = [t for t in tokens if t not in ['<pad>','</s>']]
        pred = tokenizer.convert_tokens_to_string(tokens)
        print(seperator_line)
        print(f"input=\n{x_text}")
        print(f"prediction =\t{pred:>15}")
        print(f"label      =\t{y:>15}")
        print(seperator_line)

        

            

