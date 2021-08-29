from os.path import join
from os import listdir
import pickle
import matplotlib.pyplot as plt
import defaults
from train_evaluate_main import evaluate
from model import get_model
import argparse

# this import is needed to unpickle the args!
from preprocessing import Argument

parser = argparse.ArgumentParser(description='evaluate the default model on a precompiled testset.')
parser.add_argument('--batchsize', help="Use batchsize for evaluation. 6 works for t5-base on colab. t5-small can use 24.", type=int, default=defaults.batchsize)
parser.add_argument('--debug', help="Use small dataset for testing the code.", action='store_true')
parser.add_argument('--all', help="evaluate all models in trained_models directory.", action='store_true')

args = parser.parse_args()


debug = True
all = True

if all:
    models= listdir(defaults.models_path)
    #models = [join(defaults.models_path,m) for m in models]
else:
    models =  [join(defaults.models_path,defaults.model_savename)]


for model_savename in models:
    modelpath = join(defaults.models_path,model_savename)

    # plot loss and val loss
    f_history = open(join(modelpath,'history.pkl'),'rb')
    history = pickle.load(f_history)
    f_history.close()
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = list(range(1,len(loss)+1))

    fig = plt.figure()
    ax = fig.add_subplot(111)


    plt.plot(epochs,loss,label='loss')
    plt.plot(epochs,val_loss,label='val_loss')

    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    plt.xticks(epochs)
    plt.legend()
    plt.savefig(join(modelpath,'loss.pdf'),dpi=200)
    plt.close(fig)
    plt.clf()

    # evaluate on testargs and print out scores.
    f_args_test =  open(join(defaults.data_path,'args_test.pkl'),'rb')
    args_test = pickle.load(f_args_test)

    if debug:
        args_test = args_test[0:100]

    f_args_test.close()

    if debug:
        args_test = args_test

    print(f'Evaluate {modelpath}.')
    model = get_model(modelpath)
    evaluate(model,args_test,history,defaults.batchsize)



