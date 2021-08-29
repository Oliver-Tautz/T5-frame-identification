# name of the model to train save and test
model_savename = "t5-base-frame-finetuned-debug"

# name of the model to start training from. You can use local names to continue training from previously
# saved checkpoints or choose 't5-base'/'t5-small' to start from pretrained t5.
pretrained_name = "t5-base"

# hyperparameters
epochs = 5
batchsize = 6
learning_rate=1e-4
train_min_class_count=5
test_split=0.2
val_split=0.2

# default value of tries for the test.py script
no_tries = 3

# default location of dataset
data_csv_file = "data/Webis-argument-framing.csv"
data_path = 'data'

# location of saved test set
testset_name = 'args_test.pkl'

# default location of trained models
models_path = 'trained_models'
