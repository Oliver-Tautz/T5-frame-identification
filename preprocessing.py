import csv
from transformers import AutoTokenizer
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split

class Argument():
    "Simple class storing all fields of a sample of the Webis-Framing-19 dataset."

    def __init__(self, argument_id, frame_id, frame, topic_id, topic, premise, stance, conclusion):
        self.conclusion = conclusion
        self.stance = stance
        self.premise = premise
        self.topic = topic
        self.topic_id = int(topic_id)
        self.frame = frame
        self.frame_id = int(frame_id)
        self.argument_id = int(argument_id)

        # combine premise and conclusion for full text
        self.x_text = self.premise + self.conclusion


def get_arguments(csv_filename):
    """
    Parse arguments from Webis-Framing-19 dataset to python representation.

    Args:
        csv_filename: the absolute or relative path to the csv-file containing the Webis-Argument-19 dataset.

    Returns:
        list of class 'Argument'. 

    """

    try:
        file = open(csv_filename)
    except:
        print("Cannot open file. File not found?")

    dreader = csv.DictReader(file, dialect='unix', delimiter=',', quotechar='\"')

    args = []
    topic_frame_mapping = defaultdict(lambda: set())

    for row in dreader:
        arg = Argument(row['argument_id'], row['frame_id'], row['frame'], row['topic_id'], row['topic'], row['premise'],
                       row['stance'], row['conclusion'])
        topic_frame_mapping['']
        args.append(arg)

    return args


def make_train_dict(input_ids, attention_masks, labels):
    """
    Put things in a dictionary. Just a small utility function.
    """
    return {'input_ids': input_ids, 'attention_mask': attention_masks, 'labels': labels}


def get_classes_count(args):
    classes = []
    counter = Counter()

    for arg in args:
        classes.append(arg.frame_id)
        counter[arg.frame_id] += 1

    return counter


def clean_dataset(args, classes_count, min_class_count=5):
    classes_count=get_classes_count(args)
    classes_to_remove = []
    for c in classes_count.keys():
        if classes_count[c] < min_class_count:
            classes_to_remove.append(c)

    clean_args = [a for a in args if a.frame_id not in classes_to_remove]
    removed_args = [a for a in args if a.frame_id in classes_to_remove]
    print(f"removed {len(removed_args)} arguments from dataset. {len(clean_args)} remaining.")
    return clean_args


def get_train_data(args, return_text_labels=False, tokenizer=AutoTokenizer.from_pretrained("t5-small")
                   ):
    """
    Parse list of Arguments to dict ready to be passed to T5.fit().

    Args:
        args: list of arguments to convert
        tokenizer: the tokenizer to be used. If you want to use a different model (e.g. t5-base) you need to change this!
        teturn_text_labels: if set to true return X to be fed to t5.generate(). Y will be the list of frames as str

    Returns:
        dict with 'input_ids', 'attention_mask' and 'labels' fields, ready to be used in training. 
        if 'return_text_labels' is true, return list of labels as second result.
    """

    X_text = []
    Y_text = []

    # use padding and truncation. Padding is no problem, truncation could lose Information.
    # Maybe filter long examples out beforehand?

    for arg in args:
        X_text.append(arg.premise + arg.conclusion)
        Y_text.append(arg.frame)

    X = tokenizer(X_text, return_tensors='tf', truncation=True, padding=True)
    Y = tokenizer(Y_text, return_tensors='tf', truncation=True, padding=True)

    if return_text_labels:
        return X, Y_text
    else:
        return make_train_dict(X['input_ids'], X['attention_mask'], Y['input_ids'])


def get_classes_count(args):
    """
    Return dict with {frame_id : #frame_id}.

    Args:
        args: list of arguments to count
    Returns:
        dict with {frame_id : #frame_id}
    """

    classes = []
    counter = Counter()

    for arg in args:
        classes.append(arg.frame_id)
        counter[arg.frame_id] += 1

    return counter


def clean_dataset(args, min_class_count=5):
    """
    Remove args of classes with too little occurrence from args.

    Args:
        args: list of arguments to clean
        classes_count: dict with {frame_id : #frame_id} supplied by get_classes_count
        min_class_count: minimal occurance to keep.
    """
    classes_count = get_classes_count(args)
    print(f"cleaning dataset of {len(args)} arguments")
    classes_to_remove = []
    for c in classes_count.keys():
        if classes_count[c] < min_class_count:
            classes_to_remove.append(c)

    clean_args = [a for a in args if a.frame_id not in classes_to_remove]
    removed_args = [a for a in args if a.frame_id in classes_to_remove]
    print(f"removed {len(removed_args)} arguments from dataset. {len(clean_args)} remaining.")
    return clean_args, removed_args


def args_train_test_val_split(args, random_state=42, train_min_class_count=5):
    """
    Split dataset into train, test and val set in a stratified fashion. ...
    TODO!

    """

    # split into single-sample classes and others.
    args_clean, args_singles = clean_dataset(args, min_class_count=2)
    classes_clean = [a.frame_id for a in args_clean]

    # split other in train,test and val set in a stratified fashion.
    args_train, args_test = train_test_split(args_clean, test_size=0.2, shuffle=True, stratify=classes_clean,
                                             random_state=random_state)
    classes_train = [a.frame_id for a in args_train]
    args_train, args_val = train_test_split(args_train, test_size=0.2, shuffle=True, stratify=classes_train,
                                            random_state=random_state + 70)

    # randomly split the 'singles' into partitions train,test val. disregard them for train set.
    args_train_singles, args_test_singles = train_test_split(args_singles, test_size=0.2, shuffle=True,
                                                             random_state=random_state + 42)
    _, args_val_singles = train_test_split(args_train_singles, test_size=0.2, shuffle=True,
                                           random_state=random_state + 20)

    args_train, _ = clean_dataset(args_train, min_class_count=train_min_class_count)

    # add singles back to test and val set.
    args_test.extend(args_test_singles)
    args_val.extend(args_val_singles)

    return args_train, args_test, args_val

def get_data(csv_filename,train_min_class_count,test_split=0.2,val_split=0.2):

    """
    Get data for training and testing. Three values are returned, X_train, X_val and args_test.
    X_train and X_val are usable by the model.fit() method of the identFrameT5 model. args_test can be used for evaluation.

    Args:
        csv_filename: absolute or relative path to the Webis-Framing-19 dataset
        train_min_class_count: To small classes are removed from the train set. What is the smallest number of samples
        acceptable for a class?
        test_split: portion of the data to be used for test set.
        val_split: portion of the train data to be used for validation.

    """
    args = get_arguments(csv_filename)
    args_train, args_test, args_val = args_train_test_val_split(args,train_min_class_count=train_min_class_count)

    X_train = get_train_data(args_train)
    X_val = get_train_data(args_val)

    return X_train, X_val, args_test

