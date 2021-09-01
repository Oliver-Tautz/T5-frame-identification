# T5-frame-identification

Code for task of course 'deep learning for natural language processing', at
University Bielefeld, by Oliver Tautz.

The scripts should automatically use your GPU if CUDA is correctly installed and
works with tensorflow2.  Training and evaluation is extremely slow on CPU!
Testing is possible. All code was tested using python3.8.9

To use the code I suggest to use a virtuelenv (e.g. pip or conda) and install
the necessary packages. This can be done by calling `python setup.py` from the
root of this repository. The Webis-Argument-Framing-19 dataset and the trained
T5 will also be downloaded and extracted to the correct location. Should this
fail you can place the files manually:
* Download the [dataset](https://doi.org/10.5281/zenodo.3373355). The
  'Webis-argument-framing.csv' needs to be stored in a subfolder called 'data'.
* The python packages can be installed by calling `python -m pip install -r
  requirements.txt` from the root of this repository. 
* Download the trained model
  [here](https://drive.google.com/file/d/1U0x_6WgQWGLAT4rMzfVPfv82-myZogxp/view?usp=sharing)
and extract the archive in this folder.

In addition to the `setup.py` two executable scripts are provided. To train and
evaluate the t5 call `python train_evaluate_main.py`. To test if the code is
working supply the `--debug` option. **This is recommended when using CPU.** To
fetch some random examples from the dataset and run a pretrained model on them
call e.g. `python test.py --test-no 3`. The `--test-no` option can be used to
print multiple examples in a single run.

All scripts use reasonable defaults defined in `defaults.py`. If you want to
tweak them you can do so. It is also possible to call the scripts with options.
Call them with the `-h` option to print information on what options are
available.

The best pretrained model will be in `trained_models/t5-base-frame-finetuned`.
All newly trained models will be stored in the same location by default. If you
want to train another model be sure to supply a new name for it via
`--model-savename` option or set another model_savename in the `defaults.py`.
Setting the name in `defaults.py` makes it easy to call `test.py` afterwards.

The Layout of this folder after calling setup.py and extracting the model should
be:

```
.
+-- data
|   +-- Webis-argument-framing.csv
|   +-- args_test.pkl
+-- defaults.py
+-- download_model.sh
+-- evaluate.py
+-- model.py
+-- preprocessing.py
+-- README.md
+-- requirements.txt
+-- setup.py
+-- test.py
+-- trained_models
|   +-- t5-base-frame-finetuned
    |   +-- config.json
    |   +-- tf_model.h5
+-- train-evaluate-main.py
+-- utils.py
```


