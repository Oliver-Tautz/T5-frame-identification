import subprocess
import sys
import os

def pip_install_requirements(requirements_filepath):
    subprocess.check_call([sys.executable,'-m','pip','install','-r', requirements_filepath])

def get_dataset(delete_zip=True):
    subprocess.run(['wget','https://zenodo.org/record/3598315/files/Webis-argument-framing.zip?download=1'])
    os.makedirs("data",exist_ok=True)
    finished = subprocess.run(['unzip', '-o' ,'-d', 'data', './Webis-argument-framing.zip?download=1'])
    if finished.returncode != 0:
        print("Unzipping failed. Is unzip installed?")
    subprocess.run(['rm','Webis-argument-framing.zip?download=1'])

def get_model(delete_zip = True):

    # download from gdrive link
    subprocess.call(['sh',"./download_model.sh"])
    # unzip model
    finished = subprocess.run(['unzip', '-o', './trained_model.zip'])
    #remove zip
    subprocess.run(['rm', 'trained_model.zip'] )

if __name__ == "__main__":

    yn = input("This script will install packages to use the frame-ident-T5 in the current python environment. This cannot easily be undone.\nI recommend installing to a virtual environment only. Are you sure you want to continue? \n[y/N]")
    if yn.lower() == 'y':

        print("Installing python packages to current environment.")
        pip_install_requirements('requirements.txt')
        print("Environment set up.\n","Download and extract dataset.")
        get_dataset()
        print("Done")
        get_model()
    else:
        print("Exiting. User abort")
        exit(0)
