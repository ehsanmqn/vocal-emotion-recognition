## Speech Emotion Recognition
This repository contains our work on Speech emotion recognition using emodb dataset. This dataset is available here [Emo-db](http://www.emodb.bilderbar.info/download/)

### Experimental results
Five different models applied for emotion recognition purpose (SVM, Random Forest, NN, CNN and LSTM). Each model trained and evaluated using emo-DB's four classes (Sad, Happy, Neutral and Angry). Following are experimental results from each model (test_size=0.2, random_state=42):
1) SVM accuracy: 76%
2) RF accuracy: 64%
3) NN accuracy: 80%
4) CNN accuracy: 85%
5) LSTM accuracy: 86%

### Prerequisites
Linux (preferable Ubuntu LTS). Python2.x 

### Installing dependencies 
Dependencies are listed below and in the `requirements.txt` file.

* h5py
* Keras
* scipy
* sklearn
* speechpy
* tensorflow
* tqdm

Install one of python package managers in your distro. If you install pip, then you can install the dependencies by running 
`pip2 install -r requirements.txt` 

If you prefer to accelerate keras training on GPU's you can install `tensorflow-gpu` by 
`pip2 install tensorflow-gpu`

### Directory Structure
- `speechemotionrecognition` - Package folder which contains all the code files corresponding to package
- `dataset` - Contains the speech files in wav formatted seperated into 7 folders which are the corresponding labels of those files
- `models` - Contains the saved models which obtained best accuracy on test data.
- `example.py` - Contains examples on how to use the package

### Details of the package
- `utilities.py` - Contains code to read the files, extract the features and create test and train data
- `mlmodel.py` - Code to train non DL models. We have three models
	- `1 - SVM`
	- `2 - Random Forest`
	- `3 - Neural Network`
- `dnn.py` - Code to train Deep learning Models. Supports two models given below
    - `1 - CNN`
    - `2 - LSTM`

### Using the package
Look at `example.py` for sample usage.
