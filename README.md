# Lacuna Malaria Detection Challenge

This is Team Epoch's solution to the [Lacuna Malaria Detection Challenge](https://zindi.africa/competitions/lacuna-malaria-detection-challenge), hosted by Zindi. 

A technical report will be written and uploaded after the competition finale.

## Getting started

This section contains the steps that need to be taken to get started with our project and fully reproduce our best
submission on the public and private leaderboard. The project was developed on Linux 6.8.0-48-generic, and on Python 3.10.14 on Pip version 22.0.2.

### 0. Prerequisites
Models were trained on machines with the following specifications:
- CPU: AMD Ryzen Threadripper Pro 3945WX 12-Core Processor / AMD Ryzen 9 7950X 16-Core Processor
- GPU: NVIDIA RTX A5000 / NVIDIA RTX Quadro 6000 / NVIDIA RTX A6000
- RAM: 96GB / 128GB
- OS: Linux 6.8.0-48-generic
- Python: 3.10.14
- Estimated training time: 7 hours for the DETR, 2 hours for YOLO.

For running inference, a machine with at least 32GB of RAM is recommended. We have not tried running the inference on a
machine with less RAM using all the test data that was provided by DrivenData.

### 1. Clone the repository

Make sure to clone the repository with your favourite git client or using the following command:

```
https://github.com/TeamEpochGithub/ZindiLacunaMalaria.git
```

### 2. Install Python 3.10.13

You can install the required python version here: [Python 3.10.13](https://github.com/adang1345/PythonWindows/tree/master/3.10.14)

### 3. Install the required packages

Install the required packages (on a virtual environment is recommended) using the following command:

```
pip install -r requirements.txt
```

### 4. Setup the competition data

The data of the competition can be downloaded here: [Lacuna Malaria Detection Challenge](https://zindi.africa/competitions/lacuna-malaria-detection-challenge/data)

Unzip all csv files into the `data/csv_files` directory, and all images into the `data/img` directory.

The structure should look like this:

```
data/
    ├── csv_files/ #folder 
    ├── img/
    
```

### 5. Main files explanation

- `train.py`: This file is used to train a model. `train.py` reads a configuration file from `conf/train.yaml`. This configuration file 
contains the model configuration to train with additional training parameters such as test_size and a scorer to use. 
The model selected in the `conf/train.yaml` can be found in the `conf/model` folder where a whole model configuration is stored (from preprocessing to postprocessing).
When training is finished, the model is saved in the `tm` directory with a hash that depends on the specific pre-processing, pretraining steps + the model configurations.

    - Command line arguments
    - CUDA_VISIBLE_DEVICES: The GPU to use for training. If not specified it uses DataParallel to train on multiple GPUs.  If you have multiple GPUs, you can specify which one to use.
- `submit.py`: This file does inference on the test data from the competition given trained model or an ensemble of trained models. 
It reads a configuration file from `conf/submit.yaml` which contains the model/ensemble configuration to use for inference.
Model configs can be found in the `conf/model` folder and ensemble configs in the `conf/ensemble` folder. The `conf/ensemble`
folder specifies the models (`conf/model`) to use for the ensemble and the weights to use for each model. The `submit.py` 

### 6. Place the fitted models
(For DrivenData) Any additional supplied trained models /scalers (.pt / .gbdt / .scaler) should be placed in the `tm` directory. 
When these models were trained, they are saved with a hash that depends on the specific pre-processing, pretraining steps + the model configurations.
In this way, we ensure that we load the correct saved model automatically when running `submit.py`.

### 7. Run submit.py

For reproducing our best submission, run `submit.py`. This will load the already configured `submit.yaml` file and
run the inference on the test data from the competition. `submit.yaml` in configured to what whe think is our best and our
most robust solution:
![img.png](images/best_submission.png)
`conf/ensemble/21-02-1st.yaml` contains a mixed ensemble of models trained on different feature sets, including UNets with VGG encoders, SwinTransformers and a ConvNext model.


If you get an error of that the path was not found of a model. Please ensure that you have the correct trained model in the `tm` directory.
If you don't have the trained models, you can train them 1 by 1 using `train.py` and the `conf/train.yaml` file.

`submit.py` will output a `submission.zip` in the root of the project directory. This file can be uploaded to the competition page. 
