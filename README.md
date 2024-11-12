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
    ├── csv_files/
    ├── img/
    
```

### 5. Main file explanation

- `main.py`: This code preprocesses the training dataset by filtering redundant bounding boxes, trains YOLO and DETR models using configuration files, and trains a separate NEG model. It performs inference on test images using these models and Test Time Augmentation (TTA). The NEG predictions are saved in a CSV file and YOLO/DETR predictions in separate CSV files for each model. The results are stored in the data/predictions folder, with filenames indicating the model and split. The csv files are post processed, and saved in submissions/final_submission.csv.

## Contributors

This repository was created by [Team Epoch V](https://teamepoch.ai/team#v), based in the [Dream Hall](https://www.tudelft.nl/ddream) of the [Delft University of Technology](https://www.tudelft.nl/).

Read more about this competition [here](https://teamepoch.ai/competitions).

[![Github Badge](https://img.shields.io/badge/-Emiel_Witting-24292e?style=flat&logo=Github)](https://github.com/MarJarAI)
[![Github Badge](https://img.shields.io/badge/-Jeffrey_Lim-24292e?style=flat&logo=Github)](https://github.com/madhavv197)
[![Github Badge](https://img.shields.io/badge/-Hugo_de_Heer-24292e?style=flat&logo=Github)](https://github.com/FBB0)
[![Github Badge](https://img.shields.io/badge/-Jasper_van_Selm-24292e?style=flat&logo=Github)](https://github.com/Blagues)
