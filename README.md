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
- Estimated training time: 7 hours for the DETR,.

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

The data of the competition can be downloaded here: [Kelp Wanted: Segmenting Kelp Forests](https://www.drivendata.org/competitions/255/kelp-forest-segmentation/data/)
Unzip all the files into the `data/raw` directory.
The structure should look like this:

```
data/
    ├── processed/ # Cache directory for processed data. 
    ├── raw/ # @TODO Competition data must be PLACED HERE
        ├── submission_format/ # @TODO Submission format from the competition
        ├── test_satellite/ # @TODO Test satellite images from the competition
        ├── train_kelp/ # @TODO Train kelp images from the competition
        ├── train_satellite/ # @TODO Train satellite images from the competition
        ├── metadata_fTq0l2T.csv # @TODO Metadata from the competition
    ├── test/ # Test caching
    ├── training/ # Training cache
```
