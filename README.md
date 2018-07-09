# STC ML School 2018

Test task of audio classification into 8 classes for Speech Technology Center Machine Learning summer school 2018 ([STC ML School 2018](https://mlschool.speechpro.ru)).

## Solutions

1. In [`stc_ml_school_2018.ipynb`](./stc_ml_school_2018.ipynb) the basic approach is considered. Namely, the features like *mfcc*, *chroma stft*, *melspectrogram*, *spectral contrast* and *tonnetz* are extracted from raw `.wav` files as input features to the gradient boosting algorithm (in this case `catboost`).

2. In [`stc_ml_school_2018_nn.ipynb1`](./stc_ml_school_2018_nn.ipynb) the neural network is used on *melspectrogram* images.

### Prerequisites

The following libraries are used:

* [librosa](https://librosa.github.io/librosa/) - feature extraction from raw `.wav` files
* [catboost](https://github.com/catboost/catboost) - gradient boosting model
* [pytorch](https://pytorch.org) - neural network framework

## Acknowledgments

[Blog of Aaqib Saeed](http://aqibsaeed.github.io/2016-09-03-urban-sound-classification-part-1/)
