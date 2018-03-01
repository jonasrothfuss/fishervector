[![Build Status](https://travis-ci.org/jonasrothfuss/py-fisher-vector.svg?branch=master)](https://travis-ci.org/jonasrothfuss/py-fisher-vector)

# Description
The package implements Improved Fisher Vectors as described in [1]. For a more concise description of Fisher Vectors see [2].
The functionality includes:
- Fitting a Gaussian Mixture Model (GMM)
- Determining the number of GMM components via BIC
- Saving and loading the fitted GMM
- Computing the (Improved) Fisher Vectors based on the fitted GMM

# Installation
For intsallation via pip run the following command on your terminal (requires python 3.4 or higher):
```
$ pip install fishervector
```
afterwards you should be able to import the package in python:
```
from fishervector import FisherVectorGMM
```
# First Steps
##### 1. Simulate some data / get your data ready
 We randomly sample data just for this tutorial -> use your own data e.g. SIFT features of images
```
import numpy as np
shape = [300, 20, 32] # e.g. SIFT image features
image_data = np.concatenate([np.random.normal(-np.ones(30), size=shape), np.random.normal(np.ones(30), size=shape)], axis=0)
```
##### 2. Train the GMM
```
from fishervector import FisherVectorGMM
fv_gmm = FisherVectorGMM(n_kernels=10).fit(image_data)
```
Or alternatively fit the GMM using the BIC to determine the number of GMM components automatically:
```
from fishervector import FisherVectorGMM
fv_gmm = FisherVectorGMM().fit_by_bic(test_data, choices_n_kernels=[2,5,10,20])
```
##### 3. Computing improved fisher vectors
```
image_data_test = image_data[:20] # use a fraction of the data to compute the fisher vectors
fv = fv_gmm.predict(image_data_test)
```

Contributors:
* Jonas Rothfuss (https://github.com/jonasrothfuss/)
* Fabio Ferreira (https://github.com/ferreirafabio/)

References:
- [1] https://www.robots.ox.ac.uk/~vgg/rg/papers/peronnin_etal_ECCV10.pdf
- [2] http://www.vlfeat.org/api/fisher-fundamentals.html