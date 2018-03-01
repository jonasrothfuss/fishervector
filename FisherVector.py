import numpy as np
from sklearn.mixture import GaussianMixture
import pickle, os

N_Kernel_Choices = [5, 20, 60, 100, 200, 500]

class FisherVectorGMM:
  def __init__(self, n_kernels=1, covariance_type='diag'):
    assert covariance_type in ['diag', 'full']
    assert n_kernels > 0

    self.n_kernels = n_kernels
    self.covariance_type = covariance_type
    self.fitted = False

  def score(self, X):
    return self.gmm.bic(X.reshape(-1, X.shape[-1]))

  def fit(self, X, model_dump_path=None, verbose=True):
    """
    :param X: either a ndarray with 4 dimensions (n_videos, n_frames, n_descriptors_per_image, n_dim_descriptor)
              or with 3 dimensions (n_images, n_descriptors_per_image, n_dim_descriptor)
    :param model_dump_path: (optional) path where the fitted model shall be dumped
    :param verbose - boolean that controls the verbosity
    :return: fitted Fisher vector object
    """

    if X.ndim == 4:
      self.ndim = 4
      return self._fit(X, model_dump_path=model_dump_path, verbose=verbose)

    elif X.ndim == 3:
      self.ndim = 3
      X = np.reshape(X, [1] + list(X.shape))
      return self._fit(X, model_dump_path=model_dump_path, verbose=verbose)

    else:
      raise AssertionError("X must be an ndarray with 3 or 4 dimensions")

  def fit_by_bic(self, X, choices_n_kernels=N_Kernel_Choices, model_dump_path=None, verbose=True):
    """
    Fits the GMM with various n_kernels and selects the model with the lowest BIC
    :param X: either a ndarray with 4 dimensions (n_videos, n_frames, n_descriptors_per_image, n_dim_descriptor)
              or with 3 dimensions (n_images, n_descriptors_per_image, n_dim_descriptor)
    :param choices_n_kernels: array of positive integers that specify with how many kernels the GMM shall be trained
                              default: [20, 60, 100, 200, 500]
    :param model_dump_path: (optional) path where the fitted model shall be dumped
    :param verbose - boolean that controls the verbosity
    :return: fitted Fisher vector object
    """

    if X.ndim == 4:
      self.ndim = 4
      return self._fit_by_bic(X, choices_n_kernels=choices_n_kernels, model_dump_path=model_dump_path, verbose=verbose)

    elif X.ndim == 3:
      self.ndim = 3
      X = np.reshape(X, [1] + list(X.shape))
      return self._fit_by_bic(X, choices_n_kernels=choices_n_kernels, model_dump_path=model_dump_path, verbose=verbose)

    else:
      raise AssertionError("X must be an ndarray with 3 or 4 dimensions")

  def predict(self, X, normalized=True):
    """
    Computes Fisher Vectors of provided X
    :param X: either a ndarray with 4 dimensions (n_videos, n_frames, n_descriptors_per_image, n_dim_descriptor)
              or with 3 dimensions (n_images, n_descriptors_per_image, n_dim_descriptor)
    :param normalized: boolean that indicated whether the fisher vectors shall be normalized --> improved fisher vector
              (https://www.robots.ox.ac.uk/~vgg/rg/papers/peronnin_etal_ECCV10.pdf)
    :returns fv: fisher vectors
                  if X.ndim is 4 then returns ndarray of shape (n_videos, n_frames, 2*n_kernels, n_feature_dim)
                  if X.ndim is 3 then returns ndarray of shape (n_images, 2*n_kernels, n_feature_dim)
    """
    if X.ndim == 4:
      return self._predict(X, normalized=normalized)

    elif X.ndim == 3:
      orig_shape = X.shape
      X = np.reshape(X, [1] + list(X.shape))
      result = self._predict(X, normalized=normalized)
      return np.reshape(result, (orig_shape[0], 2 * self.n_kernels, orig_shape[-1]))

    else:
      raise AssertionError("X must be an ndarray with 3 or 4 dimensions")

  def _fit(self, X, model_dump_path=None, verbose=True):
    """
    :param X: shape (n_videos, n_frames, n_descriptors_per_image, n_dim_descriptor)
    :param model_dump_path: (optional) path where the fitted model shall be dumped
    :param verbose - boolean that controls the verbosity
    :return: fitted Fisher vector object
    """
    assert X.ndim == 4
    self.feature_dim = X.shape[-1]

    X = X.reshape(-1, X.shape[-1])

    # fit GMM and store params of fitted model
    self.gmm = gmm = GaussianMixture(n_components=self.n_kernels, covariance_type=self.covariance_type, max_iter=1000).fit(X)
    self.covars = gmm.covariances_
    self.means = gmm.means_
    self.weights = gmm.weights_

    # if cov_type is diagonal - make sure that covars holds a diagonal matrix
    if self.covariance_type == 'diag':
      cov_matrices = np.empty(shape=(self.n_kernels, self.covars.shape[1], self.covars.shape[1]))
      for i in range(self.n_kernels):
        cov_matrices[i, :, :] = np.diag(self.covars[i, :])
      self.covars = cov_matrices

    assert self.covars.ndim == 3
    self.fitted = True
    if verbose:
      print('fitted GMM with %i kernels'%self.n_kernels)

    if model_dump_path:
      with open(model_dump_path, 'wb') as f:
        pickle.dump(self,f, protocol=4)
      if verbose:
        print('Dumped fitted model to', model_dump_path)

    return self

  def _fit_by_bic(self, X, choices_n_kernels=N_Kernel_Choices, model_dump_path=None, verbose=True):
    """
    Fits the GMM with various n_kernels and selects the model with the lowest BIC
    :param X: shape (n_videos, n_frames, n_descriptors_per_image, n_dim_descriptor)
    :param choices_n_kernels: array of positive integers that specify with how many kernels the GMM shall be trained
                              default: [20, 60, 100, 200, 500]
    :param model_dump_path: (optional) path where the fitted model shall be dumped
    :param verbose - boolean that controls the verbosity
    :return: fitted Fisher vector object
    """

    bic_scores = []
    for n_kernels in choices_n_kernels:
      self.n_kernels = n_kernels
      bic_score = self.fit(X, verbose=False).score(X)
      bic_scores.append(bic_score)

      if verbose:
        print('fitted GMM with %i kernels - BIC = %.4f'%(n_kernels, bic_score))

    best_n_kernels = choices_n_kernels[np.argmin(bic_scores)]

    self.n_kernels = best_n_kernels
    if verbose:
      print('Selected GMM with %i kernels' % best_n_kernels)

    return self.fit(X, model_dump_path=model_dump_path, verbose=True)

  def _predict(self, X, normalized=True):
    """
    Computes Fisher Vectors of provided X
    :param X: features - ndarray of shape (n_videos, n_frames, n_features, n_feature_dim)
    :param normalized: boolean that indicated whether the fisher vectors shall be normalized --> improved fisher vector
    :returns fv: fisher vectors - ndarray of shape (n_videos, n_frames, 2*n_kernels, n_feature_dim)
    """
    assert self.fitted, "Model (GMM) must be fitted"
    assert self.feature_dim == X.shape[-1], "Features must have same dimensionality as fitted GMM"
    assert X.ndim == 4

    n_videos, n_frames = X.shape[0], X.shape[1]

    X = X.reshape((-1, X.shape[-2], X.shape[-1])) #(n_images, n_features, n_feature_dim)
    X_matrix = X.reshape(-1, X.shape[-1])

    # set equal weights to predict likelihood ratio
    self.gmm.weights_ = np.ones(self.n_kernels) / self.n_kernels
    likelihood_ratio = self.gmm.predict_proba(X_matrix).reshape(X.shape[0], X.shape[1], self.n_kernels)

    var = np.diagonal(self.covars, axis1=1, axis2=2)

    norm_dev_from_modes = ((X[:,:, None, :] - self.means[None, None, :, :])/ var[None, None, :, :]) # (n_images, n_features, n_kernels, n_featur_dim)

    # mean deviation
    mean_dev = np.multiply(likelihood_ratio[:,:,:, None], norm_dev_from_modes).mean(axis=1) #(n_images, n_kernels, n_feature_dim)
    mean_dev = np.multiply(1 / np.sqrt(self.weights[None, :,  None]), mean_dev) #(n_images, n_kernels, n_feature_dim)

    # covariance deviation
    cov_dev = np.multiply(likelihood_ratio[:,:,:, None], norm_dev_from_modes**2 - 1).mean(axis=1)
    cov_dev = np.multiply(1 / np.sqrt(2 * self.weights[None, :,  None]), cov_dev)

    fisher_vectors = np.concatenate([mean_dev, cov_dev], axis=1)

    # final reshape - separate frames and videos
    assert fisher_vectors.ndim == 3
    fisher_vectors = fisher_vectors.reshape((n_videos, n_frames, fisher_vectors.shape[1], fisher_vectors.shape[2]))


    if normalized:
      fisher_vectors = np.sqrt(np.abs(fisher_vectors)) * np.sign(fisher_vectors) # power normalization
      fisher_vectors = fisher_vectors / np.linalg.norm(fisher_vectors, axis=(2,3))[:,:,None,None]

    fisher_vectors[fisher_vectors < 10**-4] = 0

    assert fisher_vectors.ndim == 4
    return fisher_vectors

  @staticmethod
  def load_from_pickle(pickle_path):
    """
    loads a previously dumped FisherVectorGMM instance
    :param pickle_path: path to the pickle file
    :return: loaded FisherVectorGMM object
    """
    assert os.path.isfile(pickle_path), 'pickle path must be an existing file'
    with open(pickle_path, 'rb') as f:
      fv_gmm = pickle.load(f)
      assert isinstance(fv_gmm, FisherVectorGMM), 'pickled object must be an instance of FisherVectorGMM'
    return fv_gmm