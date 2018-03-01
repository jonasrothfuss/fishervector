import os
import unittest

import numpy as np
from fishervector import FisherVectorGMM


class TestFisherVectors(unittest.TestCase):

  def test_with_gaussian_samples_video(self):
    np.random.seed(22)
    shape = [200, 15, 10, 30]
    test_data = np.concatenate([np.random.normal(np.zeros(30), size=shape), np.random.normal(np.ones(30), size=shape)], axis=0)
    n_kernels = 2
    fv_gmm = FisherVectorGMM(n_kernels=n_kernels).fit(test_data)
    n_test_videos = 20
    fv = fv_gmm.predict(test_data[:n_test_videos])
    self.assertEquals(fv.shape, (n_test_videos, 15, 2 * n_kernels, 30))

  def test_with_gaussian_samples_image(self):
    np.random.seed(22)
    shape = [200, 10, 30]
    test_data = np.concatenate([np.random.normal(np.zeros(30), size=shape), np.random.normal(np.ones(30), size=shape)], axis=0)
    n_kernels = 2
    fv_gmm = FisherVectorGMM(n_kernels=n_kernels).fit(test_data)
    n_test_videos = 20
    fv = fv_gmm.predict(test_data[:n_test_videos])
    self.assertEquals(fv.shape, (n_test_videos, 2 * n_kernels, 30))

  def test_with_gaussian_samples_video_bic(self):
    np.random.seed(23)
    shape = [200, 15, 10, 30]
    test_data = np.concatenate([np.random.normal(np.zeros(30), size=shape), np.random.normal(np.ones(30), size=shape)], axis=0)
    n_kernels = 2
    fv_gmm = FisherVectorGMM().fit_by_bic(test_data, choices_n_kernels=[2,7])
    n_test_videos = 20
    fv = fv_gmm.predict(test_data[:n_test_videos])
    self.assertEquals(fv.shape, (n_test_videos, 15, 2 * n_kernels, 30))
    self.assertEquals(fv_gmm.n_kernels, 2)

  def test_with_gaussian_samples_image_bic(self):
    np.random.seed(23)
    shape = [200, 10, 30]
    test_data = np.concatenate([np.random.normal(-np.ones(30), size=shape), np.random.normal(np.ones(30), size=shape)], axis=0)
    n_kernels = 2
    fv_gmm = FisherVectorGMM().fit_by_bic(test_data, choices_n_kernels=[2,7])
    n_test_videos = 20
    fv = fv_gmm.predict(test_data[:n_test_videos])
    self.assertEquals(fv.shape, (n_test_videos, 2 * n_kernels, 30))
    self.assertEquals(fv_gmm.n_kernels, 2)

  def test_dump_and_load(self):
    np.random.seed(22)
    shape = [200, 10, 30]
    test_data = np.concatenate([np.random.normal(np.zeros(30), size=shape), np.random.normal(np.ones(30), size=shape)],
                               axis=0)
    n_kernels = 2
    pickle_path = ".test_fv_gmm.pickle"
    fv_gmm = FisherVectorGMM(n_kernels=n_kernels).fit(test_data, model_dump_path=pickle_path)

    fv_gmm_loaded = FisherVectorGMM.load_from_pickle(pickle_path)
    os.remove(pickle_path)
    covars_match = np.all(fv_gmm.covars == fv_gmm_loaded.covars)
    means_match = np.all(fv_gmm.means == fv_gmm_loaded.means)
    self.assertTrue(covars_match and means_match)

if __name__ == '__main__':
  unittest.main()