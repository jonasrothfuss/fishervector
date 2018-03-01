from setuptools import setup

setup(name="fishervector",
      version='0.1',
      description='Fisher Vectors based on Gaussian Mixture Model',
      url='https://github.com/jonasrothfuss/py-fisher-vector',
      author='Jonas Rothfuss, Fabio Ferreira',
      author_email='jonas.rothfuss@gmail.com',
      license='MIT',
      packages=['fishervector'],
      test_suite='nose.collector',
      tests_require=['nose'],
      install_requires=[
        'numpy',
        'scikit_learn'
      ],
      zip_safe=False)