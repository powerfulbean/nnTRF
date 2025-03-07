import setuptools

setuptools.setup(
  name="nntrf",
  version="1.0.2",
  author="Jin Dou",
  author_email="jindou.bci@gmail.com",
  long_description_content_type="text/markdown",
  url="https://github.com/powerfulbean/nnTRF",
  packages=setuptools.find_packages(),
  classifiers=[
  "Programming Language :: Python :: 3",
  "License :: OSI Approved :: MIT License",
  "Operating System :: OS Independent",
  ],
  python_requires='>=3.8',
  install_requires=[
      "numpy>=1.20.1",
      "torch>=1.12.1,<2.0.0",
      "scikit-fda==0.7.1",
      "mtrf",
      "scipy",
      "mkl-devel"
  ],
)
