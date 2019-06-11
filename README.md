## Project Title

Machine Learning with Python

## Getting Started

In this repo you can find simple examples of running ML with Python in a Red Hat Fedora container.

### Supervised
ML Supervised samples

### Unsupervised
ML Unsupervised samples


### DDA-Machine_Learning
Base ML scripts used in my personal contribution to Data Driven Astrophysics with ML. 


## Installing and running it with Red Hat Fedora container.

```
$ git clone https://github.com/bchardim/python-ml.git 
$ cd python-ml
$ sudo docker build -t fedora-python-ml .
$ bash docker-run-notebook.sh
[I 11:05:23.661 NotebookApp] Writing notebook server cookie secret to /astropy/.local/share/jupyter/runtime/notebook_cookie_secret
[I 11:05:23.760 NotebookApp] Serving notebooks from local directory: /astropy
[I 11:05:23.761 NotebookApp] 0 active kernels
[I 11:05:23.761 NotebookApp] The Jupyter Notebook is running at:
[I 11:05:23.761 NotebookApp] http://0.0.0.0:8888/?token=5464317c3d8db60fe68695d47da10cc32a974b6c9badae05
[I 11:05:23.761 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 11:05:23.761 NotebookApp] 
    
    Copy/paste this URL into your browser when you connect for the first time,
    to login with a token:
        http://0.0.0.0:8888/?token=5464317c3d8db60fe68695d47da10cc32a974b6c9badae05



$ firefox http://0.0.0.0:8888/?token=5464317c3d8db60fe68695d47da10cc32a974b6c9badae05

```

## Source Info

### Books

* Introduction to Machine Learning with Python
  A Guide for Data Scientists, By Andreas C. Müller and Sarah Guido, O'Reilly Media
* https://github.com/amueller/introduction_to_ml_with_python

### Free Online courses

* https://www.coursera.org/learn/machine-learning-with-python
* https://scikit-learn.org
* https://www.coursehero.com/file/28377088/01-adspy-module1-basicspdf/
* https://www.coursera.org/learn/data-driven-astronomy
* http://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/tutorial/astronomy/
