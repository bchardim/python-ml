docker run -p 8888:8888 -v $(pwd)/notebooks:/astropy/notebooks -w /astropy/notebooks centos7-python-ml:latest jupyter-notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root
