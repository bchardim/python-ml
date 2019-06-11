docker run -u astropy -p 8888:8888 -v $(pwd)/notebooks:/astropy/notebooks:rw -w /astropy fedora-python-ml:latest jupyter-notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root
