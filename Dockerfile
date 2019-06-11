FROM registry.fedoraproject.org/fedora-minimal:27
MAINTAINER Benjamin Chardi <benjamin.chardi@gmail.com>

RUN microdnf install -y \
python3-notebook \
python2-notebook \
python2-numpy \
python2-scipy \
python2-matplotlib \
python2-ipython \
python2-scikit-learn \
python2-pandas \
python2-pydotplus \
python2-pydot \
graphviz-python \
graphviz \
dnf \
&& dnf reinstall -y tzdata \ 
&& microdnf update \
&& microdnf clean all \
&& rm -rf /var/cache/yum \
&& rm -rf /var/cache/dnf \
&& groupadd -r -g 1000 astropy \
&& useradd -r -u 1000 -g 1000 -m -d /astropy astropy

WORKDIR /astropy
EXPOSE 8888

RUN mkdir -p /astropy/notebooks && chown -R astropy:astropy /astropy/notebooks 
