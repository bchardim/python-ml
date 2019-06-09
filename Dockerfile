FROM registry.centos.org/centos:centos7
MAINTAINER Benjamin Chardi <benjamin.chardi@gmail.com>

RUN yum -y update \
&& yum -y install epel-release \
&& yum -y install gcc python-pip python-devel openssl-devel libselinux-python python3-notebook \
&& yum clean all \
&& rm -rf /var/cache/yum

RUN pip install --upgrade pip \
&& pip install numpy scipy matplotlib ipython scikit-learn pandas jupyter

RUN groupadd -r -g 1001 astropy \
&& useradd -r -u 1001 -g 1001 -m -d /astropy astropy

USER astropy
WORKDIR /astropy
RUN mkdir -p /astropy/notebooks
EXPOSE 8888
