FROM ubuntu:14.04
MAINTAINER Niels Bantilan <niels.bantilan@gmail.com>
RUN apt-get update && apt-get install -y python && apt-get install -y python-pip
RUN apt-get install -y git && apt-get install -y curl
RUN curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash

RUN echo "" >> ~/.bashrc
RUN echo 'export PATH="/root/.pyenv/bin:$PATH"' >> ~/.bashrc
RUN echo 'eval "$(pyenv init -)"' >> ~/.bashrc
RUN echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc