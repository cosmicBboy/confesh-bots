FROM ubuntu:14.04
MAINTAINER Niels Bantilan <niels.bantilan@gmail.com>
RUN apt-get update && apt-get install -y python && apt-get install -y python-pip
RUN apt-get install -y git && apt-get install -y curl

# Install global dependencies
RUN sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
    libncursesw5-dev python-dev libblas-dev liblapack-dev libatlas-base-dev \
    gfortran emacs

# Install pyenv python virtual environment management utility
ENV PYENV_ROOT /root/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash

# Install python dependencies using pyenv
RUN pyenv install 2.7.11
RUN pyenv global 2.7.11
RUN pyenv rehash
RUN pip install --upgrade pip
RUN pip install csvkit nltk
ENV NLTK_DATA /nltk_data
RUN mkdir $NLTK_DATA
RUN python -m nltk.downloader punkt stopwords
