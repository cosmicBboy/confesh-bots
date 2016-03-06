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
RUN curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash
RUN echo "" >> ~/.bashrc
RUN echo 'export PATH="/root/.pyenv/bin:$PATH"' >> ~/.bashrc
RUN echo 'eval "$(pyenv init -)"' >> ~/.bashrc
RUN echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
RUN /bin/bash -c "source ~/.bashrc"

# Install python dependencies using pyenv
RUN /bin/bash -c "pyenv install 2.7.11 \
    && pyenv global 2.7.11 \
    && pip install --upgrade pip \
    && pip install csvkit pandas numpy scipy sklearn nltk \
        textmining wordcloud beautifulsoup4 pymongo inflect bson \
    && export NLTK_DATA=/nltk_data \
    && mkdir $NLTK_DATA \
    && python -m nltk.downloader punkt stopwords"
