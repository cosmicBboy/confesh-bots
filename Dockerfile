FROM ubuntu:14.04
MAINTAINER Niels Bantilan <niels.bantilan@gmail.com>
RUN apt-get update && apt-get install -y python && apt-get install -y python-pip
RUN apt-get install -y git && apt-get install -y curl
RUN curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash

# Install global dependencies
RUN sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev

# Install pyenv python virtual environment management utility
RUN echo "" >> ~/.bashrc
RUN echo 'export PATH="/root/.pyenv/bin:$PATH"' >> ~/.bashrc
RUN echo 'eval "$(pyenv init -)"' >> ~/.bashrc
RUN echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
RUN . ~/.bashrc
RUn echo 'pyenv install 2.7.11'
