# set base image (host OS)
FROM --platform=linux/amd64 python:3.8-slim

# set the working directory in the container
WORKDIR /code

# Copy compiled tensorflow wheel
COPY tensorflow-2.7.0-cp38-cp38-linux_x86_64.whl .

# copy the dependencies file to the working directory
COPY requirements.txt .

RUN pip install --upgrade pip

# install dependencies
RUN pip install -r requirements.txt

# Install specific compiled tensorflow wheel
RUN python -m pip install --force-reinstall tensorflow-2.7.0-cp38-cp38-linux_x86_64.whl

RUN apt-get update

COPY data ./data
COPY checkpoint.bin .
COPY main.py .


EXPOSE 8080