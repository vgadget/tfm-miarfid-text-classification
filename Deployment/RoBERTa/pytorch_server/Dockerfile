# set base image (host OS)
FROM python:3.8-slim

# set the working directory in the container
WORKDIR /code

# copy the dependencies file to the working directory
COPY requirements.txt .

RUN pip install --upgrade pip

# install dependencies
RUN pip install -r requirements.txt

RUN apt-get update

COPY data ./data
COPY checkpoint.bin .
COPY main.py .


EXPOSE 8080