ARG PYTHON_VERSION=3.11.11
FROM python:${PYTHON_VERSION}-slim AS base 

# setup working directory 
WORKDIR /project 

# copy the requirements 
COPY requirements.txt .

# run and install dependencies 
RUN pip install --no-cache-dir -r requirements.txt 

# Copy the necessary files 