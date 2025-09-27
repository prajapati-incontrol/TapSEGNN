ARG PYTHON_VERSION=3.11.11
FROM python:${PYTHON_VERSION}-slim AS base 

# setup working directory 
WORKDIR /tapsegnn

# copy the requirements 
COPY requirements.txt .

# run and install dependencies 
RUN pip install --no-cache-dir -r requirements.txt 

# Copy the necessary files 
COPY . . 

# Create directories for results that stores generated markdown reports 
RUN mkdir -p results

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MATPLOTLIB_BACKEND=Agg

# optional command to run the whole training. 
CMD ["python","main.py"]
