ARG PYTHON_VERSION=3.11.11
FROM python:${PYTHON_VERSION}-slim AS base 

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    pkg-config \
    libcairo2-dev \
    libpango1.0-dev \
    && rm -rf /var/lib/apt/lists/*


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

EXPOSE 8080

# Create non-root user with minimum necessary privileges 
RUN useradd -m -u 1000 scientist && chown -R scientist:scientist /tapsegnn 
USER scientist 


# optional command to run the whole training. 
CMD ["python","src/main.py"]
