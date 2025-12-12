ARG PYTHON_VERSION=3.11.11
FROM python:${PYTHON_VERSION}-slim AS base 

# Update package lists and install system-level build tools & libraries 
# build-essential: meta-package providing gcc, g++, make and other essential build tools 
# gcc: gnu c compiler 
# pkg-config: tool to detect installed libraries and their paths during builds 
# libcairo2-dev: development headers for Cairo 2d graphics library 
# $$ rm -rf: clean up apt cache to reduce image size 
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
# PYTHONUNBUFFERED: Forces Python to output logs directly (no buffering) -- useful for real-time logs in Docker
ENV PYTHONUNBUFFERED=1 
# MATPLOTLIB_BACKEND: Use a non-GUI backend so matplotlib can render plots without a display (common in servers/containers)
ENV MATPLOTLIB_BACKEND=Agg
# PYTHONDONTWRITEBYTECODE: Prevents python from gathering .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8080

# Create non-root user with minimum necessary privileges 
# Because running applications as root inside a container is dangerous.
# If someone breaks your app, they would gain root access inside the container and potentially to the host.
# useradd -m -u 1000 scientist: creates a user named scientist, gives them a home directory, and assigns them UID 1000.
# chown -R scientist:scientist /tapsegnn: Gives this user ownership of your application directory. 
RUN useradd -m -u 1000 scientist && chown -R scientist:scientist /tapsegnn 
# switches the container to run as that user (not root).
USER scientist 


# optional command to run the whole training. 
CMD ["python","main.py"]
