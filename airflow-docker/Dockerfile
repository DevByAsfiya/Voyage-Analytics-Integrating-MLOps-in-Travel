FROM apache/airflow:2.10.0

USER root

# Install necessary dependencies like git
RUN apt-get update && \
    apt-get -y install git && \
    apt-get clean

# Revert to the airflow user for safety and to maintain security context
USER airflow

# Make sure airflow is installed (this might be redundant if using the official image)
COPY requirements.txt .

# 2. Install from the file
RUN pip install --no-cache-dir -r requirements.txt
