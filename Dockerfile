# # Use a specific Python base image
# FROM python:3.9 

# # Set the working directory for your code
# WORKDIR /app

# # Copy everything in your root project folder into the container
# COPY . /app

# # Install dependencies from requirements.txt
# RUN pip install --upgrade pip
# RUN pip install -r requirements.txt

# # Set environment variables for cleaner Python logs and no .pyc files
# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1

# # Expose the port for Flask (default is 5000)
# EXPOSE 5000

# # Start the Flask app
# CMD ["python", "app.py"]

# Use an official Python base image
FROM python:3.9-slim

# Set Python-related environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system build dependencies only if you need them (optional)
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
#     && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker layer cache
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the project files
COPY . .

# Expose Flask default port
EXPOSE 5000

# Start the Flask app
CMD ["python", "app.py"]
