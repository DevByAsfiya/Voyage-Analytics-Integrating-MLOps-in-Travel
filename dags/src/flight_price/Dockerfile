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
