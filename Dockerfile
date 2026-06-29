FROM python:3.10-slim

WORKDIR /app

# Install basic system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install CPU-only PyTorch (highly optimized for size) and other requirements
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Set dynamic PORT environment variable (Cloud Run default is 8080)
EXPOSE 8080

# Command to run the Flask application
CMD ["python", "server.py"]
