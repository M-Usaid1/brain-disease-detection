FROM python:3.11-slim

WORKDIR /app

# Install Linux libraries required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Railway provides PORT automatically
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}