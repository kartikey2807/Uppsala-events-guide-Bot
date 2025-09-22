# Use lightweight Python image
FROM python:3.11-slim

# Set work directory inside the container
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY scraper.py .

# Run the script by default
CMD ["python", "scraper.py"]