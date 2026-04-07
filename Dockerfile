FROM python:3.11-slim

# HF Spaces requires this label
LABEL space_sdk="docker"

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create empty __init__.py for env package
RUN touch env/__init__.py

# HF Spaces runs on port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
