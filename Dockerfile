# --------------------------------------------------
# Base image: official Python
# --------------------------------------------------
FROM python:3.11-slim

# --------------------------------------------------
# Set working directory inside container
# --------------------------------------------------
WORKDIR /app

# --------------------------------------------------
# Copy dependency definitions
# --------------------------------------------------
COPY requirements-inference.txt .

# --------------------------------------------------
# Install dependencies
# --------------------------------------------------
RUN pip install --no-cache-dir -r requirements-inference.txt

# --------------------------------------------------
# Copy project files
# --------------------------------------------------
COPY src/ src/
COPY inference/ inference/
COPY models/ models/
COPY config/config.yaml .

# --------------------------------------------------
# Expose API port
# --------------------------------------------------
EXPOSE 8000

# --------------------------------------------------
# Start FastAPI app
# --------------------------------------------------
CMD ["uvicorn", "inference.api:app", "--host", "0.0.0.0", "--port", "8000"]
