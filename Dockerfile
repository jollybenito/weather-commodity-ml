FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (specifically libgomp1 for LightGBM)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose the port Railway uses
EXPOSE $PORT

# Command to run the application
CMD streamlit run app.py --server.port $PORT --server.address 0.0.0.0
