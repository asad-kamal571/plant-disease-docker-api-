# Use lightweight Python 3.10 image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install OS dependencies for librosa, matplotlib, etc.
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python libraries
COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Copy all app files (Flask + Streamlit + model)
COPY . .

# Expose both Flask and Streamlit ports
EXPOSE 5000
EXPOSE 8501

# Run both Streamlit and Flask using bash
CMD bash -c "streamlit run streamlit_app.py --server.port=8501 & python app.py"
