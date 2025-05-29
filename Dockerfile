# FROM python:3.10-slim
# FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# # Install Python and pip
# RUN apt-get update && apt-get install -y python3 python3-pip ffmpeg \
#     && rm -rf /var/lib/apt/lists/*

# # Set work directory
# WORKDIR /app

# COPY backend/har/assets/COFFEESHOP_1.mp4 /home/app/COFFEESHOP_1.mp4
# COPY backend/har/weights/avak_b16.pt /home/app/avak_b16.pt
# COPY backend/postprocess/mobile_sam.pt /home/app/mobile_sam.pt

# # Copy requirements and install
# COPY requirements.txt .
# RUN pip3 install --no-cache-dir -r requirements.txt

# # Copy the rest of the code
# COPY . .

# # Expose the port Flask runs on
# EXPOSE 5000

# # Run the app with gunicorn (replace 'app:app' with your module and Flask app name)
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]




#Difference
#FROM python:3.10
FROM nvidia/cuda:12.9.0-base-ubuntu24.04

WORKDIR /app

RUN apt-get update && apt-get install -y python3 python3-pip libgl1 ffmpeg && rm -rf /var/lib/apt/lists/*


# Install system dependencies for OpenCV
#RUN apt-get update && apt-get install -y libgl1 ffmpeg && rm -rf /var/lib/apt/lists/*

COPY backend/har/assets/COFFEESHOP_1.mp4 /home/app/COFFEESHOP_1.mp4
COPY backend/har/weights/avak_b16.pt /home/app/avak_b16.pt
COPY backend/postprocess/mobile_sam.pt /home/app/mobile_sam.pt

COPY requirements.txt .

#RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY . .

EXPOSE 5000

#CMD ["python", "app.py"]
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
# CMD ["gunicorn --workers=1 --timeout=300 --bind 0.0.0.0:5000 app:app"]