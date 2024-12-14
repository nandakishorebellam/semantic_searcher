# Use an official Python slim image as the base
FROM python:3.12-slim

# Set environment variables to avoid issues with buffering
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Set the working directory in the container
WORKDIR /app

# Copy only necessary files initially (helps with caching)
COPY requirements.txt /app/

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install dependencies in chunks to avoid memory issues
RUN pip install django==5.1.3 \
    && pip install tensorflow-cpu==2.18.0 \
    && pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 -f https://download.pytorch.org/whl/torch_stable.html \
    && pip install -r /app/requirements.txt

# Copy the rest of the application code
COPY . /app

# Collect static files
RUN python manage.py collectstatic --noinput

# Expose port 8080
EXPOSE 8080

# Command to run the Django app
CMD ["python", "manage.py", "runserver", "0.0.0.0:8080"]