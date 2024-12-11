# Use an official Python image as the base
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the application code to the container
COPY . /app

# Verify that the requirements.txt is copied correctly (for debugging)
RUN ls /app

# Install dependencies
RUN pip install --no-cache-dir django
RUN pip install -r /app/requirements.txt

# Collect static files
RUN Python manage.py collectstatic --noinput

# Expose port 8080
Expose 8080

# Command to run the Django app
CMD ["python", "manage.py", "runserver", "0.0.0.0:8080"]