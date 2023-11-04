# Use the official Python 3.11 image as the base image.
FROM python:3.11.6-slim

# Expose port 9696 to allow incoming connections.
EXPOSE 9696

# Set the working directory inside the container to /app.
WORKDIR /app

# Copy necessary files from the host into the container.
# Pipfile, Pipfile.lock, predict.py, and random_forest.bin will be copied to /app.
COPY ["Pipfile", "Pipfile.lock", "predict.py", "random_forest.bin", "./"]

# Update the package repository, install pipenv, install build-essential package,
# and install Python packages defined in the Pipfile.
RUN apt-get update && \
    pip install pipenv && \
    apt-get install -y build-essential && \
    pipenv install --system --deploy

# Define the entry point for the container.
# Start the Gunicorn web server, binding it to 0.0.0.0:9696, and running the "predict:app" app.
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
