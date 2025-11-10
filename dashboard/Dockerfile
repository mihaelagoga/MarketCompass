# Dockerfile

# 1. Use an official Python base image
FROM python:3.10-slim

# 2. Set environment variables
ENV PYTHONUNBUFFERED True

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your application code into the container
COPY . .

# 6. Expose the port Gunicorn will run on
#    Cloud Run expects 8080 by default.
EXPOSE 8080

# 7. Define the command to run your app
#    This tells Gunicorn to run the 'server' object
#    from your 'app.py' file.
#    'app:server' -> [filename]:[variable_name]
CMD ["gunicorn", "--bind", ":8080", "--workers", "1", "app:server"]