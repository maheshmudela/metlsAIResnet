
#Here is a multi-stage Dockerfile optimized for production. It uses a build stage to compile dependencies and a smaller final image to reduce size and improve security


# Stage 1: Build environment
FROM python:3.11-slim as builder

# Set the working directory in the container
WORKDIR /home/Melts

# Copy only the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final (runtime) image
FROM python:3.11-slim as runtime


# Copy installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy the application code, data, and model
COPY melts.py .
COPY data/ ./data/
COPY models/ ./models/

# Set a non-root user for security (best practice)
#RUN groupadd --system appuser && useradd --system -g appuser appuser
RUN  useradd appuser
USER appuser

# Expose the port that Uvicorn will run on
EXPOSE 8080

# Set environment variable for the log access key (best practice)
#ENV LOG_ACCESS_KEY="1978"

# Command to run the FastAPI app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]

