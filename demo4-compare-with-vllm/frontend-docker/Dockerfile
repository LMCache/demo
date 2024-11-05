# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the contents of the frontend-docker directory into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install streamlit openai transformers sentencepiece

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "frontend.py", "--server.port=8501", "--server.address=0.0.0.0"]
