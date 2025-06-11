# Pull base image
FROM python:3.10

# Set environment variables
ENV PIP_DISABLE_PIP_VERSION_CHECK 1
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install dependencies
COPY ./requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy project
COPY . .

# Run migrations, load fixtures and start server
CMD sh -c "python manage.py migrate && python manage.py runserver 0.0.0.0:8000"