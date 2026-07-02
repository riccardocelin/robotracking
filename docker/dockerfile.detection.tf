FROM python:3.10-slim

WORKDIR /app
COPY app/detection_service_tf ./detection_service_tf

ENV PYTHONPATH=/app

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir tensorflow fastapi uvicorn numpy pydantic

EXPOSE 8000

CMD ["uvicorn", "detection_service_tf.api:app", "--host", "0.0.0.0", "--port", "8000"]