FROM mcr.microsoft.com/playwright/python:v1.59.0-noble

WORKDIR /app

# Install Python deps before copying source so this layer is cached
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Playwright browsers are pre-installed in the base image (chromium only needed)
RUN playwright install chromium

COPY . .

ENV DB_PATH=/app/data/decido.db

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
