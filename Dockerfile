FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY hestia.py eudaimon.py hestia_crypto.py eudaimon_crypto.py ./

# Default command can be overridden per Railway service
CMD ["python", "hestia.py"]
