# FROM python:3.11-slim
# WORKDIR /app
# COPY base_executor /app
# ENV LANGUAGE=python
# CMD ["python", "executor_api.py"]



# python/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# ensure latest pip + ca certs
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && python3 -m pip install --upgrade pip setuptools wheel \
 && rm -rf /var/lib/apt/lists/*

# copy shared executor code
COPY base_executor /app

# install python deps (Flask) if present
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

ENV LANGUAGE=python
CMD ["python3", "executor_api.py"]
