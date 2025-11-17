# FROM golang:1.23-alpine
# WORKDIR /app
# RUN apk add --no-cache python3 py3-pip
# COPY base_executor /app
# ENV LANGUAGE=go
# CMD ["python3", "executor_api.py"]






# go/Dockerfile
# code-executor/go/Dockerfile
FROM golang:1.23-alpine

WORKDIR /app

# Install python, venv support and build tools then create venv and upgrade venv pip.
RUN apk add --no-cache python3 py3-virtualenv build-base ca-certificates \
 && python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip setuptools wheel \
 && rm -rf /var/cache/apk/*

ENV PATH=/opt/venv/bin:$PATH

# copy shared executor code
COPY base_executor /app

# install python deps into the venv if present
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

ENV LANGUAGE=go
CMD ["python3", "executor_api.py"]


