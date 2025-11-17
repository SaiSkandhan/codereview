# FROM node:22-alpine
# WORKDIR /app
# COPY --from=python:3.11-slim /usr/local/bin/python3 /usr/local/bin/python3
# RUN npm install -g typescript ts-node
# COPY base_executor /app
# ENV LANGUAGE=javascript
# CMD ["python3", "executor_api.py"]


# code-executor/javascript/Dockerfile
# code-executor/javascript/Dockerfile
FROM node:22-alpine

WORKDIR /app

# Install python, virtualenv support and small build tools needed by some wheels.
# Then create a venv and use its pip to manage Python packages (avoids PEP 668).
RUN apk add --no-cache python3 py3-virtualenv build-base ca-certificates \
 && python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip setuptools wheel \
 && rm -rf /var/cache/apk/*

ENV PATH=/opt/venv/bin:$PATH

# Install TypeScript tooling (if needed)
RUN npm install -g ts-node typescript

# Copy shared executor code and requirements
COPY base_executor /app

# Install Python requirements into the venv (if present)
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

ENV LANGUAGE=javascript
CMD ["python3", "executor_api.py"]


