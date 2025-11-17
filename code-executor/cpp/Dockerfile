# FROM gcc:13
# WORKDIR /app
# RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
# COPY base_executor /app
# ENV LANGUAGE=cpp
# CMD ["python3", "executor_api.py"]



# cpp/Dockerfile

# code-executor/cpp/Dockerfile
FROM gcc:13

WORKDIR /app

# Install system packages required to create a venv and to build native wheels if needed.
# Do NOT try to 'pip upgrade' system pip; instead create and use a venv.
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip ca-certificates build-essential \
 && rm -rf /var/lib/apt/lists/*

# Create virtualenv and upgrade pip inside the venv (safe, doesn't touch system Python).
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip setuptools wheel

# Make the venv python/pip the default in this image
ENV PATH=/opt/venv/bin:$PATH

# Copy shared executor code (assumes docker-compose build context is the project root
# and base_executor exists there as a directory).
COPY base_executor /app

# Install Python requirements (into the venv). If requirements.txt is absent, skip.
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

ENV LANGUAGE=cpp
CMD ["python3", "executor_api.py"]
