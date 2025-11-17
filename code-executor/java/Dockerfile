# FROM openjdk:21-slim
# WORKDIR /app
# RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
# COPY ../base_executor /app
# ENV LANGUAGE=java
# CMD ["python3", "executor_api.py"]


# FROM eclipse-temurin:21-jdk

# WORKDIR /app

# # Install python if executor needs it and some utilities
# RUN apt-get update \
#  && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates \
#  && rm -rf /var/lib/apt/lists/*

# ENV JAVA_HOME=/usr/lib/jvm/temurin-21-jdk
# ENV PATH=$JAVA_HOME/bin:$PATH


# COPY base_executor /app

# CMD ["python3", "executor_api.py"]






# Example for rust/java/cpp/csharp (Debian/Ubuntu based)

# code-executor/java/Dockerfile
FROM eclipse-temurin:21-jdk

WORKDIR /app

# Install only system packages needed for venv and CA certs.
# Do NOT run 'python3 -m pip install --upgrade ...' against system Python.
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Create an isolated venv and upgrade pip inside the venv (safe).
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip setuptools wheel

# Use the venv's bin by default for all subsequent RUN and for the container runtime.
ENV PATH=/opt/venv/bin:$PATH

# Copy shared executor code. Make sure your docker-compose build context is the project root
# so "COPY base_executor /app" finds that directory.
COPY base_executor /app

# If there is a requirements.txt at the build context root, install into the venv's pip.
# This installs Flask (you said requirements.txt contains Flask) into the venv only.
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

ENV LANGUAGE=java

# Run the python executor inside the venv
CMD ["python", "executor_api.py"]



# (Optional) copy entrypoint or other files as needed
# COPY . /app
# CMD ["bash"]
