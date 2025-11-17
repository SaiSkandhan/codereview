# FROM mono:latest
# WORKDIR /app
# RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
# COPY base_executor /app
# ENV LANGUAGE=csharp
# CMD ["python3", "executor_api.py"]

# code-executor/csharp/Dockerfile
FROM mono:6.12

WORKDIR /app

# Update and install python for the executor API
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-pip \
 && rm -rf /var/lib/apt/lists/*

# Copy shared executor code (base_executor must be in the build context)
COPY base_executor /app

ENV LANGUAGE=csharp
CMD ["python3", "executor_api.py"]



