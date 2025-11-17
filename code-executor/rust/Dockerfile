FROM rust:1.82
WORKDIR /app
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
COPY base_executor /app
ENV LANGUAGE=rust
CMD ["python3", "executor_api.py"]
