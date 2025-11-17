FROM node:22-alpine
WORKDIR /app
RUN npm install -g ts-node typescript
RUN apk add --no-cache python3 py3-pip
COPY base_executor /app
ENV LANGUAGE=typescript
CMD ["python3", "executor_api.py"]
