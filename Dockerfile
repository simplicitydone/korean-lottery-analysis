FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV LOTTERY_HOST=0.0.0.0
ENV LOTTERY_PORT=5000
EXPOSE 5000
CMD ["python", "serve.py"]
