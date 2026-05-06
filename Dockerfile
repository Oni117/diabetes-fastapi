FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt.txt .

RUN pip install --no-cache-dir -r requirements.txt.txt

COPY . .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]