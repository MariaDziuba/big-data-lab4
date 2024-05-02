FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update
RUN apt-get install -y gcc python3-dev 
RUN pip install --upgrade pip setuptools

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD python -m src.train

EXPOSE 8001

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8001"]