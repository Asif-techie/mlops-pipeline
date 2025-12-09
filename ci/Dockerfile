# simple production Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# copy project
COPY . /app

EXPOSE 8000

# Use uvicorn for serving
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
