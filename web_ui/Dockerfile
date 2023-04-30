FROM python:3.11-bullseye AS BASE
WORKDIR /app

COPY api/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY api/model /app/model
COPY api/age_estimator.py /app/age_estimator.py
COPY api/main.py /app/main.py
COPY api/__init__.py /app/__init__.py

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
