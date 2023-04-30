FROM python:3.11.2-slim-bullseye AS BASE

WORKDIR /app

COPY web_ui/requirements.txt /app/web_ui/requirements.txt
RUN pip install -r /app/web_ui/requirements.txt

COPY web_ui/main.py /app/main.py
COPY web_ui/__init__.py /app/web_ui/__init__.py

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "/app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
