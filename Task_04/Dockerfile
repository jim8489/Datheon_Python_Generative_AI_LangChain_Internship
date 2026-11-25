# Build from the internship repository root:
# docker build -f Task_04/Dockerfile -t documind-task04 .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TASK2_PATH=/app/Task_02

WORKDIR /app

COPY Task_02/requirements.txt /app/Task_02/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /app/Task_02/requirements.txt

COPY Task_02/app /app/Task_02/app
COPY Task_04 /app/Task_04

WORKDIR /app/Task_04
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)" || exit 1

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
