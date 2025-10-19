FROM python:3.13.9-slim-trixie
WORKDIR /opt
COPY requirements.txt .
COPY generator-rules-processor.py /opt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
CMD ["python3", "generator-rules-processor.py"]