# Optional Linux verification; the organizers supply their own container.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY infra/requirements.runtime.txt /tmp/requirements.runtime.txt
RUN pip install --no-cache-dir -r /tmp/requirements.runtime.txt
WORKDIR /task
ENTRYPOINT ["python3", "/solution/solution.py"]
