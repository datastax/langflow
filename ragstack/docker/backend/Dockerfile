FROM python:3.11-slim

RUN apt-get update && apt-get install gcc g++ git make -y && apt-get clean \
	&& rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH
RUN pip install --no-cache-dir ragstack-ai-langflow --user

WORKDIR $HOME/app

ENV PYTHONUNBUFFERED=1
ENV LANGFLOW_DATABASE_URL=sqlite:///$HOME/app/langflow.db
CMD ["python", "-m", "langflow", "run", "--host", "0.0.0.0"]
