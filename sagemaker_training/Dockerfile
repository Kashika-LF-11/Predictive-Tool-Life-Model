# Use a slim python 3.13 base (matches your Pipfile requiring 3.13.x)
FROM python:3.13-slim

# Install minimal system deps needed for some Python wheels (xgboost/lightgbm may need build tools)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       ca-certificates \
       git \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/app:${PATH}"
ENV PYTHONPATH=.

# Upgrade pip and install pipenv (optional — using Pipfile in repo)
RUN pip install --upgrade pip
RUN pip install pipenv

WORKDIR /opt/app

# Copy lockfiles (if you want deterministic installs using pipenv)
COPY Pipfile Pipfile.lock ./
# Use --deploy to ensure we match lockfile exactly; --system installs into system pip (no virtualenv in container)
RUN pipenv install --deploy --system --dev || (echo "pipenv install failed — try using requirements.txt instead" && exit 1)

# Add source
COPY src/ ./
COPY train.py ./

# make sure train script is executable
RUN chmod +x /opt/app/train.py

# SageMaker expects the training entrypoint to be an executable called 'train' in many examples,
# but using python entrypoint is also fine. We'll use python to run train.py so CLI args work.
ENTRYPOINT ["python", "/opt/app/train.py"]

# Default CMD left empty to allow boto3 create_training_job hyperparameters to control behavior
CMD []