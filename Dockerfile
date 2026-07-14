FROM gcr.io/google.com/cloudsdktool/google-cloud-cli:slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    python3-venv \
    google-cloud-cli-gke-gcloud-auth-plugin kubectl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
ENV HOME=/workspace
# Route Ramble's caches and repos into the CHS data directory
ENV RAMBLE_USER_DIR=/workspace/.local/share/chs/.ramble

# Create a virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy the CHS source code and install
COPY . /app
WORKDIR /app
RUN pip install --no-cache-dir .

WORKDIR /workspace

# Tells the CHS python app how it is being invoked so it logs correctly
ARG CHS_VERSION=latest
ENV CHS_CLI_NAME="docker run chs:${CHS_VERSION}"

ENTRYPOINT ["chs"]
