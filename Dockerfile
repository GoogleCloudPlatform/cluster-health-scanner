FROM gcr.io/google.com/cloudsdktool/google-cloud-cli:580.0.0-alpine

# Install system dependencies
RUN apk add --no-cache \
    build-base \
    git \
    python3 \
    kubectl \
    && gcloud components install gke-gcloud-auth-plugin --quiet

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
