FROM gcr.io/google.com/cloudsdktool/google-cloud-cli:alpine

# Install system dependencies, latest security patches, and standalone kubectl
RUN apk update && apk upgrade --no-cache && apk add --no-cache \
    build-base \
    git \
    python3 \
    curl \
    && curl -fsSL -o /usr/local/bin/kubectl "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
    && chmod +x /usr/local/bin/kubectl \
    && gcloud components install gke-gcloud-auth-plugin --quiet \
    && pip install --no-cache-dir --upgrade pip cryptography pyopenssl

WORKDIR /workspace
ENV HOME=/workspace
# Route Ramble's caches and repos into the CHS data directory
ENV RAMBLE_USER_DIR=/workspace/.local/share/chs/.ramble

# Create a virtual environment and upgrade pip
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip

# Copy the CHS source code and install
COPY . /app
WORKDIR /app
RUN pip install --no-cache-dir .

WORKDIR /workspace

# Tells the CHS python app how it is being invoked so it logs correctly
ARG CHS_VERSION=latest
ENV CHS_CLI_NAME="docker run chs:${CHS_VERSION}"

ENTRYPOINT ["chs"]
