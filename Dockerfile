FROM gcr.io/google.com/cloudsdktool/google-cloud-cli:alpine

# Install system dependencies, latest security patches, and standalone kubectl
RUN apk update && apk upgrade --no-cache && apk add --no-cache \
    build-base \
    git \
    python3 \
    curl \
    && curl -fsSL -o /usr/local/bin/kubectl "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
    && chmod +x /usr/local/bin/kubectl \
    && gcloud components install gke-gcloud-auth-plugin --quiet

COPY requirements-system.txt /tmp/requirements-system.txt
RUN pip install --no-cache-dir --require-hashes -r /tmp/requirements-system.txt \
    && rm /tmp/requirements-system.txt

WORKDIR /workspace
ENV HOME=/workspace
# Route Ramble's caches and repos into the CHS data directory
ENV RAMBLE_USER_DIR=/workspace/.local/share/chs/.ramble

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy the CHS source code and install
COPY . /app
WORKDIR /app
# pip can't hash-check git dependencies, so clone Ramble at its pinned commit.
RUN ramble_clone_args=$(sed -n "s|^ *[\"']ramble @ git+\(https://[^\"'@]*\)@\([0-9a-f]\{40\}\)[\"'].*|--revision=\2 \1|p" pyproject.toml) \
    && { [ -n "$ramble_clone_args" ] || { echo "pyproject.toml must pin ramble to a full git commit SHA" >&2; exit 1; }; } \
    && python3 -m venv $VIRTUAL_ENV \
    && pip install --no-cache-dir --require-hashes -r requirements.txt \
    && git -c advice.detachedHead=false clone -q --depth 1 $ramble_clone_args /tmp/ramble \
    && pip install --no-cache-dir --no-deps --no-index --no-build-isolation /tmp/ramble . \
    && pip uninstall -y grpcio-tools \
    && pip check \
    && rm -rf /tmp/ramble \
    && rm -f /usr/lib/python*/ensurepip/_bundled/pip-*.whl /usr/local/lib/python*/ensurepip/_bundled/pip-*.whl

WORKDIR /workspace

# Tells the CHS python app how it is being invoked so it logs correctly
ARG CHS_VERSION=latest
ENV CHS_CLI_NAME="docker run chs:${CHS_VERSION}"

ENTRYPOINT ["chs"]
