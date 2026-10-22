FROM python:3.12-slim-trixie

ARG TARGETARCH

# gcloud runs on the image's Python instead of its bundled one.
ENV CLOUDSDK_PYTHON=/usr/local/bin/python3
ENV PATH="/google-cloud-sdk/bin:$PATH"

# Install system dependencies, latest security patches, gcloud with the GKE auth plugin, and standalone kubectl
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    openssh-client \
    && rm -rf /var/lib/apt/lists/* \
    && arch="${TARGETARCH:-$(uname -m | sed -e 's/x86_64/amd64/' -e 's/aarch64/arm64/')}" \
    && gcloud_arch=$(echo "$arch" | sed -e 's/amd64/x86_64/' -e 's/arm64/arm/') \
    && curl -fsSL "https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-${gcloud_arch}.tar.gz" | tar -xz -C / \
    && rm -rf /google-cloud-sdk/platform/bundledpythonunix /google-cloud-sdk/.install/bundled-python* \
    && gcloud components install gke-gcloud-auth-plugin --quiet \
    && rm -rf /google-cloud-sdk/.install/.backup \
    && curl -fsSL -o /usr/local/bin/kubectl "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/${arch}/kubectl" \
    && chmod +x /usr/local/bin/kubectl

COPY requirements-system.txt /tmp/requirements-system.txt
RUN python3 -m pip install --no-cache-dir --require-hashes -r /tmp/requirements-system.txt \
    && rm /tmp/requirements-system.txt

ENV HOME=/workspace
# Route Ramble's caches and repos into the CHS data directory
ENV RAMBLE_USER_DIR=/workspace/.local/share/chs/.ramble

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy the CHS source code and install
COPY . /app
WORKDIR /app
# pip can't hash-check git dependencies, so fetch Ramble at its pinned commit.
RUN ramble_pin=$(sed -n "s|^ *[\"']ramble @ git+\(https://[^\"'@]*\)@\([0-9a-f]\{40\}\)[\"'].*|\1 \2|p" pyproject.toml) \
    && { [ -n "$ramble_pin" ] || { echo "pyproject.toml must pin ramble to a full git commit SHA" >&2; exit 1; }; } \
    && set -- $ramble_pin \
    && python3 -m venv $VIRTUAL_ENV \
    && pip install --no-cache-dir --require-hashes -r requirements.txt \
    && git init -q /tmp/ramble \
    && git -C /tmp/ramble fetch -q --depth 1 "$1" "$2" \
    && git -C /tmp/ramble -c advice.detachedHead=false checkout -q FETCH_HEAD \
    && pip install --no-cache-dir --no-deps --no-index --no-build-isolation /tmp/ramble . \
    && pip uninstall -y grpcio-tools \
    && pip check \
    && rm -rf /tmp/ramble \
    && rm -f /usr/local/lib/python*/ensurepip/_bundled/pip-*.whl

WORKDIR /workspace

# Tells the CHS python app how it is being invoked so it logs correctly
ARG CHS_VERSION=latest
ENV CHS_CLI_NAME="docker run chs:${CHS_VERSION}"

ENTRYPOINT ["chs"]
