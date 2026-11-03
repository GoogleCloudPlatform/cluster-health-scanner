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

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv $VIRTUAL_ENV \
    && pip install --no-cache-dir --require-hashes -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# pip can't hash-check git dependencies, so they must be pinned to full commit SHAs.
COPY pyproject.toml /tmp/pyproject.toml
RUN python3 -c "import tomllib; deps = tomllib.load(open('/tmp/pyproject.toml', 'rb'))['project']['dependencies']; print(*[d for d in deps if ' @ git+' in d], sep=chr(10))" \
    > /tmp/git-requirements.txt \
    && if grep -qvE '@[0-9a-f]{40}( |;|$)' /tmp/git-requirements.txt; then \
        echo "pyproject.toml must pin git dependencies to full commit SHAs" >&2; exit 1; \
    fi \
    && pip install --no-cache-dir --no-deps --no-index --no-build-isolation -r /tmp/git-requirements.txt \
    && rm /tmp/git-requirements.txt /tmp/pyproject.toml

COPY . /app
RUN pip install --no-cache-dir --no-deps --no-index --no-build-isolation /app \
    && pip uninstall -y grpcio-tools \
    && pip check \
    && rm -f /usr/local/lib/python*/ensurepip/_bundled/pip-*.whl

WORKDIR /workspace

# Tells the CHS python app how it is being invoked so it logs correctly
ARG CHS_VERSION=latest
ENV CHS_CLI_NAME="docker run chs:${CHS_VERSION}"

ENTRYPOINT ["chs"]
