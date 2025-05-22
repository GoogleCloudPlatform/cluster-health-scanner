# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
ARG TARGETARCH

FROM ubuntu:latest

WORKDIR /app

RUN apt-get update &&\
    apt-get install -y --no-install-recommends git make gcc g++ util-linux software-properties-common openssh-server ca-certificates curl jq python3 python3-pip python3-venv &&\
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/${TARGETARCH}/kubectl" &&\
    chmod +x kubectl

RUN mkdir /var/run/sshd && chmod 0755 /var/run/sshd &&\
  ssh-keygen -t rsa -f /root/.ssh/google_compute_engine -b 2048 -P "" &&\
  cp /root/.ssh/google_compute_engine.pub /root/.ssh/authorized_keys &&\
  chmod 644 /root/.ssh/authorized_keys &&\
  chmod 644 /root/.ssh/google_compute_engine.pub

ARG TARGETARCH
# Install & setup Helm - https://helm.sh/docs/intro/install/
RUN echo "Downloading Helm for $TARGETARCH" \
  && curl --tlsv1.2 -fsSL -o helm.tar.gz "https://get.helm.sh/helm-v3.17.2-linux-${TARGETARCH}.tar.gz" \
  && tar -zxvf helm.tar.gz
RUN mv "linux-${TARGETARCH}/helm" /usr/local/bin/helm \
  && chmod +x /usr/local/bin/helm \
  && rm helm.tar.gz \
  && helm version

COPY src/common.proto /app/
COPY src/health_runner/health_results.proto /app/
COPY src/health_runner/health_runner_config.proto /app/

RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

RUN pip install --no-cache-dir grpcio-tools
RUN pip install kubernetes
RUN pip install protobuf
RUN pip install google-cloud-storage
RUN python3 -m grpc_tools.protoc -I /app/ --python_out=. --pyi_out=. --grpc_python_out=. --experimental_editions /app/common.proto
RUN python3 -m grpc_tools.protoc -I /app/ --python_out=. --pyi_out=. --grpc_python_out=. --experimental_editions /app/health_results.proto
RUN python3 -m grpc_tools.protoc -I /app/ --python_out=. --pyi_out=. --grpc_python_out=. --experimental_editions /app/health_runner_config.proto

# Health runner
COPY src/checker_common.py .
COPY src/health_runner/health_runner.py .
COPY src/health_runner/nccl_runner.py .

# Health checks
# Helm charts
COPY deploy/helm/health_checks/ health_checks/

RUN chmod -R g+rwx /app/
RUN chgrp -R 1000 /app/

ENV PYTHONUNBUFFERED=1

CMD ["python3", "/app/health_runner.py"]
