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

# Dockerfile for building the nccl-pairwise image.
ARG BASE_IMAGE=us-docker.pkg.dev/gce-ai-infra/gpudirect-gib/nccl-plugin-gib:v1.0.2

FROM ${BASE_IMAGE}

WORKDIR /scripts
RUN apt-get update && apt-get install -y net-tools openssh-server python3.10 ca-certificates curl python3-pip &&\
  update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 &&\
  mkdir /var/run/sshd && chmod 0755 /var/run/sshd &&\
  ssh-keygen -t rsa -f /root/.ssh/google_compute_engine -b 2048 -P "" &&\
  cp /root/.ssh/google_compute_engine.pub /root/.ssh/authorized_keys &&\
  chmod 644 /root/.ssh/authorized_keys &&\
  chmod 644 /root/.ssh/google_compute_engine.pub

ARG TARGETARCH
RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/${TARGETARCH}/kubectl" && chmod +x kubectl

COPY src/health_runner/health_results.proto /scripts/
COPY src/health_runner/health_runner_config.proto /scripts/
COPY src/nccl_healthcheck/config.proto /scripts/
COPY src/common.proto /scripts/
RUN pip install grpcio-tools
RUN pip install kubernetes
RUN pip install google-cloud-storage
RUN python3 -m grpc_tools.protoc -I /scripts/ --python_out=. --pyi_out=. --grpc_python_out=. --experimental_editions /scripts/config.proto
RUN python3 -m grpc_tools.protoc -I /scripts/ --python_out=. --pyi_out=. --grpc_python_out=. --experimental_editions /scripts/health_results.proto
RUN python3 -m grpc_tools.protoc -I /scripts/ --python_out=. --pyi_out=. --grpc_python_out=. --experimental_editions /scripts/common.proto
RUN python3 -m grpc_tools.protoc -I /scripts/ --python_out=. --pyi_out=. --grpc_python_out=. --experimental_editions /scripts/health_runner_config.proto

COPY src/nccl_healthcheck/nccl_startup.py /scripts/
COPY src/nccl_healthcheck/config.py /scripts/
COPY src/nccl_healthcheck/run-nccl-combined-plugins.sh .
RUN chmod +x /scripts/run-nccl-combined-plugins.sh
COPY src/checker_common.py /scripts/
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python3", "/scripts/nccl_startup.py"]