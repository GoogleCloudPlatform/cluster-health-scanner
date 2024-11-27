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

FROM us-docker.pkg.dev/gce-ai-infra/health-check/nccl-plugin:ubuntu22
#FROM us-docker.pkg.dev/gce-ai-infra/gpudirect-tcpx/nccl-plugin-gpudirecttcpx-nightly-cuda12.0:2024_03_04
#FROM  us-docker.pkg.dev/kernel-net-team/clouda4-nccl-dev/nccl-plugin-gib:latest
#FROM us-docker.pkg.dev/gce-ai-infra/health-check/nccl-healthcheck:a3ultra

WORKDIR /scripts
RUN apt-get update && apt-get install -y openssh-server python3.10 ca-certificates curl python3-pip slurm-client &&\
  update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 &&\
  mkdir /var/run/sshd && chmod 0755 /var/run/sshd 

RUN ssh-keygen -t rsa -f /root/.ssh/google_compute_engine -b 2048 -P "" &&\
  cp /root/.ssh/google_compute_engine.pub /root/.ssh/authorized_keys &&\
  chmod 644 /root/.ssh/authorized_keys &&\
  chmod 644 /root/.ssh/google_compute_engine.pub

RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && chmod +x kubectl

COPY src/nccl_healthcheck/config.proto /scripts/
RUN pip install grpcio-tools
RUN python3 -m grpc_tools.protoc -I /scripts/ --python_out=. --pyi_out=. --grpc_python_out=. --experimental_editions /scripts/config.proto

COPY src/nccl_healthcheck/nccl_startup.py /scripts/
COPY src/nccl_healthcheck/config.py /scripts/
COPY src/checker_common.py /scripts/
COPY src/metrics.py /scripts/
COPY src/nccl_healthcheck/a3/ a3/
COPY src/nccl_healthcheck/a3plus/ a3plus/
COPY src/nccl_healthcheck/run-nccl-combined-plugins.sh .
RUN chmod +x /scripts/run-nccl-combined-plugins.sh

RUN apt-get install -y dnsutils

# ENTRYPOINT ["python3", "/scripts/nccl_startup.py"]
# ENTRYPOINT ["bash", "/scripts/a3plus/nccl_healthcheck.sh"]
