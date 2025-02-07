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

# Build Stage
FROM ubuntu:latest

ENV PYTHONUNBUFFERED=1
WORKDIR /scripts

# Install build dependencies and utilities
RUN apt-get update &&\
    apt-get -y upgrade &&\
    apt-get -y autoremove &&\
    apt-get install -y \
        git make gcc g++ util-linux software-properties-common \
        openssh-server ca-certificates curl jq \
        python3-kubernetes python3 python3-pip python3-venv

RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" &&\
    chmod +x kubectl

# Clone and build Neper
RUN git clone https://github.com/google/neper.git &&\
    cd /scripts/neper &&\
    make tcp_stream &&\
    cp /scripts/neper/tcp_stream /scripts/ &&\
    rm -rf /scripts/neper/* &&\
    chmod +x /scripts/tcp_stream

RUN mkdir /var/run/sshd && chmod 0755 /var/run/sshd &&\
  ssh-keygen -t rsa -f /root/.ssh/google_compute_engine -b 2048 -P "" &&\
  cp /root/.ssh/google_compute_engine.pub /root/.ssh/authorized_keys &&\
  chmod 644 /root/.ssh/authorized_keys &&\
  chmod 644 /root/.ssh/google_compute_engine.pub

# Disable ssh login grace period to prevent race condition vulnerability
RUN echo "LoginGraceTime 0" >> /etc/ssh/sshd_config

RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

RUN pip install kubernetes
RUN pip install google-cloud-storage
RUN pip install --no-cache-dir grpcio-tools
COPY src/health_runner/health_results.proto /scripts/
COPY src/health_runner/health_runner_config.proto /scripts/
COPY src/common.proto /scripts/

COPY src/neper_healthcheck/neper_runner.py .
COPY src/checker_common.py .
RUN python3 -m grpc_tools.protoc -I /scripts/ --python_out=. --pyi_out=. --grpc_python_out=. --experimental_editions /scripts/health_results.proto
RUN python3 -m grpc_tools.protoc -I /scripts/ --python_out=. --pyi_out=. --grpc_python_out=. --experimental_editions /scripts/common.proto
RUN python3 -m grpc_tools.protoc -I /scripts/ --python_out=. --pyi_out=. --grpc_python_out=. --experimental_editions /scripts/health_runner_config.proto

RUN chmod +x /scripts/neper_runner.py /scripts/checker_common.py
CMD ["python3", "/scripts/neper_runner.py"]