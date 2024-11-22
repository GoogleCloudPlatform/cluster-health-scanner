# # FROM us-docker.pkg.dev/gce-ai-infra/gpudirect-tcpx/nccl-plugin-gpudirecttcpx-nightly-cuda12.0:2024_03_04

# # Use Ubuntu as the base image
# FROM ubuntu:20.04

# WORKDIR /scripts
# # Set environment variables for non-interactive installs
# ENV DEBIAN_FRONTEND=noninteractive

# # Update and install dependencies
# RUN apt-get update && apt-get install -y \
#     software-properties-common \
#     && add-apt-repository ppa:deadsnakes/ppa \
#     && apt-get update && apt-get install -y python3.9 python3.9-venv python3.9-distutils \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# # Set Python3.9 as default
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# # Verify Python version
# RUN python3 --version

# RUN apt-get update && apt-get install -y openssh-server  ca-certificates curl python3-pip &&\
#   # update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 &&\
#   mkdir /var/run/sshd && chmod 0755 /var/run/sshd &&\
#   ssh-keygen -t rsa -f /root/.ssh/google_compute_engine -b 2048 -P "" &&\
#   cp /root/.ssh/google_compute_engine.pub /root/.ssh/authorized_keys &&\
#   chmod 644 /root/.ssh/authorized_keys &&\
#   chmod 644 /root/.ssh/google_compute_engine.pub



# RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && chmod +x kubectl

# COPY src/nccl_healthcheck/config.proto /scripts/
# RUN pip install grpcio-tools
# RUN python3 -m grpc_tools.protoc -I /scripts/ --python_out=. --pyi_out=. --grpc_python_out=. --experimental_editions /scripts/config.proto

# COPY src/nccl_healthcheck/nccl_startup.py /scripts/
# COPY src/nccl_healthcheck/config.py /scripts/
# COPY src/checker_common.py /scripts/
# COPY src/metrics.py /scripts/
# COPY src/nccl_healthcheck/a3/ a3/
# COPY src/nccl_healthcheck/a3plus/ a3plus/
# COPY src/nccl_healthcheck/run-nccl-combined-plugins.sh .
# RUN chmod +x /scripts/run-nccl-combined-plugins.sh
# ENV PYTHONUNBUFFERED=1

# RUN apt-get install -y dnsutils

# ENTRYPOINT ["python3", "/scripts/nccl_startup.py"]

# FROM us-docker.pkg.dev/gce-ai-infra/gpudirect-tcpxo/nccl-plugin-gpudirecttcpx-dev:v1.0.1
# FROM us-docker.pkg.dev/gce-ai-infra/gpudirect-tcpx/nccl-plugin-gpudirecttcpx-nightly-cuda12.0:2024_03_04
# FROM us-docker.pkg.dev/kernel-net-team/gpudirect-tcpx/nccl-plugin-gpudirecttcpx-pre-test-cuda12.2:cl_630525599
# FROM ubuntu:20.04
# ENV DEBIAN_FRONTEND=noninteractive



# RUN apt-get update && apt-get install -y openssh-server python3.9 ca-certificates curl python3-pip dnsutils &&\
#   update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 &&\
#   mkdir /var/run/sshd && chmod 0755 /var/run/sshd &&\
#   ssh-keygen -t rsa -f /root/.ssh/google_compute_engine -b 2048 -P "" &&\
#   cp /root/.ssh/google_compute_engine.pub /root/.ssh/authorized_keys &&\
#   chmod 644 /root/.ssh/authorized_keys &&\
#   chmod 644 /root/.ssh/google_compute_engine.pub

# RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && chmod +x kubectl

# RUN apt install -y --no-install-recommends \
#     git openssh-server wget iproute2 vim libopenmpi-dev build-essential \
#     cmake gdb python3 \
#     protobuf-compiler libprotobuf-dev rsync libssl-dev libcurl4-openssl-dev \
#   && rm -rf /var/lib/apt/lists/*
# # Setup SSH to use port 222
# RUN cd /etc/ssh/ && sed --in-place='.bak' 's/#Port 22/Port 222/' sshd_config && \
#     sed --in-place='.bak' 's/#PermitRootLogin prohibit-password/PermitRootLogin prohibit-password/' sshd_config
# RUN ssh-keygen -t rsa -b 4096 -q -f /root/.ssh/id_rsa -N "" -C ""
# RUN touch /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
# RUN cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys

# ARG CUDA12_GENCODE='-gencode=arch=compute_90,code=sm_90'
# ARG CUDA12_PTX='-gencode=arch=compute_90,code=compute_90'

# WORKDIR /third_party
# RUN git clone https://github.com/NVIDIA/nccl-tests.git nccl-tests-mpi
# WORKDIR nccl-tests-mpi
# RUN git fetch --all --tags
# RUN make MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi CUDA_HOME=/usr/local/cuda NCCL_HOME=/third_party/nccl-netsupport/build NVCC_GENCODE="$CUDA12_GENCODE $CUDA12_PTX" -j

# WORKDIR /scripts


# COPY src/nccl_healthcheck/config.proto /scripts/
# RUN pip install grpcio-tools
# RUN python3 -m grpc_tools.protoc -I /scripts/ --python_out=. --pyi_out=. --grpc_python_out=. --experimental_editions /scripts/config.proto

# COPY src/nccl_healthcheck/nccl_startup.py /scripts/
# COPY src/nccl_healthcheck/config.py /scripts/
# COPY src/checker_common.py /scripts/
# COPY src/metrics.py /scripts/
# COPY src/nccl_healthcheck/a3/ a3/
# COPY src/nccl_healthcheck/a3plus/ a3plus/
# COPY src/nccl_healthcheck/run-nccl-combined-plugins.sh .
# RUN chmod +x /scripts/run-nccl-combined-plugins.sh
# ENV PYTHONUNBUFFERED=1

# # COPY src/nccl_healthcheck/run_startup.sh /scripts/run_startup.sh
# # RUN chmod +x /scripts/run_startup.sh
# # ENTRYPOINT ["/scripts/run_startup.sh"]
# ENTRYPOINT ["python3", "/scripts/nccl_startup.py"]
# # ENTRYPOINT ["/bin/bash", "-c", "export CUDA12_GENCODE='-gencode=arch=compute_90,code=sm_90' && export CUDA12_PTX='-gencode=arch=compute_90,code=compute_90' && export PATH='/usr/local/cuda/bin:${PATH}' && export LD_LIBRARY_PATH='/usr/local/cuda/lib64:${LD_LIBRARY_PATH}' && cd /tmp && git clone https://github.com/NVIDIA/nccl-tests.git nccl-tests-mpi && cd nccl-tests-mpi && git fetch --all --tags && make MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi CUDA_HOME=/usr/local/cuda NCCL_HOME=/tmp/nccl-netsupport/build NVCC_GENCODE=\"$CUDA12_GENCODE $CUDA12_PTX\" -j && python3 /scripts/nccl_startup.py"]

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

# FROM us-docker.pkg.dev/gce-ai-infra/gpudirect-tcpx/nccl-plugin-gpudirecttcpx-nightly-cuda12.0:2024_03_04
# FROM us-docker.pkg.dev/kernel-net-team/gpudirect-tcpx/nccl-plugin-gpudirecttcpx-pre-test-cuda12.4:elm_sctp_mitigation
# FROM us-docker.pkg.dev/gce-ai-infra/gpudirect-tcpxo/nccl-plugin-gpudirecttcpx-dev:v1.0.4
# FROM us-docker.pkg.dev/gce-ai-infra/health-check/nccl-plugin-gpudirecttcpx-pre-test-cuda12.4:elm_sctp_mitigation
FROM us-docker.pkg.dev/gce-ai-infra/health-check/nccl-plugin-gpudirecttcpx-dev:v1.0.4

WORKDIR /scripts
RUN apt-get update && apt-get install -y openssh-server python3.9 ca-certificates curl python3-pip &&\
  update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 &&\
  mkdir /var/run/sshd && chmod 0755 /var/run/sshd &&\
  ssh-keygen -t rsa -f /root/.ssh/google_compute_engine -b 2048 -P "" &&\
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
# COPY src/metrics.py /scripts/
# COPY src/nccl_healthcheck/a3/ a3/
COPY src/nccl_healthcheck/a3plus/ a3plus/
COPY src/nccl_healthcheck/run-nccl-combined-plugins.sh .
RUN chmod +x /scripts/run-nccl-combined-plugins.sh
ENV PYTHONUNBUFFERED=1

# RUN apt-get install -y dnsutils

ENTRYPOINT ["python3", "/scripts/nccl_startup.py"]