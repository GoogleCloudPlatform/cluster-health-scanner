FROM us-docker.pkg.dev/gce-ai-infra/health-check/tcpxo_debian:v107_v5_w_stats_flag

RUN apt-get update && apt-get install -y openssh-server python3.9 ca-certificates curl python3-pip munge libmunge-dev wget \
    build-essential \
    libtool \
    pkg-config \
    autoconf \
    automake \
    libslurm-dev \
    libpmix-dev &&\
  update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 &&\
  mkdir /var/run/sshd && chmod 0755 /var/run/sshd &&\
  ssh-keygen -t rsa -f /root/.ssh/google_compute_engine -b 2048 -P "" &&\
  cp /root/.ssh/google_compute_engine.pub /root/.ssh/authorized_keys &&\
  chmod 644 /root/.ssh/authorized_keys &&\
  chmod 644 /root/.ssh/google_compute_engine.pub

# Download NCCL dev package
#RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libnccl-dev_2.18.5-1+cuda12.2_amd64.deb

# Download NCCL runtime package
#RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libnccl2_2.18.5-1+cuda12.2_amd64.deb

# Install packages
#RUN dpkg -i libnccl2_2.18.5-1+cuda12.2_amd64.deb libnccl-dev_2.18.5-1+cuda12.2_amd64.deb

# If dependencies are missing, run
#RUN apt-get install -f
#RUN apt-get update && apt-get install -y wget gnupg
#RUN curl -O https://developer.nvidia.com/downloads/compute/machine-learning/nccl/secure/2.23.4/ubuntu2004/x86_64/nccl-local-repo-ubuntu2004-2.23.4-cuda12.4_1.0-1_amd64.deb
#RUN dpkg -i nccl-local-repo-ubuntu2004-2.23.4-cuda12.4_1.0-1_amd64.deb
#RUN apt-get update && apt-get install -y libnccl2 libnccl-dev
RUN apt-get update && apt-get install -y wget curl gnupg
RUN curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN apt-get update && apt-get install -y libnccl2 libnccl-dev

RUN find /usr/lib -name "libpmi*" && echo "PMI libraries found in /usr/lib" || echo "No PMI libraries found in /usr/lib"

# Download Slurm source
WORKDIR /tmp
RUN wget https://download.schedmd.com/slurm/slurm-24.05.3.tar.bz2 \
    && tar -xvf slurm-24.05.3.tar.bz2

# Configure and build Slurm
WORKDIR /tmp/slurm-24.05.3
RUN ./configure --with-pmix \
    && make -j$(nproc) \
    && make install

RUN find /usr/lib /usr/local/lib -name "libpmi2*" && echo "PMI2 libraries found" || echo "No PMI2 libraries found"

# Download and build OpenMPI with Slurm PMI2 support

ENV OPENMPI_VERSION=4.1.5
WORKDIR /tmp
RUN wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-${OPENMPI_VERSION}.tar.bz2 \
    && tar -xjf openmpi-${OPENMPI_VERSION}.tar.bz2

WORKDIR /tmp/openmpi-${OPENMPI_VERSION}
RUN ./configure \
    --prefix=/usr/local \
    --with-pmix \
    --with-slurm \
    --enable-mpirun-prefix-by-default \
    && make -j$(nproc) \
    && make install \
    && ldconfig

# Clean up
RUN rm -rf /tmp/slurm-24.05.3 /tmp/slurm-24.05.3.tar.bz2 \
    /tmp/openmpi-${OPENMPI_VERSION}* \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create slurm user and directories
RUN groupadd -r slurm \
    && useradd -r -g slurm slurm \
    && mkdir -p /etc/slurm /var/spool/slurmd /var/log/slurmd

RUN mkdir -p /var/spool/slurmd/conf-cache

COPY docker/slurm.conf /var/spool/slurmd/conf-cache
# Set OpenMPI environment variables
ENV PATH="/usr/local/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"

WORKDIR /nccl
RUN git clone https://github.com/NVIDIA/nccl-tests.git
WORKDIR nccl-tests
RUN git fetch --all --tags
RUN MPI=1 CC=mpicc CXX=mpicxx make -j


WORKDIR /scripts

RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && chmod +x kubectl

COPY src/nccl_healthcheck/config.proto /scripts/
RUN pip install grpcio-tools
RUN pip install protobuf
RUN python3 -m grpc_tools.protoc -I /scripts/ --python_out=. --pyi_out=. --grpc_python_out=. --experimental_editions /scripts/config.proto

COPY src/nccl_healthcheck/nccl_startup.py /scripts/
COPY src/nccl_healthcheck/config.py /scripts/
COPY src/nccl_healthcheck/a3plus/mpi_launcher.py /scripts/
COPY src/checker_common.py /scripts/
COPY src/metrics.py /scripts/
COPY src/nccl_healthcheck/a3/ a3/
COPY src/nccl_healthcheck/a3plus/ a3plus/
COPY src/nccl_healthcheck/run-nccl-combined-plugins.sh .
RUN chmod +x /scripts/run-nccl-combined-plugins.sh
ENV PYTHONUNBUFFERED=1

