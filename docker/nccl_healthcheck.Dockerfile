FROM us-docker.pkg.dev/gce-ai-infra/health-check/tcpxo_debian:v107_v5_w_stats_flag
#WORKDIR /scripts

#RUN apt-get update &&\
#    apt-get install -y openssh-server ca-certificates curl net-tools gnupg gdb python3-pip

# Install dependencies
#RUN apt-get update && apt-get install -y \
#    build-essential \
#    zlib1g-dev \
#    libncurses5-dev \
#    libgdbm-dev \
#    libnss3-dev \
#    libssl-dev \
#    libreadline-dev \
#    libffi-dev \
#    wget

# Download and install Python 3.10
#RUN wget https://www.python.org/ftp/python/3.10.10/Python-3.10.10.tgz && \
#    tar -xf Python-3.10.10.tgz && \
#    cd Python-3.10.10 && \
#    ./configure --enable-optimizations && \
#    make -j$(nproc) && \
#    make altinstall && \
#    cd .. && \
#    rm -rf Python-3.10.10*

# Install pip
#RUN python3.10 -m ensurepip --upgrade

#RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.10 1
#RUN update-alternatives --set python3 /usr/local/bin/python3.10
#RUN python3 --version

#RUN mkdir /var/run/sshd && chmod 0755 /var/run/sshd && \
#    ssh-keygen -t rsa -f /root/.ssh/google_compute_engine -b 2048 -P "" && \
#    cp /root/.ssh/google_compute_engine.pub /root/.ssh/authorized_keys && \
#    chmod 644 /root/.ssh/authorized_keys && \
#    chmod 644 /root/.ssh/google_compute_engine.pub

#RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" &&\
#   chmod +x kubectl

#COPY src/nccl_healthcheck/config.proto /scripts/
#RUN /usr/local/bin/python3.10 -m pip install --upgrade pip
#RUN /usr/local/bin/python3.10 -m pip install grpcio-tools
#RUN /usr/local/bin/python3.10 -m pip install kubernetes
#RUN python3 -m grpc_tools.protoc -I /scripts/ --python_out=. --pyi_out=. --grpc_python_out=. --experimental_editions /scripts/config.proto

#COPY src/nccl_healthcheck/nccl_startup.py /scripts/
#COPY src/nccl_healthcheck/config.py /scripts/
#COPY src/nccl_healthcheck/run-nccl-combined-plugins.sh .
#COPY src/checker_common.py /scripts/
#RUN chmod +x /scripts/run-nccl-combined-plugins.sh
#ENV PYTHONUNBUFFERED=1

#RUN apt-get install -y dnsutils

#ENTRYPOINT ["python3", "/scripts/nccl_startup.py"]



RUN apt-get update && apt-get install -y openssh-server python3.9 ca-certificates curl python3-pip munge libmunge-dev wget \
    build-essential \
    libtool \
    pkg-config \
    autoconf \
    automake \
    libslurm-dev \
    libpmi2-0-dev \
#    libpmi2-dev \
#    pmix \
    libpmix-dev &&\
  update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 &&\
  mkdir /var/run/sshd && chmod 0755 /var/run/sshd &&\
  ssh-keygen -t rsa -f /root/.ssh/google_compute_engine -b 2048 -P "" &&\
  cp /root/.ssh/google_compute_engine.pub /root/.ssh/authorized_keys &&\
  chmod 644 /root/.ssh/authorized_keys &&\
  chmod 644 /root/.ssh/google_compute_engine.pub

RUN find /usr/lib -name "libpmi*" && find /usr/lib -name "libpmix*"

# Download Slurm source
WORKDIR /tmp
RUN wget https://download.schedmd.com/slurm/slurm-24.05.3.tar.bz2 \
    && tar -xvf slurm-24.05.3.tar.bz2

# Configure and build Slurm
WORKDIR /tmp/slurm-24.05.3
RUN ./configure --with-pmix \
    && make -j$(nproc) \
    && make install


# Download and build OpenMPI with Slurm PMI support

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
#RUN ./configure \
#    --prefix=/usr/local \
#    --with-pmi=/usr/lib/x86_64-linux-gnu \
#    --with-pmi-libdir=/usr/lib/x86_64-linux-gnu \
#    --with-slurm \
#    --enable-mpirun-prefix-by-default \
#    && make -j$(nproc) \
#    && make install \
#    && ldconfig
# Optional: Install Python bindings
#RUN pip3 install PySlurm

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

# Create a minimal slurm.conf file during image build
#RUN echo "# Slurm configuration file" > /var/spool/slurmd/conf-cache/slurm.conf && \
#    echo "ClusterName=a3m123" >> /var/spool/slurmd/conf-cache/slurm.conf && \
#    echo "SlurmctldHost=localhost" >> /var/spool/slurmd/conf-cache/slurm.conf && \
#    echo "SlurmUser=slurm" >> /var/spool/slurmd/conf-cache/slurm.conf && \
#    echo "SlurmdUser=root" >> /var/spool/slurmd/conf-cache/slurm.conf && \
#    echo "SlurmctldPort=6817" >> /var/spool/slurmd/conf-cache/slurm.conf && \
#    echo "SlurmdPort=6818" >> /var/spool/slurmd/conf-cache/slurm.conf && \
#    echo "StateSaveLocation=/var/spool/slurmd" >> /var/spool/slurmd/conf-cache/slurm.conf && \
#    echo "SlurmdSpoolDir=/var/spool/slurmd" >> /var/spool/slurmd/conf-cache/slurm.conf

COPY docker/slurm.conf /var/spool/slurmd/conf-cache
# Set OpenMPI environment variables
ENV PATH="/usr/local/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"

WORKDIR /scripts

RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && chmod +x kubectl

COPY src/nccl_healthcheck/config.proto /scripts/
RUN pip install grpcio-tools 
RUN pip install protobuf
RUN python3 -m grpc_tools.protoc -I /scripts/ --python_out=. --pyi_out=. --grpc_python_out=. --experimental_editions /scripts/config.proto

COPY src/nccl_healthcheck/nccl_startup.py /scripts/
COPY src/nccl_healthcheck/config.py /scripts/
COPY src/checker_common.py /scripts/
COPY src/metrics.py /scripts/
COPY src/nccl_healthcheck/a3/ a3/
COPY src/nccl_healthcheck/a3plus/ a3plus/
COPY src/nccl_healthcheck/run-nccl-combined-plugins.sh .
RUN chmod +x /scripts/run-nccl-combined-plugins.sh
ENV PYTHONUNBUFFERED=1

#RUN apt-get install -y dnsutils
