# Build Stage
FROM ubuntu:latest

ENV PYTHONUNBUFFERED=1
WORKDIR /scripts

# Install build dependencies and utilities
RUN apt-get update &&\
    apt-get install -y git make gcc g++ util-linux software-properties-common openssh-server ca-certificates curl jq &&\
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" &&\
    chmod +x kubectl


# Clone and build Neper
RUN git clone https://github.com/google/neper.git && \
    cd /scripts/neper && \
    make tcp_stream && \
    cp /scripts/neper/tcp_stream /scripts/ && rm -rf /scripts/neper/* && chmod +x /scripts/tcp_stream

RUN mkdir /var/run/sshd && chmod 0755 /var/run/sshd &&\
  ssh-keygen -t rsa -f /root/.ssh/google_compute_engine -b 2048 -P "" &&\
  cp /root/.ssh/google_compute_engine.pub /root/.ssh/authorized_keys &&\
  chmod 644 /root/.ssh/authorized_keys &&\
  chmod 644 /root/.ssh/google_compute_engine.pub

COPY src/neper_healthcheck/neper_runner.py /scripts/

CMD ["python3", "/scripts/neper_runner.py"]