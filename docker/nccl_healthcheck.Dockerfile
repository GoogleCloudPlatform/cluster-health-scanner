FROM nvcr.io/nvidia/pytorch:24.04-py3

RUN mkdir /var/run/sshd && chmod 0755 /var/run/sshd &&\
  ssh-keygen -t rsa -f /root/.ssh/google_compute_engine -b 2048 -P "" &&\
  cp /root/.ssh/google_compute_engine.pub /root/.ssh/authorized_keys &&\
  chmod 644 /root/.ssh/authorized_keys &&\
  chmod 644 /root/.ssh/google_compute_engine.pub

WORKDIR /nccl
RUN git clone https://github.com/NVIDIA/nccl-tests.git
WORKDIR nccl-tests
RUN git fetch --all --tags
RUN MPI=1 CC=mpicc CXX=mpicxx make -j

WORKDIR /scripts
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
ENV PYTHONUNBUFFERED=1


