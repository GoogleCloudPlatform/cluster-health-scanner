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

FROM us-docker.pkg.dev/gce-ai-infra/health-check/tinymax-healthcheck:v1

WORKDIR /scripts

RUN apt-get update &&\
    apt-get install -y curl python3 unzip sudo &&\
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" &&\
    chmod +x kubectl

# Install gcloud sdk
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz
RUN mkdir -p /usr/local/gcloud \
  && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && /usr/local/gcloud/google-cloud-sdk/install.sh
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

COPY src/health_runner/health_results.proto /scripts/
COPY src/nccl_healthcheck/config.proto /scripts/
COPY src/common.proto /scripts/
RUN pip install grpcio-tools
RUN pip install kubernetes
RUN python3 -m grpc_tools.protoc -I /scripts/ --python_out=. --pyi_out=. --grpc_python_out=. --experimental_editions /scripts/config.proto
RUN python3 -m grpc_tools.protoc -I /scripts/ --python_out=. --pyi_out=. --grpc_python_out=. --experimental_editions /scripts/health_results.proto
RUN python3 -m grpc_tools.protoc -I /scripts/ --python_out=. --pyi_out=. --grpc_python_out=. --experimental_editions /scripts/common.proto

COPY src/tinymax_healthcheck/run-inside-container-enhance.sh /scripts/
RUN chmod +x /scripts/run-inside-container-enhance.sh
COPY src/tinymax_healthcheck/tinymax_runner.py /scripts/
COPY src/checker_common.py /scripts/
CMD ["python3", "/scripts/tinymax_runner.py"]