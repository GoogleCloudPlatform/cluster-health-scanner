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

FROM debian:bullseye-slim
RUN mkdir -p /fluent-bit/etc /fluent-bit/bin

# install dependencies
RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install -y curl bash gpg
RUN apt-get install -y procps

RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && chmod +x kubectl && mv kubectl /usr/local/bin/kubectl

# Install fluent-bit
RUN curl https://raw.githubusercontent.com/fluent/fluent-bit/master/install.sh | sh

# Copy fluent-bit configuration
COPY src/fluent-bit.conf /fluent-bit/etc/fluent-bit.conf

COPY src/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]