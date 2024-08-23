from nvidia/dcgm:3.3.5-1-ubuntu22.04
RUN apt-get update && apt-get install -y ca-certificates curl python3
workdir /app
RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && chmod +x kubectl
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz
RUN mkdir -p /usr/local/gcloud \
  && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && /usr/local/gcloud/google-cloud-sdk/install.sh
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin
COPY src/gpu_healthcheck/gpu_healthcheck.py .
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python3", "/app/gpu_healthcheck.py"]