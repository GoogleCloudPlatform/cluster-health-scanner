FROM gcr.io/cloud-builders/kubectl

# reinstall python3
RUN apt-get -y update && \
    apt-get dist-upgrade -y && \
    apt-get install -y python3-pip && \
    python3 -m pip install kubernetes

COPY entrypoint.sh .
RUN chmod +x entrypoint.sh
ENTRYPOINT ["sh", "./entrypoint.sh"]