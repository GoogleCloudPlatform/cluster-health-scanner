#!/bin/bash
set -x
#### CUSTOMER CUSTOMIZABLE SECTION ###
######################################
######### START OF SECTION ###########

NODE_COUNT=2 # NOTE: put the number of A3 nodes to provision
REGION="us-east4" # NOTE: put your region where A3 quota was granted
ZONE="us-east4-c" # NOTE: put your zone where A3 quota was granted

PROJECT="supercomputer-testing" # NOTE: put your project name
PROJECT_NUMBER="455207029971" # NOTE: put your project number
YOUR_EMAIL="${USER}@google.com" # NOTE: put your email
USE_GSC=false

# Please modify the below namings to fit your needs
PREFIX="yufanfei-testing1"
CLUSTER_NAME="yufanfei-testing1"
NODE_POOL_NAME="np"

######################################
######################################
######### END OF SECTION #############

GKE_VERSION=1.27.7-gke.1121000
GKE_NODE_VERSION=1.27.7-gke.1121000
USE_CUSTOM_COS=false
COS_VERSION="cos-105-17412-226-34"
COS_PROJECT="cos-cloud"
ACCELERATOR_ARG="type=nvidia-h100-80gb,count=8"
MACHINE_TYPE=a3-highgpu-8g

# Choose a name for the service account
SA_SHORT_NAME=a3-svc-acct
SA_FULL_NAME="${SA_SHORT_NAME?}@${PROJECT}.iam.gserviceaccount.com"
### The below SA creation section is a one-time setup. You only need to do it once per project
# Create a service account
gcloud iam service-accounts create "${SA_SHORT_NAME?}" --project "${PROJECT}"
# Allow you to impersonate the service account
gcloud iam service-accounts add-iam-policy-binding --project "${PROJECT}" "${SA_FULL_NAME?}" --member "user:${YOUR_EMAIL?}" --role roles/iam.serviceAccountTokenCreator
# Grant roles needed for calling the GKE API. Note that
# roles/iam.serviceAccountActor is required to create node instances with the # default compute service account
gcloud projects add-iam-policy-binding "${PROJECT}" --member "serviceAccount:${SA_FULL_NAME?}" --role roles/container.clusterAdmin
gcloud projects add-iam-policy-binding "${PROJECT}" --member "serviceAccount:${SA_FULL_NAME?}" --role roles/iam.serviceAccountActor
### End of SA creation section
# We need to create four additional VPCs and subnets
for N in $(seq 1 4); do
  gcloud compute --project=${PROJECT} \
    networks create \
    ${PREFIX?}-net-"$N" \
    --subnet-mode=custom \
    --mtu=8244
# You should explicitly choose the subnets that you want to use below.
# A /19 has space for 8192 IPs and this might be sufficient for your use-case,
# but you should explicitly choose the range that fits your needs.
  gcloud compute --project=${PROJECT} \
    networks subnets create \
    "${PREFIX?}-sub-$N" \
    --network="${PREFIX?}-net-$N" \
    --region=${REGION?} \
    --range="10.$N.0.0/19"
  gcloud compute --project=${PROJECT} networks subnets update "${PREFIX?}-sub-$N" \
    --region "${REGION}" \
    --private-ipv6-google-access-type=enable-outbound-vm-access
  gcloud compute --project=${PROJECT} \
    firewall-rules create \
    "${PREFIX?}-internal-$N" \
    --network="${PREFIX?}-net-$N" \
    --action=ALLOW \
    --rules=tcp:0-65535,udp:0-65535,icmp \
    --source-ranges=10.0.0.0/8
done

# Create a cluster with multi-networking enabled
gcloud beta container clusters create ${CLUSTER_NAME} \
--no-enable-autoupgrade \
--no-enable-shielded-nodes \
--enable-dataplane-v2 \
--region ${REGION} \
--enable-ip-alias \
--enable-multi-networking \
--num-nodes 15  \
--cluster-version ${GKE_VERSION}  \
--project ${PROJECT} \
--impersonate-service-account "${SA_FULL_NAME?}"

# Create a resource policy to be SB aligned
gcloud beta compute resource-policies create group-placement --project "${PROJECT}" "${PREFIX}-gp-1" \
--collocation=collocated \
--region=${REGION} \
--max-distance=2

EXTRA_NP_ARGS=""
if [ "${USE_GSC}" = "true" ]; then
 EXTRA_NP_ARGS="${EXTRA_NP_ARGS} --host-maintenance-interval=PERIODIC --reservation-affinity=specific --reservation=${RESERVATION_NAME}"
fi
IMAGE_TYPE="COS_CONTAINERD"
if [ "${USE_CUSTOM_COS}" = "true" ]; then
  IMAGE_TYPE="CUSTOM_CONTAINERD"
  EXTRA_NP_ARGS="${EXTRA_NP_ARGS} --image=${COS_VERSION} --image-project=${COS_PROJECT} --node-labels=gpu-custom-cos-image.gke.io=true"
fi

gcloud beta container node-pools create "${NODE_POOL_NAME}-1" \
  --node-version ${GKE_NODE_VERSION} \
  --cluster ${CLUSTER_NAME}  \
  --region ${REGION} \
  --image-type ${IMAGE_TYPE} \
  --node-locations ${ZONE} \
  --project ${PROJECT} \
  --accelerator ${ACCELERATOR_ARG} \
  --machine-type ${MACHINE_TYPE} \
  --num-nodes ${NODE_COUNT}  \
  --ephemeral-storage-local-ssd count=16 \
  --scopes "https://www.googleapis.com/auth/cloud-platform" \
  --additional-node-network network=${PREFIX?}-net-1,subnetwork=${PREFIX?}-sub-1 --additional-node-network network=${PREFIX?}-net-2,subnetwork=${PREFIX?}-sub-2 --additional-node-network network=${PREFIX?}-net-3,subnetwork=${PREFIX?}-sub-3 --additional-node-network network=${PREFIX?}-net-4,subnetwork=${PREFIX?}-sub-4 \
  --enable-gvnic \
  --impersonate-service-account "${SA_FULL_NAME?}" \
  --max-pods-per-node=32 \
  --placement-policy "${PREFIX}-gp-1" \
  --no-enable-autoupgrade \
  --no-enable-autorepair \
  --host-maintenance-interval=PERIODIC \
  ${EXTRA_NP_ARGS}

# Repeat the above 2 steps (resource policy creation + node pool creation) as many times as the # of nodepools you wish to create.
# Connect to the cluster
gcloud container clusters get-credentials $CLUSTER_NAME --region $REGION --project $PROJECT
# The below 3 steps need to be done once per cluster creation. You don't have to run it again
# if you're just adding new node pools to the existing cluster.
# Configuring eth1-4 with NetDevice mode
i=1
while read -r net subnet; do
  if [ -z "$net" ] || [ -z "$subnet" ]; then
    echo "Error: net or subnet is empty. Exiting."
    exit 1
  fi
  eval "net$i=\"$net\""
  eval "subnet$i=\"$subnet\""
  i=$((i + 1))
done < <(gcloud container clusters describe ${CLUSTER_NAME} --region=${REGION} --project=${PROJECT} --format="json" | jq -r 'first(.nodePools[] | select(.config.machineType=="a3-highgpu-8g")) | .networkConfig.additionalNodeNetworkConfigs[] | "\(.network) \(.subnetwork)"')

sed \
  -e "s/\$net1/$net1/g" -e "s/\$subnet1/$subnet1/g" \
  -e "s/\$net2/$net2/g" -e "s/\$subnet2/$subnet2/g" \
  -e "s/\$net3/$net3/g" -e "s/\$subnet3/$subnet3/g" \
  -e "s/\$net4/$net4/g" -e "s/\$subnet4/$subnet4/g" \
  netdevice_mode.yaml | kubectl apply -f -

if [ "${USE_CUSTOM_COS}" = "true" ]; then
  kubectl apply -f fixup_daemon_set.yaml
fi

# Install NVIDIA drivers for the cos image
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded-latest.yaml

# Verify the Cilium fix - the output should be:
#
# default GNPParamsReady ParamsReady
# default NetworkReady Ready
# vpc1 GNPParamsReady ParamsReady
# vpc1 NetworkReady Ready
# vpc2 GNPParamsReady ParamsReady
# vpc2 NetworkReady Ready
# vpc3 GNPParamsReady ParamsReady
# vpc3 NetworkReady Ready
# vpc4 GNPParamsReady ParamsReady
# vpc4 NetworkReady Ready
kubectl get networks -o json | jq -r '.items[] | .metadata.name as $name | .status.conditions[]? | "\($name) \(.reason) \(.type)"'
