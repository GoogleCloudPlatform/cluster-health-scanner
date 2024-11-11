#!/bin/bash
#
#SBATCH --partition=a3mega
#SBATCH --exclusive
#SBATCH --gpus-per-node 8
#SBATCH --nodes 1
#SBATCH --gpus=8 
#SBATCH --ntasks=1 

export NODE_NAME=$HOSTNAME
echo "Running on node: $NODE_NAME"
export HEALTH_VALIDITY_HOURS=24
export DRY_RUN=true
export R_LEVEL=1

export USER_ID=0 #"$(id -u)"
export GROUP_ID=0 #"$(id -g)"
#: "${VERSION:=slurm}"
#srun docker build -t gpu:"${VERSION}" .
#srun rm -f gpu+"${VERSION}".sqsh
#srun enroot import dockerd://gpu:"${VERSION}"

#CONTAINER_MOUNTS="/etc/slurm/:/etc/slurm/"
#CONTAINER_MOUNTS="/etc/slurm/:/var/spool/slurmd/conf-cache/"
CONTAINER_MOUNTS="/usr/local/bin,/usr/local/lib,/var/spool/slurmd,/var/spool/slurm,/var/run/slurm,/etc/passwd:/etc/passwd,/usr/lib64,/var/run/munge"
# Launch the script
#sudo srun --container-image=./gpu+slurm.sqsh --container-mounts="${CONTAINER_MOUNTS}" --export=HEALTH_VALIDITY_HOURS=24,DRY_RUN=true,R_LEVEL=1,NODE_NAME=$HOSTNAME bash -c "id -u;python3 /app/gpu_healthcheck.py"
#sudo srun --container-image=./gpu+slurm.sqsh --container-mounts="${CONTAINER_MOUNTS}" --container-env-file=gpu_healthcheck.list bash -c "id -u;python3 /app/gpu_healthcheck.py"
#sudo srun --container-image=./gpu+slurm.sqsh --container-mounts="${CONTAINER_MOUNTS}" bash -c "id -u; sudo scontrol update nodename=a3mega-a3meganodeset-0 comment=nina-was-here"
#srun docker run --rm gpu:slurm "scontrol update nodename-a3mega-a3meganodeset-0 comment=nina-was-here"
sudo srun --container-image=./gpu+slurm.sqsh --container-mounts="${CONTAINER_MOUNTS}" --export=HEALTH_VALIDITY_HOURS=24,DRY_RUN=true,R_LEVEL=1,NODE_NAME=$HOSTNAME  --gpus=8 bash -c "/usr/bin/nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits"


