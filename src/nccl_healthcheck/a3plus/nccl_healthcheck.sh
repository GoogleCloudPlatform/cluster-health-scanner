#!/bin/bash
#
#SBATCH --partition=a3mega
#SBATCH --exclusive
#SBATCH --gpus-per-node 8
#SBATCH --nodes 2
#SBATCH --ntasks-per-node=8

export HEALTH_VALIDITY_HOURS=24
export DRY_RUN=true
export NHOSTS=2
export nr=8
export BANDWIDTH_THRESHOLD=90
export START_MESSAGE_SIZE="2G"
export END_MESSAGE_SIZE="8G"
export USE_FASTRAK="true"
export NCCL_FASTRAK_USE_SNAP="1"
export NCCL_FASTRAK_ENABLE_CONTROL_CHANNEL="0"
export NCCL_FASTRAK_NUM_FLOWS="2"

export NODE_NAME=$HOSTNAME
export NHOSTS=2
CONTAINER_IMAGE=./nccl+slurm.sqsh
#NCCL_LIB_DIR="/usr/local/nvidia/lib64" source /var/lib/tcpxo/lib64/nccl-env-profile.sh
export NCCL_FASTRAK_CTRL_DEV=enp0s12
export NCCL_FASTRAK_IFNAME=enp6s0,enp7s0,enp13s0,enp14s0,enp134s0,enp135s0,enp141s0,enp142s0
export NCCL_SOCKET_IFNAME=enp0s12
export NCCL_FASTRAK_LLCM_DEVICE_DIRECTORY=/dev/aperture_devices

#HOST_VARS=$(sed 's/ \{1,\}/,/g' <<<"${!NCCL*}")

echo $HOST_VARS

CONTAINER_MOUNTS="/var/tmp:/var/tmp"


#CONTAINER_MOUNTS=${CONTAINER_MOUNTS},"/usr/sbin,/var/run/slurm,/etc/passwd:/etc/passwd,/var/run/munge,/tmp,/root/.ssh:/root/.ssh,/tmp:/tmp,/etc/ssh:/etc/ssh,/opt/apps:/opt/apps"
#CONTAINER_MOUNTS=${CONTAINER_MOUNTS},"/usr/sbin,/var/run/slurm,/etc/passwd:/etc/passwd,/var/run/munge,/tmp,/root/.ssh:/root/.ssh,/tmp:/tmp,/etc/ssh:/etc/ssh,/opt/apps:/opt/apps,/var/tmp:/var/tmp"

#sudo srun  \
#        --container-image=./nccl+slurm.sqsh \
#        --export=HEALTH_VALIDITY_HOURS=24,DRY_RUN=true,NHOSTS=2,nr=8,JOB_NAME="neper-healthcheck-${SLURM_JOB_ID}",SERVICE_NAME="neper-headless-svc-${SLURM_JOB_ID}",GOOD_THROUGHPUT="50000000000",NODE_IP=$NODE_IP,POD_NAME=$POD_NAME,NODES=$SLURM_JOB_NODELIST,NODE_NAME=$HOSTNAME,JOB_COMPLETION_INDEX=0,BANDWIDTH_THRESHOLD="90",START_MESSAGE_SIZE="2G",END_MESSAGE_SIZE="8G",INSTANCE_TYPE="a3-megagpu-8g",ITERATIONS="1" --gpus=8 --nodes 2 --exclusive \
#	--mpi=pmi2 \
#        --container-mounts="${CONTAINER_MOUNTS}" \
#        sh -c "python3 /scripts/nccl_startup.py"
	#
#srun  \
#        --container-image=./nccl+slurm.sqsh \
#        --export=NHOSTS=2 \
#        --gpus=8 --nodes 2 --exclusive \
#        --mpi=pmix \
#        --container-mounts="${CONTAINER_MOUNTS}" \
#	mpirun -np 16 hostname        
#mpirun -np 16 --allow-run-as-root sh -c "hostname && python3 -c 'from mpi4py import MPI; comm = MPI.COMM_WORLD; print(f\"Rank {comm.Get_rank()} out of {comm.Get_size()} on {MPI.Get_processor_name()}\")"
#
#sudo srun --container-image=./nccl+slurm.sqsh \
#    --nodes 2 --exclusive \
#    --container-mounts="${CONTAINER_MOUNTS}" \
#    sh -c "ip addr; hostname -f"
#
# srun  \
#        --container-image=./nccl+slurm.sqsh \
#        --gpus=8 --nodes 2 --exclusive \
#        --mpi=pmix \
#        --container-mounts="${CONTAINER_MOUNTS},/var/run/munge:/var/run/munge" \
#        mpirun -v -mca oob_base_verbose 100 -mca plm_base_verbose 100 \
#        sh -c "hostname; ip addr"
  #
# srun  \
#        --container-image=./nccl+slurm.sqsh \
#        --nodes 2 --exclusive \
#        --container-mounts="${CONTAINER_MOUNTS}" \
#        sh -c "ompi_info"
#
#srun  \
#        --container-image=./nccl+slurm.sqsh \
#        --gpus=8 --nodes 2 --exclusive \
#        --mpi=pmix \
#        --container-mounts="${CONTAINER_MOUNTS},/var/run/munge:/var/run/munge" \
#        sh -c "
#            which munge || (apt-get update && apt-get install -y munge);
#            ls -l /var/run/munge;
##            munge -n;
#            munge -n | unmunge || true
#        "
#        sh -c "
#            ls -l /var/run/munge;
#            munged --version;
#            systemctl status munge;
#            munge -n;
#            unmunge < <(munge -n)
#        "
#
#srun \
#    --container-image=./nccl+slurm.sqsh \
#    --gpus=8  --gres=gpu:8  --nodes=2 --exclusive \
#    --mpi=pmix \
#    --container-mounts="${CONTAINER_MOUNTS},/var/run/munge:/var/run/munge" \
#    mpirun -v -mca oob_base_verbose 100 -mca plm_base_verbose 100 \
#    sh -c "hostname; ip addr"
#srun \
#    --container-image=./nccl+slurm.sqsh \
#    --gpus=8 \
#    --gres=gpu \
#    --nodes=2 \
#    --exclusive \
#    --mpi=pmix \
#    --container-mounts="${CONTAINER_MOUNTS},/var/run/munge:/var/run/munge" \
#    sh -c "sinfo -O NodeHost,Gres"
#
#srun --gres=gpu:8 --nodes=2 --exclusive hostname
#srun \
#    --container-image=./nccl+slurm.sqsh \
#    --gres=gpu:8 \
#    --nodes=2 \
#    --exclusive \
#    --mpi=pmix \
#    --container-mounts="${CONTAINER_MOUNTS},/var/run/munge:/var/run/munge" \
#    mpirun -v -mca oob_base_verbose 100 -mca plm_base_verbose 100 \
#    sh -c "hostname; ip addr"
# Verify network connectivity between nodes
#srun -N2 /bin/ip addr

NCCL_LIB_DIR="/var/lib/tcpxo/lib64" source /var/lib/tcpxo/lib64/nccl-env-profile.sh
export NCCL_FASTRAK_CTRL_DEV=enp0s12
export NCCL_FASTRAK_IFNAME=enp6s0,enp7s0,enp13s0,enp14s0,enp134s0,enp135s0,enp141s0,enp142s0
export NCCL_SOCKET_IFNAME=enp0s12
export NCCL_FASTRAK_LLCM_DEVICE_DIRECTORY=/dev/aperture_devices

# Here we grab all the environment variables that need to be
# passed down into the container. Slurm would otherwise only pass these env vars
# to the job environment on the host.
# shellcheck disable=SC2001
HOST_VARS=$(sed 's/ \{1,\}/,/g' <<<"${!NCCL*}")

# Simplify the command to isolate the issue
#srun \
#    --container-image=./nccl+slurm.sqsh \
#    --ntasks-per-node=8 \
#    --nodes=2 \
#    --mpi=pmi2 \
#    --container-env="${HOST_VARS}" \
#    --container-mounts="/var/tmp:/var/tmp,/var/lib/tcpxo/lib64/" \
#    sh -c "
#  mpirun --version;mpiexec --version; mpicc --version; slurmd -V;
#  export LD_LIBRARY_PATH=/var/lib/tcpxo/lib64:/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH;
#  /nccl/nccl-tests/build/all_gather_perf -b 8M -e 8G -f 2 -g 1 -w 5 --iters 200 -c 0;"
    #sh -c "python3 /scripts/nccl_startup.py"
    #sh -c "python3 mpi_launcher.py all_gather_perf  2G 8G 20 8 2"
    #sh -c "mpirun -np 2 \
    #--host a3m123-a3meganodeset-3:1,a3m123-a3meganodeset-29:1 \
    #--allow-run-as-root \
    #--debug-daemons \
    #-mca plm slurm \
    #-mca btl tcp,self \
    #-mca oob tcp \
    #hostname"
   # sh -c "ip addr
#cat /etc/hosts
#cat /etc/resolv.conf"
   # mpirun -np 2 \
   # --host a3m123-a3meganodeset-3:1,a3m123-a3meganodeset-29:1 \
   # --allow-run-as-root \
   # -mca plm slurm \
   # -mca btl_tcp_if_include enp0s12 \
   # -mca oob_tcp_if_include enp0s12 \
   # hostname"
    #sh -c "mpirun -np 2 \
    #--host a3m123-a3meganodeset-3:1,a3m123-a3meganodeset-29:1 \
    #--allow-run-as-root \
    #-mca plm_rsh_args '-o StrictHostKeyChecking=no' \
    #-mca btl_tcp_if_include enp0s12 \
    #-mca oob_tcp_if_include enp0s12 \
    #-mca plm_base_verbose 10 \
    #hostname"
    #sh -c "ssh -i /root/.ssh/id_rsa -o StrictHostKeyChecking=no a3m123-a3meganodeset-3 hostname; ssh -i /root/.ssh/id_rsa -o StrictHostKeyChecking=no a3m123-a3meganodeset-29 hostname"
#    sh -c "ssh a3m123-a3meganodeset-3 hostname -o StrictHostKeyChecking=no;ssh a3m123-a3meganodeset-29 hostname -o StrictHostKeyChecking=no"
#    sh -c "mpirun -np 2 --host a3m123-a3meganodeset-3:1,a3m123-a3meganodeset-29:1 \
#  --allow-run-as-root \
#  --debug-devel \
#  hostname"
#   
#
NCCL_LIB_DIR="/var/lib/tcpxo/lib64" source /var/lib/tcpxo/lib64/nccl-env-profile.sh
export NCCL_FASTRAK_CTRL_DEV=enp0s12
export NCCL_FASTRAK_IFNAME=enp6s0,enp7s0,enp13s0,enp14s0,enp134s0,enp135s0,enp141s0,enp142s0
export NCCL_SOCKET_IFNAME=enp0s12
export NCCL_FASTRAK_LLCM_DEVICE_DIRECTORY=/dev/aperture_devices

# Here we grab all the environment variables that need to be
# passed down into the container. Slurm would otherwise only pass these env vars
# to the job environment on the host.
# shellcheck disable=SC2001
HOST_VARS=$(sed 's/ \{1,\}/,/g' <<<"${!NCCL*}")

sudo srun \
    --container-image=./nccl+slurm.sqsh \
    --ntasks-per-node=8 \
    --nodes=2 \
    --mpi=pmi2 \
    --container-mounts="${CONTAINER_MOUNTS},/var/run/munge:/var/run/munge,/opt/apps:/opt/apps,/usr/sbin,/var/run/slurm,/tmp:/tmp,/etc/ssh:/etc/ssh,/etc/passwd:/etc/passwd,/var/lib/tcpxo/lib64,/usr/local/bin,/usr/local/lib,/var/spool/slurmd,/var/spool/slurm,/usr/lib64" \
    --container-env="${HOST_VARS}" \
    sh -c "export NODE_NAME=$HOSTNAME;export NHOSTS=2;export nr=8;export JOB_COMPLETION_INDEX=0; export BANDWIDTH_THRESHOLD=150;
    export START_MESSAGE_SIZE=2G; export END_MESSAGE_SIZE=8G; export JOB_NAME=JOB_NAME;export SERVICE_NAME=SERVICE_NAME; export DRY_RUN=true;
    export INSTANCE_TYPE=a3-megagpu-8g; export ITERATIONS=1;
    python3 /scripts/nccl_startup.py"
    #sh -c "export LD_LIBRARY_PATH=/var/lib/tcpxo/lib64:/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH;/nccl/nccl-tests/build/all_gather_perf -b 8M -e 8G -f 2 -g 1 -w 5 --iters 200 -c 0;"
