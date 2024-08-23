WORKSPACE=copybara_bnd_fig
cd ~/vis && \
    rm -rf bnd_copybara_target && \
    git clone ~/bnd_copybara_target && \
    cd bnd_copybara_target && \
    git remote add other ../gke-tcpx-early-access/bad-node-detectors && \
    git fetch other

rm -rf ~/bnd_copybara_target && \
    git init --bare ~/bnd_copybara_target && \
    copybara /google/src/cloud/$USER/$WORKSPACE/google3/cloud/cluster/supercomputer/validation/bad_node_detectors/copy.bara.sky --force piper_to_git /google/src/cloud/$USER/$WORKSPACE/

