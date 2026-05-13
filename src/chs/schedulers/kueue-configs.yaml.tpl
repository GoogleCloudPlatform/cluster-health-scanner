---
apiVersion: kueue.x-k8s.io/v1beta1
kind: Topology
metadata:
  name: chs-node-topology
spec:
  levels:
  - nodeLabel: kubernetes.io/hostname
---
apiVersion: kueue.x-k8s.io/v1beta1
kind: ResourceFlavor
metadata:
  name: chs-flavor
spec:
  topologyName: chs-node-topology
  nodeLabels:
    node.kubernetes.io/instance-type: {instance_type}
---
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata:
  name: chs-cluster-queue
spec:
  namespaceSelector: {{}}
  resourceGroups:
  - coveredResources: ["nvidia.com/gpu"]
    flavors:
    - name: chs-flavor
      resources:
      - name: "nvidia.com/gpu"
        nominalQuota: "{nominal_quota}"
---
apiVersion: kueue.x-k8s.io/v1beta1
kind: LocalQueue
metadata:
  name: chs-local-queue
spec:
  clusterQueue: chs-cluster-queue
