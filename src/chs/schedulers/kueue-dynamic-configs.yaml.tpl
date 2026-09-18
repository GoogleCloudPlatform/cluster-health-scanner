---
apiVersion: kueue.x-k8s.io/{kueue_api_version}
kind: Topology
metadata:
  name: chs-node-topology-dynamic
spec:
  levels:
  - nodeLabel: cloud.google.com/gce-topology-block
  - nodeLabel: cloud.google.com/gke-tpu-partition-{dynamic_topology}-id
  - nodeLabel: kubernetes.io/hostname
---
apiVersion: kueue.x-k8s.io/{kueue_api_version}
kind: Topology
metadata:
  name: chs-node-topology-static
spec:
  levels:
  - nodeLabel: kubernetes.io/hostname
---
apiVersion: kueue.x-k8s.io/{kueue_api_version}
kind: ResourceFlavor
metadata:
  name: chs-flavor-dynamic
spec:
  topologyName: chs-node-topology-dynamic
  nodeLabels:
    node.kubernetes.io/instance-type: {instance_type}
---
apiVersion: kueue.x-k8s.io/{kueue_api_version}
kind: ResourceFlavor
metadata:
  name: chs-flavor-static
spec:
  topologyName: chs-node-topology-static
  nodeLabels:
    node.kubernetes.io/instance-type: {instance_type}
---
apiVersion: kueue.x-k8s.io/{kueue_api_version}
kind: ClusterQueue
metadata:
  name: chs-cluster-queue
spec:
  namespaceSelector: {{}}
  resourceGroups:
  - coveredResources: ["{resource_name}"]
    flavors:
    - name: chs-flavor-dynamic
      resources:
      - name: "{resource_name}"
        nominalQuota: "{nominal_quota}"
    - name: chs-flavor-static
      resources:
      - name: "{resource_name}"
        nominalQuota: "{nominal_quota}"
---
apiVersion: kueue.x-k8s.io/{kueue_api_version}
kind: LocalQueue
metadata:
  name: chs-local-queue
  namespace: {namespace}
spec:
  clusterQueue: chs-cluster-queue
