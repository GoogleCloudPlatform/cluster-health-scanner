---
apiVersion: kueue.x-k8s.io/{kueue_api_version}
kind: Topology
metadata:
  name: chs-node-topology
spec:
  levels:
  - nodeLabel: kubernetes.io/hostname
---
apiVersion: kueue.x-k8s.io/{kueue_api_version}
kind: ResourceFlavor
metadata:
  name: chs-flavor
spec:
  topologyName: chs-node-topology
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
    - name: chs-flavor
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
