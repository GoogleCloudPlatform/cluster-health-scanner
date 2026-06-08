apiVersion: kueue.x-k8s.io/v1beta1
kind: LocalQueue
metadata:
  name: chs-local-queue
  namespace: {namespace}
spec:
  clusterQueue: chs-cluster-queue
