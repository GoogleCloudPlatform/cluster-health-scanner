apiVersion: accelerator.gke.io/v1beta1
kind: Slice
metadata:
  name: {slice_name}
  labels:
    created-by: chs
spec:
  type: {tpu_type}
  topology: {shape}
  partitionIds:
    - {partition_id}
