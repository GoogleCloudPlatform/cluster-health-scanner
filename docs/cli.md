# CLI reference

```bash
chs [-v] <command> [OPTIONS] [ARGS]...
```

- `-v`, `--verbose`: enable verbose logging. Pass it before `<command>`, for example `chs -v run ...`.
- `-h`, `--help`: show help for `chs` or any command, for example `chs run -h`.
- Run `chs --version` to print the version.

> **Note:** CHS stores run history, metadata, and settings **locally in `~/.local/share/chs/`** (or in the `chs-local-state` Docker volume) on the machine where `chs` is run. Commands like `chs results` and `chs history` only show runs started from that environment.

## chs list

List supported machine types, suites, tests, and schedulers.

**Arguments:**

- `chs list`: List all supported machine types, suites, tests, and schedulers.
- `chs list suites`: List available test suites (`lite`, `standard`).
- `chs list tests`: List individual tests.
- `chs list machine-types`: List available machine types.
- `chs list schedulers`: List available schedulers (`kubernetes`, `slurm`, `standalone`).

**Options:**

| Option | Description |
| --- | --- |
| `--tpu` | Show only TPU tests, suites, machine types, and schedulers. |
| `--gpu` | Show only GPU tests, suites, machine types, and schedulers. |
| `--suite SUITE_NAME` | Filter tests by suite (`chs list tests` only). |
| `--machine-type MACHINE_TYPE` | Filter tests by machine type (`chs list tests` only). |

## chs run

Run tests or a suite on your cluster or TPU VM. All nodes targeted in a single run must be the **same accelerator (machine type)**. See [Tests and suites](../src/chs/tests/README.md) and [Machine types and thresholds](../src/chs/platforms/README.md) for the supported suites, tests, and machine types.

**Options:**

| Option | Description |
| --- | --- |
| `-s`, `--suite SUITE_NAME` | Predefined test suite to run (`lite` or `standard`). Defaults to `lite` when `--tests` is not specified. |
| `-t`, `--tests TEST_LIST` | Comma-separated list (or multiple flags) of individual tests to run instead of a full suite. |
| `--nodepools NODEPOOL_LIST` | Comma-separated list (or multiple flags) of Kubernetes node pools to target. Required when targeting multi-host TPU tests on Kubernetes. |
| `--nodes NODE_LIST` | Comma-separated list (or multiple flags) of individual nodes to run the tests on. For TPUs, individual nodes are supported for single-host tests; multi-host tests require `--nodepools`. |
| `--partitions PARTITION_LIST` | Comma-separated list (or multiple flags) of Slurm partitions to run the tests on. |
| `--nodelists NODE_PATTERN_LIST` | Slurm nodelist pattern representing the nodes to target (for example `node-[1-3],node-[4-5]`). |
| `-y`, `--auto-approve` | Automatically confirm the execution preview without prompting. Useful for automated scripts and CI environments. |
| `--run-id TEXT` | Custom run ID for this CHS execution. If omitted, a unique identifier based on the cluster name and suite (for example `cluster-name-lite-hash`) is automatically generated. |
| `--scheduler [kubernetes\|slurm\|standalone]` | Scheduler to use, overriding automatic scheduler detection. |
| `--storage-path DESTINATION` | Where to save this run's results. A `gs://bucket[/prefix]` URI uploads them to GCS instead of the configured storage. Any other value is a local directory: results go to `<DESTINATION>/<workspace_name>/` instead of `./chs-output/`. With Docker, use a path under `/workspace/chs-output` or a bind-mounted directory. |

**Examples:**

```bash
# Run the default 'lite' suite on Kubernetes node pools
chs run --nodepools tpu-pool-1,tpu-pool-2

# Run specific tests on a Kubernetes node pool
chs run --tests tpu-hbm,tpu-collectives --nodepools tpu-pool-1

# Run the 'standard' suite on a Kubernetes node pool
chs run --suite standard --nodepools tpu-pool-1

# Run the default 'lite' suite on a Slurm partition
chs run --partitions gpu-partition

# Run the default 'lite' suite on a standalone TPU VM
chs run
```

## chs results

Print a run's ID, submission time, number of nodes tested, and the locations of its interactive health report and Bill of Health. If no `RUN_ID` is provided, you are prompted to select one.

**Examples:**

```bash
# Select from recent runs interactively
chs results

# View results for a specific run ID
chs results my-cluster-lite-a1b2
```

## chs history

Show the history of the last 5 runs (run ID, submission status, storage target, age, and command executed).

```bash
chs history
```

## chs cancel

Cancel a CHS run and clean up its resources. Cancels the run's jobs and deletes its resources on the cluster, which releases its nodes for other runs. Without `RUN_ID`, your latest run is cancelled. `RUN_ID` can be any run on the cluster, including runs started by other users.

**Options:**

| Option | Description |
| --- | --- |
| `--all` | Cancel every CHS run on the cluster, including runs started by other users, and delete all CHS resources. |
| `-y`, `--auto-approve` | Automatically confirm cancellation without prompting. |

**Examples:**

```bash
# Cancel your latest run
chs cancel

# Cancel a specific run
chs cancel my-cluster-lite-ab12

# Cancel every CHS run on the cluster
chs cancel --all
```

## chs configure

Configure CHS settings.

**Subcommands:**

- `chs configure storage gcs [BUCKET_OR_PATH]`: Configure GCS storage. Accepts a bucket name, a `bucket/prefix` path, or a `gs://` URI (for example `my-bucket`, `my-bucket/team-a`, `gs://my-bucket/team-a`).
- `chs configure storage show`: Show currently configured storage.
- `chs configure storage remove`: Remove configured storage.
- `chs configure telemetry enable`: Enable telemetry. See [Telemetry](telemetry.md).
- `chs configure telemetry disable`: Disable telemetry.
- `chs configure telemetry status`: Show whether telemetry is enabled.
