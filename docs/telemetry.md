# Telemetry

CHS sends anonymous usage metrics to Google (`play.googleapis.com`) to help improve the tool. It doesn't collect personally identifiable information (PII). Telemetry is on by default.

## What is collected

CHS reports the following metrics (never node names, cluster names, bucket names, flag values, or raw metric values):

| Metric | Description |
| --- | --- |
| `PARENT_COMMAND` | Top-level `chs` subcommand (`run`, `results`, `list`, `cancel`, `history`, `configure`). |
| `COMMAND_FLAGS` | CLI flags passed to the command (flag names only, without their values). |
| `CLUSTER_TYPE` | Environment where CHS ran (`GKE`, `SLURM`, `STANDALONE`, or `UNKNOWN`). |
| `STORAGE_TYPE` | Configured storage backend (`GCS`, or `NONE` when saving to local disk only). |
| `MACHINE_TYPE` | Detected machine type of the target nodes. |
| `ZONE` | GCP zones where the target nodes are located. |
| `NODE_COUNT` | Total number of nodes targeted in the run. |
| `IS_TEST_DATA` | Whether the run used a CHS team test project (`true` or `false`). |
| `TPU_SLICE_TYPE` | TPU slice provisioning type (`static` or `dynamic`) on Kubernetes. |
| `TPU_SLICE_TOPOLOGIES` | Number of TPU slices tested per topology (for example `{"2x2x1": 3, "4x4x4": 2}`). |
| `TPU_NODEPOOL_COUNT` | Number of TPU node pools tested when targeting with `--nodepools`. |
| `TEST_SUITE` | Name of the test suite run, when `--suite` is used. |
| `TEST_NAME` | Names of the diagnostic tests executed in the run. |
| `TEST_DURATION_MS` | Total run duration in milliseconds. |
| `TEST_RESULTS` | Per-test target counts (nodes, node pairs, or domains) by outcome. |
| `DETECTED_ERRORS` | Number of nodes with detected errors or warnings, grouped by test, level, severity, and category. |
| `UPLOAD_STATUS` | Upload status (`success` or `failed`) per artifact (`bill_of_health`, `workspace`) when remote storage is configured. |
| `ARTIFACT_SIZE_BYTES` | Size in bytes of each uploaded artifact when uploading to GCS. |
| `EXIT_CODE` | Process exit code of the `chs` command (`0` on success). |
| `ERROR_CATEGORY` | Python exception type if the `chs` command exited with an error. |

To inspect the exact telemetry payload sent during a command, pass `-v` (for example `chs -v run`).

## Opting out

- Run `chs configure telemetry disable` to turn telemetry off (`enable` turns it back on, and `status` shows the current state).
- Set `CHS_TELEMETRY=0` or `DO_NOT_TRACK=1` to disable telemetry via environment variable (for example `docker run -e CHS_TELEMETRY=0 ...`).
