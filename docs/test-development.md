# Test development guide

Each test lives in [`src/chs/tests/<test-name>/`](../src/chs/tests). CHS discovers tests automatically from that directory, and the folder name becomes the test name used in `chs run --tests` and `chs list tests`. See [`stream`](../src/chs/tests/stream) for a complete example.

## 1. plugin.py

Define a `Plugin` class that inherits from one of the base classes in [`src/chs/testkit/base.py`](../src/chs/testkit/base.py):

- `SingleNodePlugin`: Runs a separate job on each target node.
- `MultiNodePlugin`: Runs a single coordinated job across multiple nodes.
- `SlicePlugin`: Runs across a TPU slice (set `requires_whole_slice = True` if the test requires the entire slice).

```python
from chs.testkit import SingleNodePlugin


class Plugin(SingleNodePlugin):
    name = "my-test"              # Must match the folder name
    category = "Compute"          # Shown in `chs list tests`
    description = "Short summary shown in `chs list tests`"
    long_description = "Longer explanation shown in the interactive health report."
    application = "my-test"       # Must match `name` in application.py
    default_workloads = ("standard",)
```

## 2. application.py

CHS uses [Ramble](https://github.com/GoogleCloudPlatform/ramble) to generate job scripts and parse log files. Define an `ExecutableApplication` with:

- `executable()` and `workload()`: The command to run on the target nodes. Placeholders like `{size}` are filled from `workload_variable()` defaults or `config.yaml`.
- `figure_of_merit()`: A regular expression (`fom_regex`) that extracts a metric from the job's stdout log.

```python
from ramble.appkit import *


class MyTest(ExecutableApplication):
    name = "my-test"

    executable("execute", "./my-benchmark --size {size}", use_mpi=True)
    workload("standard", executables=["execute"])

    workload_variable(
        "size",
        default="1024",
        description="Problem size",
        workloads=["standard"],
    )

    figure_of_merit(
        "bandwidth",
        fom_regex=r"Bandwidth:\s+(?P<bw>[0-9.]+)\s+GB/s",
        group_name="bw",
        units="GB/s",
    )
```

## 3. config.yaml

Configures where the test can run and which container image and variables to use:

- `platforms`: Supported [machine types](../src/chs/platforms/README.md). Use `machine-type: [kubernetes]` to restrict a machine type to specific schedulers.
- `defaults`: Settings shared across all machine types, with scheduler-specific blocks under `kubernetes`, `slurm`, or `standalone`.
- `machines`: Per-machine-type overrides for `defaults`.

```yaml
platforms:
- a3-ultragpu-8g
- a4-highgpu-8g
- ct6e-standard-4t: [kubernetes, standalone]

defaults:
  variables:
    size: '4096'
  kubernetes:
    variables:
      container_image: us-docker.pkg.dev/my-project/my-image:1.0
      job_template: job_standard.yaml.tpl
      mpi_command: ''
  slurm:
    modifiers:
    - name: pyxis-enroot
    variables:
      container_name: my-image:1.0
      container_uri: docker://us-docker.pkg.dev#my-project/my-image:1.0
      srun_args: --container-image {container_path} --wait=60

machines:
  a4-highgpu-8g:
    variables:
      size: '8192'
```

## 4. thresholds.yaml

Sets the pass/fail baselines per machine type for the metrics defined in `application.py`. Optional; if omitted, metrics are reported without a verdict. `type` can be `min` (metric >= baseline), `scalar` (metric equals baseline), or `relative` (ratio to peak >= baseline).

```yaml
type: min
a3-ultragpu-8g:
  bandwidth: 300
a4-highgpu-8g:
  bandwidth: 600
```

## 5. extractor.py, renderer.py, and \_\_init\_\_.py

Format the parsed metrics for the Bill of Health (`bill_of_health.json`) and interactive health report (`bill_of_health.html`):

- `extractor.py`: Subclasses [`DefaultExtractor`](../src/chs/core/default_plugin/extractor.py) to customize how parsed metrics are stored in the Bill of Health.
- `renderer.py`: Subclasses [`DefaultRenderer`](../src/chs/core/default_plugin/renderer.py) and implements `get_column_schema()` and `get_detail_data()` to build the test's detailed metric table in the interactive health report (see [`stream/renderer.py`](../src/chs/tests/stream/renderer.py)).
- `__init__.py`: Combines the renderer, extractor, and [`TestPlugin`](../src/chs/core/plugin_base.py) into `__chs_plugin__`.

`extractor.py`:

```python
from chs.core.default_plugin.extractor import DefaultExtractor


class MyTestExtractor(DefaultExtractor):
    pass
```

`renderer.py`:

```python
from chs.core.default_plugin.renderer import DefaultRenderer
from chs.core.plugin_base import ColumnDef


class MyTestRenderer(DefaultRenderer):
    def get_column_schema(self) -> list[ColumnDef]:
        return [
            ColumnDef("node", "Node"),
            ColumnDef("bandwidth", "Bandwidth (GB/s)", align="right"),
        ]
```

`__init__.py`:

```python
from chs.core.plugin_base import TestPlugin
from .extractor import MyTestExtractor
from .renderer import MyTestRenderer


class MyTestPlugin(MyTestRenderer, MyTestExtractor, TestPlugin):
    pass


__chs_plugin__ = {
    "test_name": "my-test",
    "plugin": MyTestPlugin,
}
```
