import os
import pathlib
import re
import shutil
import pytest
import yaml
from chs.cli import cli
from click.testing import CliRunner
from tests.integration.test_utils.workspace_scrubber import WorkspaceScrubber

GOLDENS_DIR = pathlib.Path(__file__).parent / "goldens"


def get_test_cases():
    if not GOLDENS_DIR.exists():
        return []
    return [
        test_dir.name
        for test_dir in GOLDENS_DIR.iterdir()
        if test_dir.is_dir() and (test_dir / "test_config.yaml").exists()
    ]


@pytest.fixture
def mock_cluster_env(tmp_path, monkeypatch):
    """Sets up a mock environment with dummy cluster binaries to bypass host
    hardware checks."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    mock_script = (
        "#!/bin/bash\n"
        'cmd=$(basename "$0")\n'
        'case "$cmd" in\n'
        "  srun)\n"
        '    if [[ "$*" == *"ip route show default"* ]]; then\n'
        '      echo "default via 10.0.0.1 dev enp0s1"\n'
        '    elif [[ "$*" == *"ip route show"* ]]; then\n'
        '      echo -e "10.0.0.0/24 dev enp0s1 proto kernel scope link '
        "src 10.0.0.2\n10.1.0.0/24 dev enp0s2 proto kernel scope link "
        'src 10.1.0.2"\n'
        "    else\n"
        "      exit 0\n"
        "    fi\n"
        "    ;;\n"
        "  kubectl)\n"
        '    if [[ "$*" == *"get networks.networking.gke.io"* ]]; then\n'
        '      echo \'{"items": [{"metadata": {"name": "default"}, '
        '"spec": {"type": "L3"}}, {"metadata": {"name": "gpu-net-1"}, '
        '"spec": {"type": "Device"}}]}\'\n'
        '    elif [[ "$*" == *"get crd"* ]]; then\n'
        '      if [[ "$MOCK_KUEUE_MISSING" == "1" ]]; then\n'
        '        echo "jobsets.jobset.sigs.k8s.io"\n'
        "      else\n"
        '        echo -e "jobsets.jobset.sigs.k8s.io\\n'
        'kueues.kueue.sigs.k8s.io"\n'
        "      fi\n"
        '    elif [[ "$*" == *"apply -f"* ]]; then\n'
        '      if [[ "$MOCK_KUEUE_APPLY_FAIL" == "1" ]]; then\n'
        '        echo "apply failed due to permissions" >&2\n'
        "        exit 1\n"
        "      fi\n"
        "      exit 0\n"
        '    elif [[ "$*" == *"get deployment -n kueue-system"* ]]; then\n'
        '      echo "registry.k8s.io/kueue/kueue:v0.14.5"\n'
        '    elif [[ "$*" == *"get node"* ]]; then\n'
        '      echo \'{"metadata": {"labels": '
        '{"cloud.google.com/gke-tpu-topology": "2x2x1"}}}\'\n'
        '    elif [[ "$*" == *"get clusterqueue"* ]]; then\n'
        '      if [[ "$MOCK_KUEUE_APPLY_FAIL" == "1" ]]; then\n'
        "        exit 1\n"
        "      fi\n"
        '      echo "True"\n'
        "    else\n"
        "      exit 0\n"
        "    fi\n"
        "    ;;\n"
        "  *)\n"
        "    exit 0\n"
        "    ;;\n"
        "esac\n"
    )
    for mock_bin in [
        "enroot",
        "unsquashfs",
        "srun",
        "kubectl",
        "sinfo",
        "scontrol",
    ]:
        mock_path = bin_dir / mock_bin
        mock_path.write_text(mock_script)
        mock_path.chmod(0o755)

    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ.get('PATH', '')}")
    return bin_dir


@pytest.mark.parametrize("test_case", get_test_cases())
def test_golden_config(
    test_case, tmp_path, mock_cluster_env, mocker, monkeypatch
):
    runner = CliRunner()
    test_dir = GOLDENS_DIR / test_case
    config_path = test_dir / "test_config.yaml"
    snapshot_dir = test_dir / "snapshot"

    config = yaml.safe_load(config_path.read_text())

    env_vars = config.get("env", {})
    for k, v in env_vars.items():
        monkeypatch.setenv(k, str(v))

    command = config.get("command", [])

    workspaces_dir = tmp_path / "workspaces"
    workspaces_dir.mkdir()

    mocker.patch(
        "chs.utils.app_data.ChsAppData.get_workspaces_dir",
        return_value=str(workspaces_dir),
    )

    result = runner.invoke(cli, command)
    assert (
        result.exit_code == 0
    ), f"Command failed: {result.output}\nException: {result.exception}"

    generated_workspaces = list(workspaces_dir.iterdir())
    assert (
        len(generated_workspaces) == 1
    ), "Expected exactly one workspace to be generated"
    workspace_path = generated_workspaces[0]

    def normalize_content(content):
        # Sort comma-separated env vars inside --container-env
        # to avoid non-deterministic ordering.
        def sort_env(m):
            sep, envs = m.group(1), m.group(2)
            sorted_envs = ",".join(sorted(envs.split(",")))
            return f"--container-env{sep}{sorted_envs}"

        normalized = re.sub(
            r"--container-env([=\s])([^\-\s]+)",
            sort_env,
            content,
        )
        return sorted(
            line.rstrip() for line in normalized.splitlines() if line.strip()
        )

    scrubber = WorkspaceScrubber(str(workspace_path))
    temp_snapshot_dir = tmp_path / "temp_snapshot"
    scrubber.extract_and_scrub(str(temp_snapshot_dir))

    update_goldens = os.environ.get("UPDATE_GOLDENS") == "1"

    if update_goldens:
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        for golden_item in sorted(snapshot_dir.rglob("*"), reverse=True):
            rel_path = golden_item.relative_to(snapshot_dir)
            temp_item = temp_snapshot_dir / rel_path

            if (
                not temp_item.exists()
                or golden_item.is_dir() != temp_item.is_dir()
            ):
                if golden_item.is_file():
                    golden_item.unlink()
                elif golden_item.is_dir():
                    shutil.rmtree(golden_item)

        for new_file in temp_snapshot_dir.rglob("*"):
            if not new_file.is_file():
                continue

            rel_path = new_file.relative_to(temp_snapshot_dir)
            golden_file = snapshot_dir / rel_path

            if golden_file.exists():
                new_content = new_file.read_text()
                old_content = golden_file.read_text()
                if normalize_content(new_content) == normalize_content(
                    old_content
                ):
                    continue

            golden_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(new_file, golden_file)

        assert snapshot_dir.exists()
        return

    assert snapshot_dir.exists(), (
        f"Golden snapshot not found for {test_case}. "
        "Run with UPDATE_GOLDENS=1."
    )

    current_files = {
        str(filepath.relative_to(temp_snapshot_dir)): filepath.read_text()
        for filepath in temp_snapshot_dir.rglob("*")
        if filepath.is_file()
    }

    golden_files = {
        str(filepath.relative_to(snapshot_dir)): filepath.read_text()
        for filepath in snapshot_dir.rglob("*")
        if filepath.is_file()
    }

    # Verify no missing or extra files
    assert set(current_files.keys()) == set(
        golden_files.keys()
    ), "Mismatch in generated snapshot files"

    # Verify content exactly (pytest will show clear diffs on failure)
    for rel_path, current_content in current_files.items():
        golden_content = golden_files[rel_path]

        # Since some execution scripts and commands in Ramble have
        # non-deterministic line ordering, we simply sort the lines of all
        # generated files before diffing to prevent flakiness.
        # We also ignore trailing whitespaces and empty lines.
        assert normalize_content(current_content) == normalize_content(
            golden_content
        ), (
            "Content mismatch (ignoring order and trailing whitespace) "
            f"in {rel_path}"
        )
