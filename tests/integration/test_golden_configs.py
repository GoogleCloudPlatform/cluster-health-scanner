import os
import pathlib
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
    mock_script = "#!/bin/bash\nexit 0\n"
    for mock_bin in ["enroot", "unsquashfs"]:
        mock_path = bin_dir / mock_bin
        mock_path.write_text(mock_script)
        mock_path.chmod(0o755)

    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ.get('PATH', '')}")
    return bin_dir


@pytest.mark.parametrize("test_case", get_test_cases())
def test_golden_config(test_case, tmp_path, mock_cluster_env, mocker):
    runner = CliRunner()
    test_dir = GOLDENS_DIR / test_case
    config_path = test_dir / "test_config.yaml"
    snapshot_dir = test_dir / "snapshot"

    config = yaml.safe_load(config_path.read_text())

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

    scrubber = WorkspaceScrubber(str(workspace_path))
    update_goldens = os.environ.get("UPDATE_GOLDENS") == "1"

    if update_goldens:
        scrubber.extract_and_scrub(str(snapshot_dir))
        assert snapshot_dir.exists()
        return

    temp_snapshot_dir = tmp_path / "temp_snapshot"
    scrubber.extract_and_scrub(str(temp_snapshot_dir))

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

    def normalize_content(content):
        return sorted(
            line.rstrip() for line in content.splitlines() if line.strip()
        )

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
