import os
import subprocess
import sys
from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPyCommand(build_py):
  """Custom build_py command to generate protobuf files."""

  def run(self):
    from grpc_tools import protoc
    source_dir = "src"
    proto_file = os.path.join(
        source_dir, "chs", "protos", "bill_of_health.proto"
    )

    if os.path.exists(proto_file):
      # TODO: Replace print statements with proper logging statements
      print(f"Compiling {proto_file}...")
      command = [
          sys.executable,
          "-m",
          "grpc_tools.protoc",
          f"-I{source_dir}",
          f"--python_out={source_dir}",
          proto_file,
      ]

      try:
        subprocess.check_call(command)
        # TODO: Replace print statements with proper logging statements
        print("Protobuf compilation successful.")
      except subprocess.CalledProcessError as e:
        print(f"Protobuf compilation failed: {e}")
        sys.exit(1)
    else:
      print(f"Proto file not found: {proto_file}")

    super().run()


setup(
    cmdclass={
        "build_py": BuildPyCommand,
    },
)
