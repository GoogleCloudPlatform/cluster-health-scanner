#!/bin/bash
/builder/kubectl.bash # default script for image, which sets up various environment variables

python3 tests.py "$@"
