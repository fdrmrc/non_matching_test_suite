#!/usr/bin/env bash
set -ex

# This is the master script for the capsule. When you click "Reproducible Run", the code in this file will execute.
cd ~/capsule/build/examples/
../../scripts/run_all_tests.sh "$@"
