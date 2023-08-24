#!/usr/bin/env bash
set -ex

# Build the source files
test -d build || mkdir build
cd build && cmake -GNinja -DCMAKE_BUILD_TYPE=Release .. && ninja
cd ../..

# The run_all_tests.sh scripts expects to run from the github root directory.
# This way, we can share the script with github actions.
./code/scripts/run_all_tests.sh "$@"
