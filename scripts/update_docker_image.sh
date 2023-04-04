#!/bin/sh
test -f docker/Dockerfile || echo \
    "You must run this script from the top level directory of the repository."
# Run this script from the top level directory
docker build . -f docker/Dockerfile --tag ghcr.io/fdrmrc/non_matching_test_suite:latest
docker push ghcr.io/fdrmrc/non_matching_test_suite:latest

