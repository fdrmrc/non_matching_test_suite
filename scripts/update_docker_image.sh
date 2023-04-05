#!/bin/sh
test -f docker/Dockerfile || echo \
    "You must run this script from the top level directory of the repository."

TAG=latest

if [[ "$1" != "" ]]; then 
    TAG=$1
fi
echo "Will build and push ghcr.io/fdrmrc/non_matching_test_suite:$TAG"

# Run this script from the top level directory
docker build . -f docker/Dockerfile \
    --tag ghcr.io/fdrmrc/non_matching_test_suite:$TAG

docker push ghcr.io/fdrmrc/non_matching_test_suite:$TAG
