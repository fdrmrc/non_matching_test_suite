#!/bin/bash
if ! test -f docker/Dockerfile; then
    echo "You must run this script from the top level directory of the repository."
    exit 1
fi

TAG=latest

if [ -n "$1" ]; then 
    TAG=$(echo $1 | sed "s/\//-/g")
    if [ "$TAG" == "main" ]; then 
	TAG=latest
    fi
fi
echo "Will build and push ghcr.io/fdrmrc/non_matching_test_suite:$TAG"

# Run this script from the top level directory
docker build . -f docker/Dockerfile \
    --tag ghcr.io/fdrmrc/non_matching_test_suite:$TAG

docker push ghcr.io/fdrmrc/non_matching_test_suite:$TAG
