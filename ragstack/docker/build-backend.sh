#!/bin/bash
set -e
RAGSTACK_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. >/dev/null 2>&1 && pwd )"

cd $RAGSTACK_DIR
version=$(poetry version | awk '{print $2}')
echo "build docker image version $version ..."

cd $RAGSTACK_DIR/docker/backend
echo "Building backend image"
docker build --build-arg VERSION=${version} -t ragstack-ai-langflow-backend:latest -f Dockerfile ../../..
echo "Done ragstack-ai-langflow-backend:latest "

cd $RAGSTACK_DIR/docker/backend-ep
docker build --build-arg VERSION=${version} -t ragstack-ai-langflow-backend-ep:latest -f Dockerfile ../../..
echo "Done ragstack-ai-langflow-backend-ep:latest "

