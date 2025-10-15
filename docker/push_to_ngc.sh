#!/bin/bash
set -e

TAG_NAME=latest
DRY_RUN_MODE=false
DATASET_PATH=""
DATASET_DIR=""
DATASET_NAME=""
CONTAINER_ID=""
PUSH_TO_NGC=false

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

while getopts ":t:v:dph" OPTION; do
    case $OPTION in
        t)
            TAG_NAME=$OPTARG
            echo "Tag name is ${TAG_NAME}."
            ;;
        d)
            DRY_RUN_MODE=true
            echo "DRY_RUN mode (build and run docker locally)."
            ;;
        v)
            DATASET_PATH=$OPTARG
            DATASET_DIR=$(dirname "$DATASET_PATH")
            DATASET_NAME=$(basename "$DATASET_PATH")
            echo "Dataset path is ${DATASET_PATH}. Only used in dry run mode (-d)."
            ;;
        p)
            PUSH_TO_NGC=true
            echo "PUSH_TO_NGC (build and push to ngc)."
            ;;
        h | *)
            echo "Helper script for pushing mindmap images to NGC."
            echo "Usage:"
            echo "- dry run mode:"
            echo "    run_docker.sh -d -v <PATH_TO_DATASET>"
            echo "- pushing to NGC:"
            echo "    run_docker.sh -p -t tag_name"
            echo "- see help message:"
            echo "    run_docker.sh -h"
            echo ""
            echo "  -d Dry run mode."
            echo "  -v Dataset path for dry run mode."
            echo "  -p Push to NGC mode."
            echo "  -t Tag name of the image."
            echo "  -h help (this output)"
            exit 0
            ;;
    esac
done

# On exit, stop the container.
cleanup() {
    if [ -n "$CONTAINER_ID" ]; then
        echo "Stopping Docker container with ID: $CONTAINER_ID"
        docker stop $CONTAINER_ID
    fi
}
trap cleanup EXIT

# Get the NGC path.
NVBLOX_MINDMAP_IMAGE_NAME=nvblox_mindmap
NGC_PATH=nvcr.io/PLACEHOLDER_NGC_PATH/${NVBLOX_MINDMAP_IMAGE_NAME}:${TAG_NAME}

# Build the base image.
MINDMAP_DEPS_IMAGE_NAME="mindmap_deps"
"$SCRIPT_DIR"/build_images.sh $MINDMAP_DEPS_IMAGE_NAME

docker build --progress=plain --network=host -t ${NVBLOX_MINDMAP_IMAGE_NAME} . -f docker/Dockerfile.build_for_ngc \
    --build-arg BASE_IMAGE=${MINDMAP_DEPS_IMAGE_NAME}

# Remove any old containers (exited or running).
if [ "$(docker ps -a --quiet --filter name=$NVBLOX_MINDMAP_IMAGE_NAME)" ]; then
    docker rm -f $NVBLOX_MINDMAP_IMAGE_NAME > /dev/null
fi

if [ "$DRY_RUN_MODE" = true ]; then

    DOCKER_RUN_ARGS=(
            "--name" "$NVBLOX_MINDMAP_IMAGE_NAME"
            "-detach"
            "--rm"
            "--privileged"
            "--ulimit" "memlock=-1"
            "--ulimit" "stack=-1"
            "--ipc=host"
            "--net=host"
            "--gpus" 'all,"capabilities=compute,utility,graphics"'
            "-v" "${DATASET_DIR}:/dataset"
    )
    # Start the container
    CONTAINER_ID=$(docker run "${DOCKER_RUN_ARGS[@]}" ${NVBLOX_MINDMAP_IMAGE_NAME} /bin/bash -c "sleep infinity")
    echo "Docker container started with ID: $CONTAINER_ID"

    # Start the training to verify the container.
    echo "Start training for verification. Stop at any point you feel like you verified the container is good to go."
    wandb_offline_cmd="wandb offline"
    multi_gpu_training_cmd="torchrun --standalone --nnodes 1 --nproc_per_node 1 --master_port 29400 "
    training_cmd="diffuser_actor/run_training.py --dataset /dataset/${DATASET_NAME} --valset /dataset/${DATASET_NAME}"
    docker exec -it $NVBLOX_MINDMAP_IMAGE_NAME bash -ic "${wandb_offline_cmd} && ${multi_gpu_training_cmd}${training_cmd}"

elif [ "$PUSH_TO_NGC" = true ]; then

    # Tag and push the image to NGC.
    echo "Pushing container to ${NGC_PATH}."
    docker tag ${NVBLOX_MINDMAP_IMAGE_NAME} ${NGC_PATH}
    docker push ${NGC_PATH}
    echo "Pushing complete."

else

    echo "No mode selected. -d for dry run and -p for pushing to ngc."

fi
