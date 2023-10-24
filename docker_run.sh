#!/bin/bash
docker run -it \
    --name rl_nav_husky \
    --user ros \
    --gpus all \
    --env NVIDIA_VISIBLE_DEVICES=all   \
    --env NVIDIA_DRIVER_CAPABILITIES=all  \
    --env DISPLAY=${DISPLAY}  \
    --volume /tmp/.X11-unix:/tmp/.X11-unix \
    -v $PWD:/docker_ws \
    --network host \
    --runtime nvidia \
    cuda_humble_pytorch:v1

