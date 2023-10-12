#!/usr/bin/env bash
docker run --runtime nvidia -it \
    -v $(pwd):/workspace \
    -v $HOME/.keras:/root/.keras \
    flwr-tf python -m experiments.exp2_cifar10 $@