# DockerImage

Docker (development) image with everything set up to work with `tomolab` and `NiftyRec`

## Info
- This image is based on Ubuntu 18.04 and integrates the development version of `cuda` toolkit (a smaller release version will be released in the future)
- The image is no greatly optimized. Once built, it needs around 5 GB of disk space, which more or less double one a new container is started from that image

## Pre-requisites
- Docker image manager (the one you are more confortable with, depending on you OS: e.g. docker-ce)
- Nvidia Docker 2 (needed to access GPU hardware on the host side)
- Updated Nvidia drivers installed on the host computer

## Quick start
- Clone this repository
- Make `run.sh` executable
- Make all scripts in folder `scripts/` executable
- From within the main folder, fire up a terminal and execute `sh run.sh` or just `./run.sh`
- Hopefully, the image will be built and a (self destructive) container will start right away.
- The process will start a `jupyter` server, which you may access from a browser of your host computer at:

  - http://localhost:8888 --> jupyer notebooks
  - http://localhost:8888/lab --> jupyter lab interface


- Be sure that no other process is blocking port 8888
- To stop the notebook server and destroy the container (the image will not be destroyed) just press `Ctrl+C` from within the terminal window
