# Controllable TalkNet 

This is a fork of Controllable TalkNet to more easily control things programatically

## How to run

### Docker (Linux)
* Install Docker and NVIDIA Container Toolkit.
* [Download the Dockerfile.](https://raw.githubusercontent.com/abb128/ControllableTalkNet/main/Dockerfile)
Open a terminal, and navigate to the directory where you saved it.
* Run ```docker build -t talknet-offline-c .``` to build the image. Add ```sudo``` if you're not using rootless Docker.
* Run ```docker run -it --gpus all talknet-offline-c``` to start TalkNet