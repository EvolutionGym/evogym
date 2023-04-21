FROM python:3.8-bullseye
USER root
RUN apt-get update
RUN apt-get install -y --no-install-recommends xorg-dev libglu1-mesa-dev libglew-dev
WORKDIR /project
COPY . /project/evogym
RUN pip install --upgrade pip
RUN pip install -r /project/evogym/requirements.txt
RUN pip install /project/evogym/
