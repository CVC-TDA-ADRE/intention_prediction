ARG IMAGE_BASE

FROM ${IMAGE_BASE}

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
    python3-opencv ca-certificates python3-dev git wget sudo
# RUN ln -sv /usr/bin/python3 /usr/bin/python

WORKDIR /code
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

RUN pip3 install pytorch-lightning
RUN git clone https://github.com/facebookresearch/pytorchvideo.git
RUN sed -i "s|>=3.7|>=3.6 |g" pytorchvideo/setup.py
RUN pip3 install --no-cache-dir -e pytorchvideo

RUN pip3 install git+https://github.com/facebookresearch/fvcore.git
RUN pip3 install ffmpeg-python
RUN pip3 install scipy

RUN pip3 install --user tensorboard cmake   # cmake from apt-get is too old
RUN pip3 uninstall -y torch torchvision
RUN pip3 install torch torchvision
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
ENV FORCE_CUDA="1"
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
RUN pip3 install --user -e detectron2_repo

# COPY ../../SegFormer /software/SegFormer
# RUN pip3 install /software/SegFormer
