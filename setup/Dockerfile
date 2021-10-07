FROM nvidia/cuda:11.1-devel AS cuda

FROM tensorflow/tensorflow:2.4.1-gpu
COPY --from=cuda /usr/local/cuda/bin/ptxas /usr/local/cuda/bin/ptxas
RUN apt update && apt install -y libgl1
ADD requirements.txt requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt && \
    mkdir work && cd work
WORKDIR /work