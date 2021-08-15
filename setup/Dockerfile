FROM tensorflow/tensorflow:2.4.0-gpu

RUN mkdir work && cd work
ADD requirements.txt requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt
WORKDIR /work