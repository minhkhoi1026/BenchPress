FROM ubuntu:20.04

WORKDIR /home/

RUN apt-get update
RUN apt-get install -y sudo git wget

RUN git clone https://github.com/minhkhoi1026/benchpress.git

WORKDIR /home/benchpress
COPY requirements.apt .
RUN bash requirements.apt

WORKDIR /home/benchpress
RUN mkdir build
WORKDIR /home/benchpress/build
RUN cmake ..
RUN make -j 12

# Run a simple example
WORKDIR /home/benchpress
ENTRYPOINT ["/bin/bash"]
# CMD ./benchpress --config model_zoo/github/bq_C_db.pbtxt
