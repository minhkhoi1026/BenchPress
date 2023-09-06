FROM ubuntu:20.04

WORKDIR /home/

RUN apt-get update
RUN apt-get install -y sudo git wget

RUN git clone https://github.com/minhkhoi1026/benchpress.git

WORKDIR /home/benchpress
RUN echo "test"
RUN bash requirements.apt

WORKDIR /home/benchpress
RUN mkdir build
WORKDIR /home/benchpress/build
RUN cmake ..
RUN make -j 4

# Run a simple example
CMD ./benchpress --config model_zoo/github/bq_C_db.pbtxt
