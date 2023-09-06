FROM ubuntu:latest

WORKDIR /home/

RUN apt-get update
RUN apt-get install -y sudo git wget

RUN git clone https://github.com/minhkhoi1026/BenchPress.git

WORKDIR /home/benchpress
RUN bash requirements.apt

WORKDIR /home/benchpress
RUN mkdir build
WORKDIR /home/benchpress/build
RUN cmake ..
RUN make -j 12

# Run a simple example
CMD ./clgen --min_samples 10 --config model_zoo/BERT/tiny.pbtxt
