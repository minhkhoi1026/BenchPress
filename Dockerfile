FROM ubuntu:latest

WORKDIR /home/

RUN apt-get update
RUN apt-get install -y sudo git

RUN git clone https://3f2defbceff83ef75197a0d924fd2d96ef86e327@github.com/fivosts/clgen.git

WORKDIR /home/clgen
RUN bash apt_deps.sh
RUN bazel build //deeplearning/clgen

# Run a simple example
CMD ./clgen --min_samples 10 --config model_zoo/BERT/tiny.pbtxt
