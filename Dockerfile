FROM debian:latest

RUN apt-get -y update && apt-get install -y git python3-pip python3-dev python3-tk vim procps curl

#Face classificarion dependencies & web application
RUN pip3 install numpy scipy scikit-learn pillow tensorflow pandas h5py opencv-python==3.2.0.8 keras statistics pyyaml pyparsing cycler matplotlib Flask

ADD . /ekholabs/face-classifier

WORKDIR ekholabs/face-classifier

ENV PYTHONPATH=$PYTHONPATH:src
ENV FACE_CLASSIFIER_PORT=8084
EXPOSE $FACE_CLASSIFIER_PORT

ENTRYPOINT ["python3"]
CMD ["src/web/faces.py"]
