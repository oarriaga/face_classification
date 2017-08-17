FROM debian:latest

RUN apt-get -y update && apt-get install -y git python2.7 python-pip python-dev python-tk vim procps curl

#Face classificarion dependencies & web application
RUN pip install numpy scipy scikit-learn pillow tensorflow pandas h5py opencv-python keras statistics pyyaml pyparsing cycler matplotlib Flask

ADD . /ekholabs/face-classifier

WORKDIR ekholabs/face-classifier

ENV PYTHONPATH=$PYTHONPATH:src
ENV FACE_CLASSIFIER_PORT=8084
EXPOSE $FACE_CLASSIFIER_PORT

ENTRYPOINT ["python"]
CMD ["src/web/faces.py"]
