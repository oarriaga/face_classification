FROM debian:latest

RUN apt-get -y update && apt-get install -y git python2.7 python-pip python-dev science-statistics python-yaml python-flask

#Face classificarion dependencies & web application
RUN pip install numpy scipy scikit-learn pillow tensorflow pandas h5py opencv-python keras

WORKDIR ekholabs

#Clone the code
RUN git clone https://github.com/ekholabs/face_classification.git

WORKDIR face_classification

ENV FACE_CLASSIFIER_PORT=8084
EXPOSE $FACE_CLASSIFIER_PORT

ENTRYPOINT ["python"]
CMD ["src/web/faces.py"]
