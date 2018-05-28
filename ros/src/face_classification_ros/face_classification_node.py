#!/usr/bin/env python

from statistics import mode
import operator
import rospy
import rospkg
import os
import cv2
import numpy as np
from keras.models import load_model


from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage, RegionOfInterest
from face_classification.srv import FaceClassificationBuffered , FaceClassificationContinuous
from face_classification.msg import FaceClassification,FaceClassificationArray

from utils.datasets import get_labels
from utils.inference import detect_faces, load_detection_model, apply_offsets, draw_bounding_box, draw_text
from utils.preprocessor import preprocess_input

from cv_bridge import CvBridge, CvBridgeError

class BufferFrames(object):

  def __init__(self):

    self.images = {}
    self.headers = {}
    self.faces_coordinates={}
    self.genders = {}
    self.emotions = {}

    self.counter = {}

  def add_frame(self,cv2_img,header,faces_classified,genders,emotions):

    # changing the key it's possible to change the mode criterion
    key = str(genders.count("woman"))+ "_"+ str(genders.count("man"))

    if key in self.counter:
      self.counter[key] = self.counter[key] + 1
    else:
      self.counter[key] = 1

    self.images[key] = cv2_img
    self.headers[key] = header
    self.faces_coordinates[key] = faces_classified
    self.genders[key] = genders
    self.emotions[key] = emotions

    rospy.logdebug(" This is the %d frame added with %s persons", self.counter[key], key)
    return

  def mode_frame(self):
    #mode
    key = max(self.counter.iteritems(), key=operator.itemgetter(1))[0]
    return self.images[key], self.headers[key], self.faces_coordinates[key], self.genders[key], self.emotions[key]

class FaceClassifier(object):

  def __init__(self):

    #Show image on cv2
    self.show_image=rospy.get_param("~show_image", False)

    # Models for face detection, emotion and gender classification
    detection_model = rospy.get_param("~detection_model",'haarcascade_frontalface_default.xml')
    emotion_model = rospy.get_param("~emotion_model",'fer2013_mini_XCEPTION.102-0.66.hdf5')
    gender_model = rospy.get_param("~gender_model",'simple_CNN.81-0.96.hdf5')

    # Dataset names
    emotion_dataset = rospy.get_param("~emotion_dataset", 'fer2013')
    gender_dataset = rospy.get_param("~gender_dataset", 'imdb')

    # Image topic
    self.image_topic = rospy.get_param("~image_topic", "/camera/rgb/image_raw/compressed")

    # delay before starting acquiring first image from the topic
    self.camera_delay = rospy.get_param("~camera_delay", 1)

    # boolean do define if the image kind of msg is CompressedImage or Image
    if rospy.get_param("~compressed", True):
      self.image_type = CompressedImage
    else:
      self.image_type = Image

    # Rate of the node
    self.node_rate = rospy.Rate(rospy.get_param("~node_rate",60))

    # Rate of the image acquisition in buffer mode
    self.image_acquisition_rate = rospy.Rate(rospy.get_param("image_acquisition_rate",4))

    # percentage in [width,height] that the bounding box will be amplied
    self.gender_offsets_percentage = rospy.get_param("~gender_offsets_percentage",(20, 40))
    self.emotion_offsets_percentage = rospy.get_param("~emotion_offsets_percentage",(20, 40))

    # get an instance of RosPack with the default search paths
    rospack = rospkg.RosPack()

    # paths for loading data and images
    detection_model_path =os.path.join(rospack.get_path('face_classification'),'trained_models','detection_models',detection_model)
    emotion_model_path = os.path.join(rospack.get_path('face_classification'),'trained_models','emotion_models',emotion_model)
    gender_model_path = os.path.join(rospack.get_path('face_classification'),'trained_models','gender_models', gender_model)

    #labels of each dataset
    self.emotion_labels = get_labels(emotion_dataset)
    self.gender_labels = get_labels(gender_dataset)


    self.bridge = CvBridge()

    # Store last image to process here
    self.last_image = None

    # Also a handy flag if the image is new
    self.is_new_image = False

    # Flag to anounce that a new mode (cointinuous or buffered) was activated
    self.activate_mode = False

    # SERVICES
    rospy.Service('face_classification_buffered', FaceClassificationBuffered, self.face_classification_buffer_srv_callback)
    rospy.Service('face_classification_continuous', FaceClassificationContinuous, self.face_classification_continuously_srv_callback)

    # SUBSCRIBER
    self.image_subscriber = None

    # PUBLISHERS
    self.publisher = rospy.Publisher("~faces_classified" , FaceClassificationArray ,queue_size=1)
    self.publisher_image = rospy.Publisher("~image_raw" , Image ,queue_size=1)
    self.publisher_image_compressed = rospy.Publisher("~image_raw/compressed" , CompressedImage ,queue_size=1)

    # node mode (idle, continuos or buffer)
    self.mode = "idle"

    # buffer_size in buffer mode
    self.buffer_size=None

    # loading models
    self.face_detection = load_detection_model(detection_model_path)
    self.emotion_classifier = load_model(emotion_model_path, compile=False)
    self.gender_classifier = load_model(gender_model_path, compile=False)

    # getting input model shapes for inference
    self.emotion_target_size = self.emotion_classifier.input_shape[1:3]
    self.gender_target_size = self.gender_classifier.input_shape[1:3]


  ##############################################################################
  #                             CALLBACK FUNCTIONS
  ##############################################################################

  def image_subscriber_callback(self,image):
    '''
    Image topic subscriber callback
    '''
    self.last_image = image
    self.is_new_image = True


  def face_classification_buffer_srv_callback(self, req):
    '''
    Buffer mode service callback
      change the mode of the node to buffer mode if possible
      (if is not already in buffer or continuous mode)
    '''

    if self.mode == "continuous":
      rospy.logwarn('Node working on continuous mode, request for buffer mode not available')
      return False

    elif self.mode == "buffer":
      rospy.logwarn('Node is working already on buffered_mode, with buffer_size %d' % self.buffer_size)
      return False

    else: #self.mode == "idle":
      self.activate_mode = True
      self.buffer_size=req.buffer_size
      self.mode = "buffer"
      return True

  def face_classification_continuously_srv_callback(self, req):
    '''
    Continuous mode service callback
      enable node to continous mode if possible (if is not already in buffer or continuous mode)
      disable node from continous mode if possible (if is not already disable) enabling dile mode
    '''

    if self.mode=="continuous" and req.enable:
      rospy.logwarn('Continuous_mode already selected')
      return False

    elif not self.mode=="continuous" and not req.enable:
      rospy.logwarn('Continuous_mode already turned off')
      return False

    elif self.mode=="buffer" and req.enable:
      rospy.logwarn('Node working on buffer mode, request for continuous mode not available')
      return False

    elif self.mode=="continuous" and not req.enable:
      self.image_subscriber.unregister()
      self.mode="idle"
      rospy.loginfo('%s mode activated' % self.mode)
      self.is_new_image = False
      return True

    else: #self.mode=="idle" and req.enable:
      self.activate_mode = True
      self.mode = "continuous"
      return True





  def pub_msgs(self,cv2_image,input_image_header,faces_coordinates,faces_genders,faces_emotions,image_format="passthrough"):
    '''
    Create the messages and publish it
      Arguments:
        cv2_image - image to publish as Image and CompressedImage
        faces_coordinates - list of faces coordinates [x y w h] of each face detected
        faces_genders - list of gender for each face
        faces_emotions - list of emotions for each face
        image_format - the encoding of the image data
    '''
    #time_stamp of published msgs
    time_stamp = rospy.Time.now()

    # Array of detected faces
    faces_array = FaceClassificationArray()
    faces_array.header.stamp = time_stamp
    faces_array.image_input_header = input_image_header


    # for each face create the msg and add it to the face_array
    for idx_face, face_coordinates in enumerate(faces_coordinates):
      new_face = FaceClassification()
      # face id equal to the number of face detected by default
      new_face.id=idx_face
      # face name by default
      new_face.name="face"+ str(idx_face)
      new_face.gender=faces_genders[idx_face]
      new_face.emotion=faces_emotions[idx_face]

      # create the bounding box
      new_face.bounding_box= RegionOfInterest()
      new_face.bounding_box.x_offset,new_face.bounding_box.y_offset,new_face.bounding_box.width,new_face.bounding_box.height=face_coordinates
      new_face.bounding_box.do_rectify=False

      #add the face to the array
      faces_array.faces.append(new_face)

    #publish the message on the topic
    self.publisher.publish(faces_array)

    #if there is some node subscribing the topic with output image, publish it
    if self.publisher_image.get_num_connections() > 0:
      try:
        msg = self.bridge.cv2_to_imgmsg(cv2_image, image_format)
        msg.header.stamp = time_stamp
      except CvBridgeError as e:
        rospy.logerr("Error on converting image for publishing: " + str(e) + " (Wrong image_format)")

      #publish the Image msg with the output
      self.publisher_image.publish(msg)

    #if there is some node subscribing the topic with output image compressed, publish it
    if self.publisher_image_compressed.get_num_connections() > 0:

      msg  = CompressedImage()
      msg.header.stamp = time_stamp
      msg.format = "jpeg"
      msg.data = np.array(cv2.imencode('.jpg', cv2_image)[1]).tostring()

      #publish the ImageCompressed msg with the output
      self.publisher_image_compressed.publish(msg)

    return


  def convert_img_to_cv2(self, image_msg):
    """
    Convert the image message into a cv2 image (numpy.ndarray)
    to be able to do OpenCV operations in it.
        image_msg: the message to transform
    """
    if self.image_type == CompressedImage:
      # Image to numpy array
      np_arr = np.fromstring(image_msg.data, np.uint8)
      # Decode to cv2 image and store
      return cv2.imdecode(np_arr, 1)
    elif self.image_type == Image:
      # Use CvBridge to transform
      try:
        return self.bridge.imgmsg_to_cv2(image_msg, image_msg.encoding)  # "bgr8"
      except CvBridgeError as e:
        rospy.logerr("Error when converting image: " + str(e))
        return None
    else:
      rospy.logerr("We don't know how to transform image of type " + str(type(image_msg)) + " to cv2 format.")
      return None

  def analyse_image(self):

    #convert image formate to CV2
    cv2_img = self.convert_img_to_cv2(self.last_image)

    #get the gray image and the rgb image
    gray_image = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

    #getting a list of detected faces in the gray image
    faces = detect_faces(self.face_detection, gray_image)

    # arrays with all the faces detected and classified (gender and emotion)
    genders=[]
    emotions=[]
    faces_classified=[]

    # for each face analyse it and classify it
    for face_coordinates in faces:

      #create a bigger bounding box where:
      #   (x1,y1) is the top left corner
      #   (x2,y2) is the bottom right corner
      x1, x2, y1, y2 = apply_offsets(face_coordinates, self.gender_offsets_percentage)
      rgb_face = rgb_image[y1:y2, x1:x2]

      x1, x2, y1, y2 = apply_offsets(face_coordinates, self.gender_offsets_percentage)
      gray_face = gray_image[y1:y2, x1:x2]

      # Resize the images to classify
      try:
        rgb_face = cv2.resize(rgb_face, (self.gender_target_size))
        gray_face = cv2.resize(gray_face, (self.emotion_target_size))
      except Exception as e:
        rospy.logwarn(e)
        continue

      gray_face = preprocess_input(gray_face, False)
      gray_face = np.expand_dims(gray_face, 0)
      gray_face = np.expand_dims(gray_face, -1)
      emotion_label_arg = np.argmax(self.emotion_classifier.predict(gray_face))
      emotion_text = self.emotion_labels[emotion_label_arg]

      rgb_face = np.expand_dims(rgb_face, 0)
      rgb_face = preprocess_input(rgb_face, False)
      gender_prediction = self.gender_classifier.predict(rgb_face)
      gender_label_arg = np.argmax(gender_prediction)
      gender_text = self.gender_labels[gender_label_arg]


      if gender_text == self.gender_labels[0]:
        color = (0, 0, 255)
      else:
        color = (255, 0, 0)

      # If no one is subscribing to the topics and the output image is to not be displayed, not even draw the bounding boxes
      if (self.publisher_image.get_num_connections() + self.publisher_image_compressed.get_num_connections()) > 0  or self.show_image:
        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, gender_text, color, 0, -20, 1, 1)
        draw_text(face_coordinates, rgb_image, emotion_text, color, 0, -45, 1, 1)

      faces_classified.append(face_coordinates)
      genders.append(gender_text)
      emotions.append(emotion_text)

    cv2_img = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    if self.show_image:
      cv2.imshow('window_frame', cv2_img)
      cv2.waitKey(3)

    if self.mode == "buffer":
      self.buffer_frames.add_frame(cv2_img,self.last_image.header,faces_classified,genders,emotions)

      self.buffer_size = self.buffer_size - 1
      if self.buffer_size == 0:
        self.image_subscriber.unregister()
        self.mode="idle"
        cv2_img,header,faces_classified,genders,emotions = self.buffer_frames.mode_frame()
        self.pub_msgs(cv2_img,header,faces_classified,genders,emotions)
        self.buffer_frames=None
        rospy.loginfo('%s mode activated' % self.mode)

    else: #self.mode == "continuous"

      self.pub_msgs(cv2_img,self.last_image.header,faces_classified,genders,emotions)


    self.is_new_image=False

  def run_process(self):

    #main loop
    while not rospy.is_shutdown():
      # if it's not in idle mode
      if not self.mode=="idle":

        # if it's the first time that the the node is activated subscribe the topic and wait till the camera image stabilize
        if self.activate_mode:
          rospy.loginfo('%s mode activated' % self.mode)
          self.image_subscriber = rospy.Subscriber(self.image_topic, self.image_type , self.image_subscriber_callback,  queue_size = 1)
          rospy.sleep(self.camera_delay)
          self.activate_mode=False
          if self.mode=="buffer":
            self.buffer_frames= BufferFrames()
        #if there's new image it will be analysed
        if self.is_new_image:
          self.analyse_image()

        #if the node is in buffer mode sleep in order to respect acquisition rate
        if self.mode=="buffer":
          self.image_acquisition_rate.sleep()

      self.node_rate.sleep()


def main():

  #Init node
  rospy.init_node('face_classifier', log_level=rospy.INFO)
  rospy.loginfo('Ready to receive requests or messages')

  mbot_classifier = FaceClassifier()
  mbot_classifier.run_process()

  #Shutting down
  rospy.loginfo("Shutting down")
  if mbot_classifier.show_image:
    cv2.destroyAllWindows()
