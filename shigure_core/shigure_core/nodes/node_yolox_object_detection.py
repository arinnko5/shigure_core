import datetime

import cv2
import numpy as np
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, CompressedImage
from shigure_core_msgs.msg import DetectedObjectList, DetectedObject, BoundingBox
from bbox_ex_msgs.msg import BoundingBoxes

from shigure_core.enum.detected_object_action_enum import DetectedObjectActionEnum
from shigure_core.nodes.common_model.timestamp import Timestamp
from shigure_core.nodes.node_image_preview import ImagePreviewNode
from shigure_core.nodes.yolox_object_detection.color_image_frame import ColorImageFrame
from shigure_core.nodes.yolox_object_detection.color_image_frames import ColorImageFrames
from shigure_core.nodes.yolox_object_detection.frame_object import FrameObject
from shigure_core.nodes.yolox_object_detection.judge_params import JudgeParams
from shigure_core.nodes.yolox_object_detection.logic import ObjectDetectionLogic

class YoloxObjectDetectionNode(ImagePreviewNode):
  object_list: list
  def __init__(self):
    super().__init__("yolox_object_detection_node")
    # QoS Settings
    shigure_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
    
    # publisher, subscriber
    self.detection_publisher = self.create_publisher(
      DetectedObjectList, 
      '/shigure/object_detection', 
      10
    )
    yolox_bbox_subscriber = message_filters.Subscriber(
      self, 
      Boundingboxes,
      '/bounding_boxes',
      qos_profile = shigure_qos
    )
    color_subscriber = message_filters.Subscriber(
      self, 
      CompressedImage,
      '/rs/color/compressed', 
      qos_profile = shigure_qos
    )
    depth_camera_info_subscriber = message_filters.Subscriber(
      self, 
      CameraInfo,
      '/rs/aligned_depth_to_color/cameraInfo', 
      qos_profile=shigure_qos
    )  
    
    
    self.time_synchronizer = message_filters.TimeSynchronizer(
      [yolox_bbox_subscriber, color_subscriber, depth_camera_info_subscriber], 1000)
    self.time_synchronizer.registerCallback(self.callback)
    
    self.yolox_object_detection_logic = ObjectDetectionLogic()
    
    self.frame_object_list: List[FrameObject] = []
    self._color_img_buffer: List[np.ndarray] = []
    self._color_img_frames = ColorImageFrames()
    self._buffer_size = 90
    
    
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
