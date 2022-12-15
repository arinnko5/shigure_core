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
    
    self._judge_params = JudgeParams(5)
    
    self.object_index = 0
    
    def callback(self, yolox_bbox_src: Boundingboxes, color_img_src: CompressedImage, camera_info: CameraInfo):
      self.get_logger().info('Buffering start', once=True)
      self.frame_count_up()
      
      color_img: np.ndarray = self.bridge.compressed_imgmsg_to_cv2(color_img_src)
      height, width = color_img.shape[:2]
      if not hasattr(self, 'object_list'):
        self.object_list = []
        black_img = np.zeros_like(color_img)
        for i in range(4):
          self.object_list.append(cv2.resize(black_img.copy(), (width // 2, height // 2)))
       
      if len(self._color_img_buffer) > 30:
        self._color_img_buffer = self._color_img_buffer[1:]
        self._color_img_frames.get(-30).new_image = color_img
      self._color_img_buffer.append(color_img)
      
      timestamp = Timestamp(color_img_src.header.stamp.sec, color_img_src.header.stamp.nanosec)
      frame = ColorImageFrame(timestamp, self._color_img_buffer[0], color_img)
      self._color_img_frames.add(frame)
      frame_object_dict = self.object_detection_logic.execute(yolox_bbox_src, timestamp,
                                                                self.frame_object_list,self._judge_params)
      
      self.frame_object_list = list(chain.from_iterable(frame_object_dict.values()))
      
      
    
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
