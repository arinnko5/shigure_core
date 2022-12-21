import datetime

import cv2
import message_filters
from typing import List
import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from shigure_core_msgs.msg import DetectedObjectList, DetectedObject#, BoundingBox, BoundingBoxes, YoloxBoundingBox
from bbox_ex_msgs.msg import BoundingBoxes

from shigure_core.enum.detected_object_action_enum import DetectedObjectActionEnum
from shigure_core.nodes.common_model.timestamp import Timestamp
from shigure_core.nodes.node_image_preview import ImagePreviewNode
from shigure_core.nodes.yolox_object_detection.color_image_frame import ColorImageFrame
from shigure_core.nodes.yolox_object_detection.color_image_frames import ColorImageFrames
from shigure_core.nodes.yolox_object_detection.frame_object import FrameObject
from shigure_core.nodes.yolox_object_detection.judge_params import JudgeParams
from shigure_core.nodes.yolox_object_detection.logic import YoloxObjectDetectionLogic

class YoloxObjectDetectionNode(ImagePreviewNode):
	object_list: list
	def __init__(self):
		super().__init__("yolox_object_detection_node")
		# QoS Settings
		shigure_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
		
		# ros params
		is_debug_mode_descriptor = ParameterDescriptor(type=ParameterType.PARAMETER_BOOL,description='If true, run debug mode.')
		self.declare_parameter('is_debug_mode', False, is_debug_mode_descriptor)
		self.is_debug_mode: bool = self.get_parameter("is_debug_mode").get_parameter_value().bool_value
		
		# publisher, subscriber
		self.detection_publisher = self.create_publisher(
			DetectedObjectList, 
			'/shigure/object_detection', 
			10
		)
		yolox_bbox_subscriber = message_filters.Subscriber(
			self, 
			BoundingBoxes,
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
		
		self.yolox_object_detection_logic = YoloxObjectDetectionLogic()
		
		self.frame_object_list: List[FrameObject] = []
		self._color_img_buffer: List[np.ndarray] = []
		self._color_img_frames = ColorImageFrames()
		self._buffer_size = 90
		
		self._judge_params = JudgeParams(5)
		
		self.object_index = 0
		
	def callback(self, yolox_bbox_src: BoundingBoxes, color_img_src: CompressedImage, camera_info: CameraInfo):
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
		frame_object_dict = self.object_detection_logic.execute(yolox_bbox_src, timestamp,color_img,self.frame_object_list,self._judge_params)
		
		self.frame_object_list = list(chain.from_iterable(frame_object_dict.values()))
		
		#result_img = cv2.cvtColor(subtraction_analysis_img, cv2.COLOR_GRAY2BGR)
		
		if self._color_img_frames.is_full():
			self.get_logger().info('Buffering end', once=True)
			
			frame = self._color_img_frames.top_frame
			
			sec, nano_sec = frame.timestamp.timestamp
			detected_object_list = DetectedObjectList()
			detected_object_list.header.stamp.sec = sec
			detected_object_list.header.stamp.nanosec = nano_sec
			detected_object_list.header.frame_id = camera_info.header.frame_id
			
			timestamp_str = str(frame.timestamp)
			if timestamp_str in frame_object_dict.keys():
				frame_object_list = frame_object_dict.get(timestamp_str)
				
				# 検知開始時間が同じのフレームオブジェクトが検知済でなければ警告
				if not all([frame_object.is_finished() for frame_object in frame_object_list]):
					self.get_logger().warning('検知が終了していないオブジェクトを含んでいます')
					
				detected_object_list = self.create_msg(frame_object_list, detected_object_list, frame)
			
			self.detection_publisher.publish(detected_object_list)
			
			if self.is_debug_mode:
				for i, bbox in enumerate(yolox_bboxes):
					x = bbox.xmin
					y = bbox.ymin
					xmax = bbox.xmax
					ymax = bbox.ymax
					height = ymax - y
					width = xmax - x
					class_id = bbox.class_id
					if i != 0 and class_id != person:
						color = self._colors[i % 255]
						result_img = cv2.rectangle(color_img, (x, y), (xmax, ymax), color, thickness=3)
						brack_img = np.zeros_like(color_img)
						img = self.print_fps(brack_img)
						tile_img = cv2.hconcat([result_img, img])
						cv2.namedWindow('yolox_object_detection', cv2.WINDOW_NORMAL)
						cv2.imshow("yolox_object_detection", tile_img)
						cv2.waitKey(1)
			else:
				print(f'[{datetime.datetime.now()}] fps : {self.fps}', end='\r')
				
	def  create_msg(self, frame_object_list: List[FrameObject], detected_object_list: DetectedObjectList, frame: ColorImageFrame) -> DetectedObjectList:
		for frame_object in frame_object_list:
			action, bounding_box_src, size, mask_img, _ = frame_object.item.items
			x, y, width, height = bounding_box_src.items
			
			detected_object = DetectedObject()
			detected_object.action = action.value
			detected_object.mask = self.bridge.cv2_to_compressed_imgmsg(mask_img, 'png')
			
			bounding_box = BoundingBox()
			bounding_box.x = float(x)
			bounding_box.y = float(y)
			bounding_box.width = float(width)
			bounding_box.height = float(height)
			detected_object.bounding_box = bounding_box
			
			detected_object_list.object_list.append(detected_object)
			
			self.frame_object_list.remove(frame_object)
			
			if self.is_debug_mode:
				item_color_img = frame.new_image if action == DetectedObjectActionEnum.BRING_IN else frame.old_image
				print('オブジェクトが検出されました(',
					f'action: {action.value}, x: {x}, y: {y}, width: {width}, height: {height}, size: {size})')
				icon = np.zeros((height + 10, width, 3), dtype=np.uint8)
				icon[0:height, 0:width, :] = item_color_img[y:y + height, x:x + width, :]
				
				img_height, img_width = item_color_img.shape[:2]
				icon = cv2.resize(icon.copy(), (img_width // 2, img_height // 2))
				cv2.putText(icon, f'Action : {action.value}', (0, img_height // 2 - 5), cv2.FONT_HERSHEY_PLAIN, 1.5,
					(255, 255, 255), thickness=2)
				
				self.object_list[self.object_index] = icon
				self.object_index = (self.object_index + 1) % 4
				
			return detected_object_list
      
def main(args=None):
	rclpy.init(args=args)
	
	yolox_object_detection_node = YoloxObjectDetectionNode()
	
	try:
		rclpy.spin(yolox_object_detection_node)
		
	except KeyboardInterrupt:
		pass
		
	finally:
		print()
		# 終了処理
		yolox_object_detection_node.destroy_node()
		rclpy.shutdown()
        
      
if __name__ == '__main__':
	main()
