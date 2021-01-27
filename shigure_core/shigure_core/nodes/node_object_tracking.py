import cv2
import message_filters
import numpy as np
import rclpy
from sensor_msgs.msg import CompressedImage, CameraInfo
from shigure_core_msgs.msg import DetectedObjectList, TrackedObjectList, TrackedObject, Cube

from shigure_core.nodes.node_image_preview import ImagePreviewNode
from shigure_core.nodes.object_tracking.logic import ObjectTrackingLogic
from shigure_core.nodes.object_tracking.tracking_info import TrackingInfo
from shigure_core.util import compressed_depth_util


class ObjectTrackingNode(ImagePreviewNode):

    def __init__(self):
        super().__init__('object_tracking_node')

        self._publisher = self.create_publisher(TrackedObjectList, '/shigure/object_tracking', 10)

        depth_subscriber = message_filters.Subscriber(self, CompressedImage,
                                                      '/rs/aligned_depth_to_color/compressedDepth')
        depth_camera_info_subscriber = message_filters.Subscriber(self, CameraInfo,
                                                                  '/rs/aligned_depth_to_color/cameraInfo')
        object_detection_subscriber = message_filters.Subscriber(self, DetectedObjectList, '/shigure/object_detection')

        if not self.is_debug_mode:
            self.time_synchronizer = message_filters.TimeSynchronizer(
                [depth_subscriber, object_detection_subscriber, depth_camera_info_subscriber], 30000)
            self.time_synchronizer.registerCallback(self.callback)
        else:
            color_subscriber = message_filters.Subscriber(self, CompressedImage, '/rs/color/compressed')
            self.time_synchronizer = message_filters.TimeSynchronizer(
                [depth_subscriber, object_detection_subscriber, depth_camera_info_subscriber, color_subscriber], 400000)
            self.time_synchronizer.registerCallback(self.callback_debug)

        self.people_tracking_logic = ObjectTrackingLogic()

        self._tracking_info = TrackingInfo()

    def callback(self, depth_src: CompressedImage, detected_object_list: DetectedObjectList,
                 camera_info: CameraInfo):
        self.frame_count_up()

        depth_img = compressed_depth_util.convert_compressed_depth_img_to_cv2(depth_src)
        depth_img: np.ndarray = depth_img.astype(np.float32)

        # 焦点距離取得
        #     [fx  0 cx]
        # K = [ 0 fy cy]
        #     [ 0  0  1]
        k = camera_info.k.reshape((3, 3))

        self._tracking_info = ObjectTrackingLogic.execute(depth_img, detected_object_list, self._tracking_info)

        # publish
        publish_msg = TrackedObjectList()
        publish_msg.header = detected_object_list.header

        k_inv = np.linalg.inv(k)
        height, width = depth_img.shape[:2]
        for object_id, item in self._tracking_info.object_dict.items():
            stay_object, bounding_box = item

            tracked_object = TrackedObject()
            tracked_object.object_id = object_id
            tracked_object.action = stay_object.action
            tracked_object.bounding_box = stay_object.bounding_box

            bounding_box = stay_object.bounding_box
            left = min(int(bounding_box.x), width - 1)
            top = min(int(bounding_box.y), height - 1)
            right = min(int(bounding_box.x + bounding_box.width), width - 1)
            bottom = min(int(bounding_box.y + bounding_box.height), height - 1)

            depth_min = depth_img[top:bottom, left:right].min()
            depth_max = depth_img[top:bottom, left:right].max()

            s1 = np.asarray([[bounding_box.x, bounding_box.y, 1]]).T
            s2 = np.asarray([[bounding_box.x + bounding_box.width,
                              bounding_box.y + bounding_box.height, 1]]).T

            m1 = (depth_img[top, left] * np.matmul(k_inv, s1)).T
            m2 = (depth_img[bottom, right] * np.matmul(k_inv, s2)).T

            collider = Cube()
            collider.x, collider.y = m1[0, 0], m1[0, 1]
            collider.width, collider.height = m2[0, 0] - m1[0, 0], m2[0, 1] - m1[0, 1]
            collider.z = float(depth_min)
            collider.depth = float(depth_max - depth_min)
            tracked_object.collider = collider

            publish_msg.tracked_object_list.append(tracked_object)

        self._publisher.publish(publish_msg)

    def callback_debug(self, depth_src: CompressedImage, detected_object_list: DetectedObjectList,
                       camera_info: CameraInfo, color_src: CompressedImage):
        color_img: np.ndarray = self.bridge.compressed_imgmsg_to_cv2(color_src)

        self.callback(depth_src, detected_object_list, camera_info)

        height, width = color_img.shape[:2]
        for object_id, item in self._tracking_info.object_dict.items():
            stay_object, bounding_box = item

            bounding_box = stay_object.bounding_box
            left = min(int(bounding_box.x), width - 1)
            top = min(int(bounding_box.y), height - 1)
            right = min(int(bounding_box.x + bounding_box.width), width - 1)
            bottom = min(int(bounding_box.y + bounding_box.height), height - 1)

            cv2.rectangle(color_img, (left, top), (right, bottom), (0, 0, 255), thickness=1)
            cv2.putText(color_img, f'ID : {object_id} ({stay_object.action})', (left, top), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 255, 0))

        self.print_fps(color_img)
        cv2.imshow('Result', color_img)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)

    object_tracking_node = ObjectTrackingNode()

    try:
        rclpy.spin(object_tracking_node)

    except KeyboardInterrupt:
        pass

    finally:
        # 終了処理
        object_tracking_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
