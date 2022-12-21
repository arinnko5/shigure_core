from collections import defaultdict
from typing import List, Dict, Tuple

import cv2
import numpy as np

#from shigure_core_msgs.msg import BoundingBoxes,YoloxBoundingBox
from bboxes_ex_msgs.msg import BoundingBoxes

from shigure_core.enum.detected_object_action_enum import DetectedObjectActionEnum
from shigure_core.nodes.common_model.timestamp import Timestamp
from shigure_core.nodes.common_model.union_find_tree import UnionFindTree
from shigure_core.nodes.common_model.bounding_box import BoundingBox
from shigure_core.nodes.yolox_object_detection.frame_object import FrameObject
from shigure_core.nodes.yolox_object_detection.frame_object_item import FrameObjectItem
from shigure_core.nodes.yolox_object_detection.judge_params import JudgeParams


class YoloxObjectDetectionLogic:
    """物体検出ロジッククラス"""

    @staticmethod
    def execute(yolox_bbox: BoundingBoxes, started_at: Timestamp, color_img:np.ndarray, frame_object_list: List[FrameObject],
                judge_params: JudgeParams) -> Dict[str, List[FrameObject]]:
        """
        物体検出ロジック

        :param yolox_bbox:
        :param started_at:
        :param frame_object_list:
        :param judge_params:
        :return: 検出したObjectリスト, 更新された既知マスク
        """
        # ラベリング処理
        
        prev_frame_object_dict = {}
        frame_object_item_list = []
        union_find_tree: UnionFindTree[FrameObjectItem] = UnionFindTree[FrameObjectItem]()

        result = defaultdict(list)

        # 検知が終了しているものは除外
        for frame_object in frame_object_list:
            if frame_object.is_finished():
                frame_object.item.action = DetectedObjectActionEnum.TAKE_OUT
                #frame_object.item.mask = 
                result[str(frame_object.item.detected_at)].append(frame_object)
            else:
                prev_frame_object_dict[frame_object.item] = frame_object
        frame_object_list = list(prev_frame_object_dict.values())
        
        yolox_bboxes = yolox_bbox.bounding_boxes


        for i, bbox in enumerate(yolox_bboxes):
            x = bbox.xmin
            y = bbox.ymin
            xmax = bbox.xmax
            ymax = bbox.ymax
            height = ymax - y
            width = xmax - x
            class_id = bbox.class_id

            if i != 0 and class_id != 'person':
                brack_img = np.zeros_like(color_img)
                brack_img[y:y + height, x:x + width] = 255
                mask_img:np.ndarray = brack_img[y:y + height, x:x + width]

                action = DetectedObjectActionEnum.BRING_IN

                bounding_box = BoundingBox(x, y, width, height)
                area = width*height
                
                item = FrameObjectItem(action, bounding_box, area, mask_img, started_at)
                frame_object_item_list.append(item)

                for prev_item, frame_object in prev_frame_object_dict.items():
                    is_matched, size = prev_item.is_match(item)
                    if is_matched:
                        if not union_find_tree.has_item(prev_item):
                            union_find_tree.add(prev_item)
                            frame_object_list.remove(frame_object)
                        if not union_find_tree.has_item(item):
                            union_find_tree.add(item)
                            frame_object_item_list.remove(item)
                        union_find_tree.unite(prev_item, item)

        # リンクした範囲を1つにまとめる
        groups = union_find_tree.all_group_members().values()
        for items in groups:
            new_item: FrameObjectItem = items[0]
            mask_img = YoloxObjectDetectionLogic.update_mask_image(np.zeros(color_img.shape[:2]),
                                                              new_item)
            for item in items[1:]:
                new_item, mask_img = ObjectDetectionLogic.update_item(new_item, item, mask_img)

            result[str(new_item.detected_at)].append(FrameObject(new_item, judge_params.allow_empty_frame_count))

        # リンクしなかったframe_objectは空のフレームを挟む
        for frame_object in frame_object_list:
            frame_object.add_empty_frame()
            result[str(frame_object.item.detected_at)].append(frame_object)

        # リンクしなかったframe_object_itemは新たなframe_objectとして登録
        for frame_object_item in frame_object_item_list:
            frame_object = FrameObject(frame_object_item, judge_params.allow_empty_frame_count)
            result[str(frame_object_item.detected_at)].append(frame_object)

        return result

    @staticmethod
    def update_item(left: FrameObjectItem, right: FrameObjectItem,
                    mask_img: np.ndarray) -> Tuple[FrameObjectItem, np.ndarray]:
        x = min(left.bounding_box.x, right.bounding_box.x)
        y = min(left.bounding_box.y, right.bounding_box.y)
        width = max(left.bounding_box.x + left.bounding_box.width,
                    right.bounding_box.x + right.bounding_box.width) - x
        height = max(left.bounding_box.y + left.bounding_box.height,
                     right.bounding_box.y + right.bounding_box.height) - y
        mask_img = ObjectDetectionLogic.update_mask_image(mask_img, right)
        size = np.count_nonzero(mask_img[y:y + height, x:x + width])

        new_bounding_box = BoundingBox(x, y, width, height)

        action = left.action
        left_is_before = left.detected_at.is_before(right.detected_at)

        # 持ち込み時は新しい方を選択
        new_detected_at = left.detected_at if left_is_before else right.detected_at

        return FrameObjectItem(action, new_bounding_box, size, mask_img[y:y + height, x:x + width],
                               new_detected_at), mask_img

    @staticmethod
    def update_mask_image(mask_img: np.ndarray, item: FrameObjectItem) -> np.ndarray:
        _, bounding_box, _, mask, _ = item.items
        x, y, width, height = bounding_box.items
        mask_img[y:y + height, x:x + width] = np.where(mask > 0, mask, mask_img[y:y + height, x:x + width])
        return mask_img
