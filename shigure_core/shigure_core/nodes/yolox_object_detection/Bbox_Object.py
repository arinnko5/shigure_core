from typing import Tuple

import numpy as np

from shigure_core.enum.detected_object_action_enum import DetectedObjectActionEnum
from shigure_core.nodes.common_model.timestamp import Timestamp
from shigure_core.nodes.common_model.bounding_box import BoundingBox


class BboxObject:

    def __init__(self,  bounding_box:BoundingBox, size:int, mask_img:np.ndarray, started_at:Timestamp,class_id:str,found_count:int, not_found_count:int):
        self._bounding_box = bounding_box
        self._size = size
        self._mask = mask_img
        self._started_at = started_at
        self._class_id = class_id
        self._found_count = 0
        self._not_found_count = 0
        
        self.not_match_count = 0

    def is_match(self, other):
    	bbox_x = abs(self._bounding_box._x - other._bounding_box._x)
    	bbox_y = abs(self._bounding_box._y - other._bounding_box._y)
    	bbox_width = abs(self._bounding_box._width - other._bounding_box._width)
    	bbox_height = abs(self._bounding_box._height - other._bounding_box._height)
    	if (self._class_id==other._class_id)& bbox_x < 30 & bbox_y < 30: #& bbox_width < 30 & bbox_height < 30:
    		self._found_count += 1
    		self._started_at = other._started_at
    		return True
    	else:
    		other.not_match_count += 1
    		self._not_found_count += 1
    		return False
    		
    	
    def not_mach(self, List):
    	if self.not_match_count > len(List):
    		return True
            
    
    def found_count_is(self):
    	if self._found_count==5:
    		self._found_count = 0
    		self._not_found_count = 0
    		return True
    		
    
    def not_found_count_is(self):
    	if self._not_found_count >= 500:
    		return True
    		
