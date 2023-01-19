from typing import Tuple

import numpy as np

from shigure_core.enum.detected_object_action_enum import DetectedObjectActionEnum
from shigure_core.nodes.common_model.timestamp import Timestamp
from shigure_core.nodes.common_model.bounding_box import BoundingBox


class BboxObject:

    def __init__(self,  bounding_box:BoundingBox, size:int, mask_img:np.ndarray, found_at:Timestamp,class_id:str,found_count:int, not_found_count:int):
        self._bounding_box = bounding_box
        self._size = size
        self._mask = mask_img
        self._found_at = found_at
        self._class_id = class_id
        self._found_count = 0
        self._not_found_count = 0
        
        self.not_match_count = 0

    def is_match(self, other):
    	bbox_x = abs(self._bounding_box._x - other._bounding_box._x)
    	bbox_y = abs(self._bounding_box._y - other._bounding_box._y)
    	bbox_width = abs(self._bounding_box._width - other._bounding_box._width)
    	bbox_height = abs(self._bounding_box._height - other._bounding_box._height)
    	if (self._class_id==other._class_id)and(bbox_x < 100) and (bbox_y < 100): #& bbox_width < 30 & bbox_height < 30:
    		
    		return True
    	else:
    		
    		return False
    		
    def add_found_count(self):
    	self._found_count += 1
    	
    def add_not_found_count(self):
    	self._not_found_count += 1
    	
    def reset_found_count(self):
    	self._found_count == 0
    
    def reset_not_found_count(self):
    	self._not_found_count == 0
    	
    #def not_mach(self, List):
    	#if self.not_match_count > len(List):
    		#return True
            
    def is_found(self):
    	if self._found_count >= 10:
    		return True
    
    def is_not_found(self):
    	if self._not_found_count >= 100:
    		return True
    		


