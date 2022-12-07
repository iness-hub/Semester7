import os
import settings
import cv2
import numpy as np



def save_upload_image(fileObj):
    filename = fileObj.filename    
    name , ext = filename.split('.')    
    save_filename = 'upload.'+ext
    upload_image_path = settings.join_path(settings.SAVE_DIR, 
                                           save_filename)
    
    fileObj.save(upload_image_path)
    
    return upload_image_path


# def array_to_json_format(numpy_array):
#     points = []
#     for pt in numpy_array.tolist():
#         points.append({'x':pt[0],'y':pt[1]})
        
#     return point