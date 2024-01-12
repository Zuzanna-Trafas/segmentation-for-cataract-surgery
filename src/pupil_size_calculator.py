import cv2
import numpy as np
import os
import re

class PupilSizeCalculator:
     def __init__(self, threshold):
          """
          percentage_threshold : measurement of a change from which it is counted as significant (in percentage)
          """
          self.percentage_threshold = threshold
          self.previous_pupil_size = None
          self.previous_file_path = None
          self.change_counter = 0
     

     ######### Getters and setters ######### 

     def reset_previous_pupil_size(self):
          self.previous_pupil_size = None
     
     def set_previous_pupil_size(self, size):
          self.previous_pupil_size = size
     
     def get_previous_pupil_size(self):
          return self.previous_pupil_size

     def set_previous_file_path(self, path):
          self.previous_file_path = path
     
     def get_previous_file_path(self):
          return self.previous_file_path
     
     def get_threshold(self):
          return self.percentage_threshold
     
     def get_change_counter(self):
          return self.change_counter

     def increase_counter(self):
          self.change_counter += 1

     ####################################### 

     def convert_image(self,image_path):
          """
          Reads the image, write all the pupil pixels to 1 and everything else to 0
          """
          label_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
          pupil_mask = (label_image == 0)
          highlighted_image = np.copy(label_image)
          highlighted_image[pupil_mask] = 1  
          highlighted_image[~pupil_mask] = 0
          return highlighted_image, pupil_mask
     
     def get_pupil_contour(self,recording):
          """
          Warning: the recording picture should show only the pupil
          If some instruments' segmentation cover the pupil, the contour of the pupil can be in multiple pieces.
          The function returns the (x,y,w,h) parameters for the merged contours.
          """
          contours, _ = cv2.findContours(recording.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
          if contours:
               leftmost = (recording.shape[1], 0)
               rightmost = (0, 0)
               topmost = (0, recording.shape[0])
               bottommost = (0, 0)

               # Go through each contour to find the overall leftmost, rightmost, topmost, and bottommost points
               for contour in contours:
                    lm = tuple(contour[contour[:, :, 0].argmin()][0])
                    rm = tuple(contour[contour[:, :, 0].argmax()][0])
                    tm = tuple(contour[contour[:, :, 1].argmin()][0])
                    bm = tuple(contour[contour[:, :, 1].argmax()][0])

                    if lm[0] < leftmost[0]:
                         leftmost = lm
                    if rm[0] > rightmost[0]:
                         rightmost = rm
                    if tm[1] < topmost[1]:
                         topmost = tm
                    if bm[1] > bottommost[1]:
                         bottommost = bm

               width = rightmost[0] - leftmost[0]
               height = bottommost[1] - topmost[1]

               bounding_rect = (leftmost[0], topmost[1], width, height)
          return bounding_rect

     def calculate_width(self,image_path):
          """
          Calculates the width of the masked image based on the contour, and the difference of the most right and most left pixels.
          Returns nones if pupil is not found, or not the full pupil is detected on the image.
          """
          recording, pupil_mask = self.convert_image(image_path)
          x,y,w,h = self.get_pupil_contour(recording)
          #if it is not near to a rounded shape
          if w < 0.80*h or h < 0.80*w:
               # if the edge of the bounding box is exactly on the edge of the image shape, the pupil is probably not full on the image
               if (x + w) == recording.shape[1] or (y + h) == recording.shape[0] or x == 0 or y == 0:
                    return None
          return w

     def save_pupil_mask(self, file_path, img_path):
          """
          Save the image for testing. The pixels, assigned to the pipil are red and everything else is black.
          """
          img, pupil_mask = self.convert_image(img_path)
          im = np.zeros((*img.shape, 3), dtype=np.uint8)
          im[img == 0] = [0,0,0]
          im[img == 1] = [0,0,255]

          x,y,w,h = self.get_pupil_contour(img)
          top_left = (x, y)
          bottom_right = (x + w, y + h)
          cv2.rectangle(im, top_left, bottom_right, (0, 255, 0), 2)

          if not cv2.imwrite(file_path, im):
               print("Saving highlighted image was unsuccessful")

     
     def has_significant_change(self, current_pupil_size, image_path):
          """
          Calculates the current width and compare it to the previous one.
          """
          previous_pupil_size = self.get_previous_pupil_size()
          if previous_pupil_size is None:
               self.set_previous_pupil_size(current_pupil_size)
               self.set_previous_file_path(image_path)
               return False
          change = abs(current_pupil_size - previous_pupil_size) / previous_pupil_size * 100
          self.set_previous_pupil_size(current_pupil_size)
          if change > self.get_threshold():
               #self.save_pupil_mask(f"pupil_change_{self.get_change_counter()}_start.png", self.get_previous_file_path())
               self.set_previous_file_path(image_path)
               #self.save_pupil_mask(f"pupil_change_{self.get_change_counter()}_end.png", self.get_previous_file_path())
               self.increase_counter()
               return True
          return False


def sort_key(filename):
     """
     sort files with regex based on the numbers in zfill() naming convention
     """
     match = re.search(r'frame(\d+)', filename)
     return int(match.group(1)) if match else 0


def process_folder(folder_path, threshold):
     print(f"Folder processing started...")
     calculator = PupilSizeCalculator(threshold)

     sorted_filenames = sorted(os.listdir(folder_path), key=sort_key)

     for filename in sorted_filenames:
          if filename.endswith(".png"):
               image_path = os.path.join(folder_path, filename)
               pupil_width= calculator.calculate_width(image_path)
               if pupil_width:
                    print(f"Width of Pupil: {pupil_width} in file: {image_path}")
                    if calculator.has_significant_change(pupil_width, image_path):
                         print(f"***** Significant width change detected in {filename} *****")
               else:
                    calculator.reset_previous_pupil_size()
                    print(f"Unable to detect a full pupil on the recording {filename}.")

process_folder("/home/data/CaDISv2/Video03/Labels", threshold=10)



     