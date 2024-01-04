import cv2
import numpy as np
import os

class PupilSizeCalculator:
     def __init__(self, threshold):
          """
          percentage_threshold : measurement of a change from which it is counted as significant (in percentage)
          """
          self.percentage_threshold = threshold
          self.previous_pupil_size = None
          self.previous_file_path = None
          self.change_counter = 0
     
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
     
     def calculate_width(self,image_path):
          """
          Calculates the width of the masked image based on the contour, and the difference of the most right and most left pixels.
          In this way, when an instrument covers some parts of the pupil, and those pixels are not defined are part of the pupil, the width calculation will be right.
          """
          recording, pupil_mask = self.convert_image(image_path)

          contours, _ = cv2.findContours(pupil_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
          if contours:
               largest_contour = max(contours, key=cv2.contourArea)

               leftmost = tuple(largest_contour[largest_contour[:,:,0].argmin()][0])
               rightmost = tuple(largest_contour[largest_contour[:,:,0].argmax()][0])
               
               width = rightmost[0] - leftmost[0]
               
               return width
          else:
               return None

     def calculate_width_with_sum(self,image_path):
          """
          Calculates the width based on only summing the pixels in the row where there are the most pixels defined as pupil parts.
          """
          recording, mask = self.convert_image(image_path)
          max_pixels_row = np.argmax(np.sum(mask, axis=1))
          width_of_area_px = np.sum(mask[max_pixels_row])
          return width_of_area_px
     
     def calculate_area(self, image_path):
          """
          Calculates the full area of the pupil (not instrument handling ready)
          """
          recording, mask = self.convert_image(image_path)
          pupil_size = np.sum(mask)
          return pupil_size

     def save_pupil_mask(self, file_path, img_path):
          """
          Save the image for testing. The pixels, assigned to the pipil are red and everything else is black.
          """
          img, pupil_mask = self.convert_image(img_path)
          im = np.zeros((*img.shape, 3), dtype=np.uint8)
          im[img == 0] = [0,0,0]
          im[img == 1] = [0,0,255]

          if not cv2.imwrite(file_path, im):
               print("Saving highlighted image was unsuccessful")

     
     def has_significant_change(self, current_pupil_size, image_path):
          """
          Calculates the current width and compare it to the previous one.
          """
          if self.previous_pupil_size is None:
               self.previous_pupil_size = current_pupil_size
               self.previous_file_path = image_path
               return False
          change = abs(current_pupil_size - self.previous_pupil_size) / self.previous_pupil_size * 100
          self.previous_pupil_size = current_pupil_size
          if change > self.percentage_threshold:
               #self.save_pupil_mask(f"pupil_change_{self.change_counter}_start.png", self.previous_file_path)
               self.previous_file_path = image_path
               #self.save_pupil_mask(f"pupil_change_{self.change_counter}_end.png", self.previous_file_path)
               self.change_counter += 1
               return True
          return False

def process_folder(folder_path, threshold):
     print(f"Folder processing started...")
     calculator = PupilSizeCalculator(threshold)
     for filename in os.listdir(folder_path):
          if filename.endswith(".png"):
               image_path = os.path.join(folder_path, filename)
               pupil_size = calculator.calculate_width(image_path)
               print(f"Width of Pupil: {pupil_size} in file: {image_path}")
               if calculator.has_significant_change(pupil_size, image_path):
                    print(f"***** Significant width change detected in {filename} *****")

process_folder("/home/data/CaDISv2/Video01/Labels", threshold=15)

"""
# Find the row with the most pixels with the mask
max_pixels_row = np.argmax(np.sum(pupil_mask, axis=1))
width_of_area_px = np.sum(pupil_mask[max_pixels_row])

# Find the column with the most pixels with the mask
max_pixels_column = np.argmax(np.sum(pupil_mask, axis=0))
height_of_area_px = np.sum(pupil_mask[:, max_pixels_column])

#pupil mask area
pupil_size = np.sum(pupil_mask)
img_size = label_image.shape[0] * label_image.shape[1]
"""


     