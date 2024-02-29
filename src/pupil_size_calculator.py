import cv2
import numpy as np
import os
import re
import argparse

class PupilSizeCalculator:
     def __init__(self, threshold):
          """
          percentage_threshold : measurement of a change from which it is counted as significant (in percentage)
          """
          self.percentage_threshold = threshold
          self.previous_pupil_size = None
          self.previous_file_path = None
          self.change_counter = 0
          self.processed_counter = 0
     

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

     def increase_change_counter(self):
          self.change_counter += 1

     def get_processed_counter(self):
          return self.processed_counter

     def increase_processed_counter(self):
          self.processed_counter += 1

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
               edge_points = (leftmost, topmost, rightmost, bottommost)
          return bounding_rect, edge_points

     def calculate_width(self,image_path):
          """
          Calculates the width of the masked image based on the contour, and the difference of the most right and most left pixels.
          Returns nones if pupil is not found, or not the full pupil is detected on the image.
          """
          recording, pupil_mask = self.convert_image(image_path)
          (x,y,w,h),(leftmost, topmost, rightmost, bottommost) = self.get_pupil_contour(recording)
          #if it is not near to a rounded shape
          if w < 0.80*h or h < 0.80*w:
               # if the edge of the bounding box is exactly on the edge of the image shape, the pupil is probably not full on the image
               if (x + w) == recording.shape[1] or (y + h) == recording.shape[0] or x == 0 or y == 0:
                    return None
          return w

     def save_pupil_mask(self, file_path, img_path, width, color):
          """
          Save the image for testing. The pixels, assigned to the pipil are red and everything else is black.
          """
          img, pupil_mask = self.convert_image(img_path)
          im = np.zeros((*img.shape, 3), dtype=np.uint8)
          im[img == 0] = [0,0,0]
          if color == (0,0,255):
               im[img == 1] = [0,0,255]
               point_color = (255,255,255)
          else:
               im[img == 1] = [255,255,255]
               point_color=color
               

          (x,y,w,h),(leftmost, topmost, rightmost, bottommost) = self.get_pupil_contour(img)
          top_left = (x, y)
          bottom_right = (x + w, y + h)
          #cv2.rectangle(im, top_left, bottom_right, (0, 255, 0), 2)

          font = cv2.FONT_HERSHEY_SIMPLEX 
          fontScale = 0.5
          thickness = 1
          cv2.putText(im, str(width), (img.shape[1] - 50, img.shape[0] - 30), font,  fontScale, color, thickness, cv2.LINE_AA)


          for point in (leftmost, topmost, rightmost, bottommost):
               radius = 1
               cv2.circle(im, point, radius, point_color, thickness=5)

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
          
          abs_diff = abs(current_pupil_size - previous_pupil_size)
          average_value = (current_pupil_size + previous_pupil_size) / 2
          change = (abs_diff / average_value) * 100
          self.set_previous_pupil_size(current_pupil_size)
          self.set_previous_file_path(image_path)
          if change > self.get_threshold():
               #self.save_pupil_mask(f"pupil_change_{self.get_change_counter()}_start.png", self.get_previous_file_path(), previous_pupil_size, (0,0,255))
               #self.set_previous_file_path(image_path)
               self.increase_change_counter()
               current_processed_counter = self.get_processed_counter()
               current_change_counter = self.get_change_counter()
               filename = os.path.join(args.output_folder, f"pupil_{current_processed_counter}_change_{current_change_counter}.png")
               self.save_pupil_mask(filename, self.get_previous_file_path(), current_pupil_size,(0,0,255))
               self.increase_processed_counter()
               return True
          else:
               current_processed_counter = self.get_processed_counter()
               current_change_counter = self.get_change_counter()
               filename = os.path.join(args.output_folder,f"pupil_{current_processed_counter}_change_{current_change_counter}.png")
               self.save_pupil_mask(filename, self.get_previous_file_path(), current_pupil_size, (0,255,0))
               self.increase_processed_counter()
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
     print(len(sorted_filenames))

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


parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_folder",
    help="Folder name where the labeled pictures are",
    default="/home/data/CaDISv2/Video01/Labels",
)
parser.add_argument(
    "--output_folder",
    help="Folder name where the pictures should be produced",
    default="/home/guests/dominika_darabos/segmentation-for-cataract-surgery/samples/pupil_size",
    type=str,
)

parser.add_argument(
    "--change_threshold",
    help="The percentage from which a change is defined as significant",
    default=30,
    type=int,
)
args = parser.parse_args()
process_folder(args.input_folder, threshold=args.change_threshold)



     