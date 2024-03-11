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
          self.previous_label_path = None
          self.change_counter = 0
          self.processed_counter = 0
     

     ######### Getters and setters ######### 

     def reset_previous_pupil_size(self):
          self.previous_pupil_size = None
     
     def set_previous_pupil_size(self, size):
          self.previous_pupil_size = size
     
     def get_previous_pupil_size(self):
          return self.previous_pupil_size

     def set_previous_label_path(self, path):
          self.previous_label_path = path
     
     def get_previous_label_path(self):
          return self.previous_label_path
     
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

     def convert_image(self,label_image):
          """
          Reads the image, write all the pupil pixels to 1 and everything else to 0
          """
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
          bounding_rect = (0,0,0,0)
          edge_points = ((0, 0), (0, 0), (0, 0), (0, 0))
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

     def calculate_width(self,img):
          """
          Calculates the width of the masked image based on the contour, and the difference of the most right and most left pixels.
          Returns nones if pupil is not found, or not the full pupil is detected on the image.
          """
          recording, pupil_mask = self.convert_image(img)
          (x,y,w,h),(leftmost, topmost, rightmost, bottommost) = self.get_pupil_contour(recording)
          #if it is not near to a rounded shape
          if w < 0.80*h or h < 0.80*w:
               # if the edge of the bounding box is exactly on the edge of the image shape, the pupil is probably not full on the image
               if (x + w) == recording.shape[1] or (y + h) == recording.shape[0] or x == 0 or y == 0:
                    print("Only the part of the pupil is visible.")
                    return None
          return w

     def save_pupil_mask(self, file_path, img_path, width, color):
          """
          Save the image for testing. The pixels, assigned to the pipil are red and everything else is black.
          """
          label_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
          img, pupil_mask = self.convert_image(label_image)
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
     
     def save_original_with_pupil_size(self, out_file_path, orig_img_path, label_img_path, width, color):
          """
          Save the image for testing. The pixels, assigned to the pipil are red and everything else is black.
          """
          label_image = cv2.imread(label_img_path, cv2.IMREAD_GRAYSCALE)
          img, pupil_mask = self.convert_image(label_image)
          orig_im = cv2.imread(orig_img_path)
          if orig_im is None:
               print(f"Error: Unable to load image from {orig_img_path}")
               return
          print("SHAPE OF ORIGINAL: ", orig_im.shape)
          im = np.zeros((*img.shape, 3), dtype=np.uint8)
          im[img == 0] = [0,0,0]

          (x,y,w,h),(leftmost, topmost, rightmost, bottommost) = self.get_pupil_contour(img)
          #top_left = (x, y)
          #bottom_right = (x + w, y + h)
          #cv2.rectangle(im, top_left, bottom_right, (0, 255, 0), 2)
          """
          # if the shapes are the same

          font = cv2.FONT_HERSHEY_SIMPLEX 
          fontScale = 1
          thickness = 1
          cv2.putText(orig_im, str(width), (img.shape[1] - 50, img.shape[0] - 30), font,  fontScale, color, thickness, cv2.LINE_AA)

          for point in (leftmost, topmost, rightmost, bottommost):
               radius = 1
               cv2.circle(orig_im, point, radius, color, thickness=8)
          """

          # scale the two images
          img_height, img_width = img.shape[:2]
          orig_height, orig_width = orig_im.shape[:2]

          width_scale = orig_width / img_width
          height_scale = orig_height / img_height

          font = cv2.FONT_HERSHEY_SIMPLEX 
          fontScale = 1
          thickness = 2
          background_color = (40,40,110)
          text = str(width)

          text_size, _ = cv2.getTextSize(text, font, fontScale, thickness)
          text_position = (orig_width - 200, orig_height - 100)
          background_position = (text_position[0] - 5, text_position[1] - text_size[1] - 10)
          background_size = (text_size[0] + 10, text_size[1] + 20)

          # Draw background rectangle
          cv2.rectangle(orig_im, background_position, (background_position[0] + background_size[0], background_position[1] + background_size[1]), background_color, cv2.FILLED)
          cv2.putText(orig_im, text, text_position, font, fontScale, color, thickness, cv2.LINE_AA)

          transformed_points = [(int(point[0] * width_scale), int(point[1] * height_scale)) for point in (leftmost, topmost, rightmost, bottommost)]

          # Draw circles on orig_im around the transformed points
          radius = 3
          for point in transformed_points:
               cv2.circle(orig_im, point, radius, color, thickness=8)


          if not cv2.imwrite(out_file_path, orig_im):
               print("Saving highlighted image was unsuccessful")
     
     def save_comparison(self, out_file_path, orig_img_path, label_img_path, width, color):
          """
          Save the image for testing. The pixels, assigned to the pipil are red and everything else is black.
          """
          label_image = cv2.imread(label_img_path, cv2.IMREAD_GRAYSCALE)
          img, pupil_mask = self.convert_image(label_image)
          orig_im = cv2.imread(orig_img_path)
          if orig_im is None:
               print(f"Error: Unable to load image from {orig_img_path}")
               return
          print("SHAPE OF ORIGINAL: ", orig_im.shape)
          im = np.zeros((*img.shape, 3), dtype=np.uint8)
          im[img == 0] = [0,0,0]
          (x,y,w,h),(leftmost, topmost, rightmost, bottommost) = self.get_pupil_contour(img)


          if color == (0,0,255):
               im[img == 1] = [0,0,255]
               point_color = (255,255,255)
          else:
               im[img == 1] = [255,255,255]
               point_color=color

          # scale the two images
          img_height, img_width = img.shape[:2]
          orig_height, orig_width = orig_im.shape[:2]
          width_scale = orig_width / img_width
          height_scale = orig_height / img_height

          font = cv2.FONT_HERSHEY_SIMPLEX 
          fontScale = 1
          thickness = 2
          background_color = (40,40,110)
          text = str(width)
          text_size, _ = cv2.getTextSize(text, font, fontScale, thickness)
          text_position = (orig_width - 200, orig_height - 100)
          background_position = (text_position[0] - 5, text_position[1] - text_size[1] - 10)
          background_size = (text_size[0] + 10, text_size[1] + 20)
          # Draw background rectangle
          cv2.rectangle(orig_im, background_position, (background_position[0] + background_size[0], background_position[1] + background_size[1]), background_color, cv2.FILLED)
          cv2.putText(orig_im, text, text_position, font, fontScale, (255,255,255), thickness, cv2.LINE_AA)
          fontScale = 0.6
          thickness = 1
          cv2.putText(im, str(width), (img.shape[1] - 100, img.shape[0] - 50), font,  fontScale, color, thickness, cv2.LINE_AA)

          transformed_points = [(int(point[0] * width_scale), int(point[1] * height_scale)) for point in (leftmost, topmost, rightmost, bottommost)]
          radius = 1
          for point in (leftmost, topmost, rightmost, bottommost):
               cv2.circle(im, point, radius, point_color, thickness=5)
          radius = 3
          for point in transformed_points:
               cv2.circle(orig_im, point, radius, (255,255,255), thickness=8)


          spacing = 50
          max_height = max(orig_im.shape[0], im.shape[0])
          im_resized = cv2.resize(im, (orig_im.shape[1], orig_im.shape[0]))
          total_width = orig_im.shape[1] + im_resized.shape[1] + spacing * 3
          total_height = max_height + spacing * 2
          combined_image = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255
          combined_image[spacing:orig_im.shape[0]+spacing, spacing:orig_im.shape[1]+spacing, :] = orig_im
          combined_image[spacing:im_resized.shape[0]+spacing, 
                              orig_im.shape[1]+spacing*2:orig_im.shape[1]+spacing*2+im_resized.shape[1], :] = im_resized
          cv2.imwrite(out_file_path, combined_image)
     
     def has_significant_change(self, current_pupil_size, orig_image_path, label_image_path):
          """
          Calculates the current width and compare it to the previous one.
          """
          previous_pupil_size = self.get_previous_pupil_size()
          if previous_pupil_size is None:
               self.set_previous_pupil_size(current_pupil_size)
               self.set_previous_label_path(label_image_path)
               return False
          
          abs_diff = abs(current_pupil_size - previous_pupil_size)
          average_value = (current_pupil_size + previous_pupil_size) / 2
          change = (abs_diff / average_value) * 100
          self.set_previous_pupil_size(current_pupil_size)
          self.set_previous_label_path(label_image_path)

          if change > self.get_threshold():
               self.increase_change_counter()
               color = (0,0,255)
               return True
          else:
               color = (0,255,0)
               
          current_processed_counter = self.get_processed_counter()
          current_change_counter = self.get_change_counter()
          filename = os.path.join(args.output_folder, f"pupil_{current_processed_counter}_change_{current_change_counter}.png")
          filename_comp = os.path.join(args.output_folder, f"pupil_{current_processed_counter}_comparison.png")
          if args.purpose == "mask":
               self.save_pupil_mask(filename, self.get_previous_label_path(), current_pupil_size,color)
          elif args.purpose == "original":
               self.save_original_with_pupil_size(filename, orig_image_path, self.get_previous_label_path(), current_pupil_size,(255,255,255))
          elif args.purpose == "comparison":
               self.save_comparison(filename_comp, orig_image_path, self.get_previous_label_path(), current_pupil_size,color)
          else:
               print("Invalid purpose.")
               return
          self.increase_processed_counter()
          if change > self.get_threshold():
               return True
          else:
               return False



def sort_key(filename):
     """
     sort files with regex based on the numbers in zfill() naming convention
     """
     match = re.search(r'frame(\d+)', filename)
     return int(match.group(1)) if match else 0


def process_folder(label_folder_path, orig_folder_path, threshold):
     print(f"Folder processing started...")
     calculator = PupilSizeCalculator(threshold)

     sorted_filenames = sorted(os.listdir(label_folder_path), key=sort_key)

     for filename in sorted_filenames:
          if filename.endswith(".png"):
               label_image_path = os.path.join(label_folder_path, filename)
               name, ext = os.path.splitext(filename)
               # Then, change the extension to .jpg
               new_filename = name + '.png'
               orig_image_path = os.path.join(orig_folder_path, new_filename)
               label_image = cv2.imread(label_image_path, cv2.IMREAD_GRAYSCALE)
               pupil_width= calculator.calculate_width(label_image)
               if pupil_width:
                    print(f"Width of Pupil: {pupil_width} in file: {label_image_path}")
                    if calculator.has_significant_change(pupil_width, orig_image_path, label_image_path):
                         print(f"***** Significant width change detected in {filename} *****")
               else:
                    calculator.reset_previous_pupil_size()
                    print(f"Unable to detect a full pupil on the recording {filename}.")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_label_folder",
    help="Folder name where the labeled pictures are",
    default="/home/data/CaDISv2/Video01/Labels",
)
parser.add_argument(
    "--input_original_folder",
    help="Folder name where the original pictures are",
    default="/home/data/CaDISv2/Video01/Images",
)
parser.add_argument(
    "--output_folder",
    help="Folder name where the pictures should be produced",
    default="/home/guests/dominika_darabos/segmentation-for-cataract-surgery/samples/pupil_size/cadis_video_01",
    type=str,
)

parser.add_argument(
    "--change_threshold",
    help="The percentage from which a change is defined as significant",
    default=30,
    type=int,
)

parser.add_argument(
    "--purpose",
    help="Options: mask, original, comparison",
    type=str,
)

if __name__ == "__main__":
     args = parser.parse_args()
     process_folder(args.input_label_folder, args.input_original_folder, threshold=args.change_threshold)



     