import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re


def natural_key(string_):
    """See https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def find_continuous_sequences(arr):
    # Create a mask for elements not equal to -1
    mask_unequal_minus_one = np.array(arr != -1)

    # Use np.diff to find differences in the mask, prepend with True
    diff_arr = np.diff(mask_unequal_minus_one, prepend=[False])

    # Create a mask for positions where both conditions are true
    mask = np.logical_and(mask_unequal_minus_one, diff_arr)

    # Find the indices where the mask is True
    change_points = np.where(mask)[0]

    # Initialize a list to store the start and end indices of each sequence
    sequences = []

    # Iterate over each index of -1
    for idx in change_points:
        # Find the end index of the current sequence
        end_idx = idx - 1
        while end_idx >= 0 and arr[end_idx] != -1:
            end_idx -= 1

        # Find the start index of the current sequence
        start_idx = idx + 1
        while start_idx < len(arr) and arr[start_idx] != -1:
            start_idx += 1

        # Append the start and end indices to the list
        sequences.append((end_idx + 1, start_idx))

    return sequences


def handle_discontinuity(x, y):
    # This function will take all centroids coordinates WITH discontinuity.
    # Every time a "jump" is met, the coordinates will be separate in a new trajectory
    new_x = []
    new_y = []
    continuous_sequences_x = find_continuous_sequences(x)
    continuous_sequences_y = find_continuous_sequences(y)
    for seq_x, seq_y in zip(continuous_sequences_x, continuous_sequences_y):
        if seq_x == seq_y:
            start_ind, end_ind = seq_x
            new_x.append(x[start_ind: end_ind])
            new_y.append(y[start_ind: end_ind])
    return new_x, new_y


def tracking(tool_ind=0, path="/home/data/CaDISv2/Video01/Labels", output="/home/guests/nguyentoan_le/Praktikum",
             plot_name='Trajectory.png'):
    threshold_value = tool_ind
    
    # Load two images
    image_name = sorted(os.listdir(path), key=natural_key)
    images = [cv2.imread(os.path.join(path, name), cv2.IMREAD_GRAYSCALE) for name in image_name]
    height, width = images[0].shape

    # Initialize Centroids
    centroid_x = -1 * np.ones((len(images),))
    centroid_y = -1 * np.ones((len(images),))
    #imp = []
    for i, image in enumerate(images):
        # Apply threshold
        binary_image = np.zeros_like(image)
        binary_image[image == threshold_value] = 255

        # Calculate moments of binary image
        M = cv2.moments(binary_image)

        # Calculate x, y coordinate of center
        if M['m00'] != 0:
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])
            centroid_x[i] = x
            centroid_y[i] = y

        '''
        # Find contours in the binary mask
        contours_image, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours_image) >= 1:
            contours_image = contours_image[:1]

        for contour in contours_image:
            # Calculate centroid for each contour and print or use as needed
            moments = cv2.moments(contour)
            if moments['m00'] != 0:
                x = int(moments['m10'] / moments['m00'])
                y = int(moments['m01'] / moments['m00'])
                centroid_x[i] = x
                centroid_y[i] = y
                #imp.append(image_name[i])
                # Highlight centroid on the original image
                #cv2.circle(images[i], (x, y), 5, (255, 255, 255), -1)  # Draw a white circle
                #cv2.imwrite(os.path.join(output_path, image_name[i]), images[i])
                #cv2.circle(binary_image1, (centroid_x, centroid_y), 5, (255, 255, 255), -1)  # Draw a white circle
        '''
    '''
    for i, (x, y) in enumerate(zip(centroid_x, centroid_y)):
        if x != -1:
            print(i, x, y)
    '''
    #print(imp)
    centroid_x, centroid_y = handle_discontinuity(centroid_x, centroid_y)

    fig, ax = plt.subplots()
    for x, y in zip(centroid_x, centroid_y):
        ax.plot(x, y, 'b-')
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.xaxis.tick_top()
    ax.invert_yaxis()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Trajectory of tool '+ str(threshold_value))
    fig.savefig(os.path.join(output, plot_name), dpi=300)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--object_id",
        help="Object to track [eg. for Pupil, enter --image_id 0]",
        required=True,
    )
    parser.add_argument("--segmentation", help="Path to folder where segmented images are saved", required=True)
    parser.add_argument("--output", help="Output path to save plot", required=True)
    parser.add_argument("--name_trajectory", help="Name of the plot", required=True)
    args = parser.parse_args()

    assert os.path.isdir(args.segmentation), f"{args.segmentation} directory does not exist"

    if int(args.object_id) != -1:
        tracking(tool_ind=int(args.object_id), path=args.segmentation, output=args.output,
                 plot_name=args.name_trajectory)
    else:
        for i in range(36):
            s = args.name_trajectory.split(".")
            s[0] = s[0] + str(i)
            name = ".".join(s)
            tracking(tool_ind=int(i), path=args.segmentation, output=args.output,
                     plot_name=name)
