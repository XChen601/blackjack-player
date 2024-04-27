import os
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

class CardChecker:
    def convert_to_bw(self, image_path):
        with Image.open(image_path) as img:
            bw_img = img.convert('L')  # Convert image to grayscale

            bw_img.save(image_path.replace('.jpg', '_bw.jpg'))  # Save the black and white image
            return image_path.replace('.jpg', '_bw.jpg')

    def count_appearance(self, main_image_path, template_image_path):
        main_image = cv2.imread(main_image_path)
        template_image = cv2.imread(template_image_path)

        if main_image is None or template_image is None:
            print("Error loading images.")
            return

        main_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

        # Apply template matching
        res = cv2.matchTemplate(main_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8  # Define a threshold for how good the match should be
        loc = np.where(res >= threshold)
        points = np.column_stack((loc[1], loc[0]))  # Switch x and y coordinates

        if points.size == 0:
            return 0
        # Use DBSCAN to cluster points
        clustering = DBSCAN(eps=10, min_samples=1).fit(points)
        labels = clustering.labels_
        unique_labels = set(labels)

        count = 0

        # # Draw rectangles based on the clusters
        # for label in unique_labels:
        #     points_in_cluster = points[labels == label]
        #     x_min, y_min = np.min(points_in_cluster, axis=0)
        #     x_max, y_max = np.max(points_in_cluster, axis=0)
        #     width, height = template_image.shape[1], template_image.shape[0]
        #     top_left = (x_min, y_min)
        #     bottom_right = (x_max + width, y_max + height)
        #     cv2.rectangle(main_image, top_left, bottom_right, (0, 255, 0), 2)
        #     count += 1
        #
        # # Convert color back to RGB for displaying in matplotlib
        # main_image = cv2.cvtColor(main_image, cv2.COLOR_BGR2RGB)
        #
        # # Display using matplotlib
        # plt.imshow(main_image)
        # plt.title('Matched Image')
        # plt.show()
        #
        # print(f"{template_image_path} appears {count} times with {res.max()} confidence")
        return count

    def card_appearances(self, main_image_path, image_directory):
        card_counts = {}
        for filename in os.listdir(image_directory):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                # Construct the full file path
                image_path = os.path.join(image_directory, filename)

                count = self.count_appearance(main_image_path, image_path)
                card_counts[filename] = count

        print(card_counts)
        return card_counts


card_checker = CardChecker()
# Specify the main image and directory
main_image_path = "player_hand.png"
image_directory = "cards"

# Run the function
card_checker.card_appearances(main_image_path, image_directory)
