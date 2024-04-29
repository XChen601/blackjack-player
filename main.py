import os
import cv2
import pyautogui
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import time

class CardChecker:
    def __init__(self):
        self.clusters = {}
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
            return []

        main_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

        # Define the scale range and the scaling factor
        scale_range = np.linspace(0.7, 1.5, 7)  # Example range from 0.5x to 1.5x original size
        match_results = []

        for scale in scale_range:
            # Resize the main image according to the current scale
            resized_main = cv2.resize(main_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(resized_main, template_gray, cv2.TM_CCOEFF_NORMED)
            threshold = 0.86
            loc = np.where(res >= threshold)
            if loc[0].size > 0:
                for pt in zip(*loc[::-1]):
                    match_results.append((pt[0] / scale, pt[1] / scale))

        # Convert points to numpy array
        points = np.array(match_results)

        if points.size == 0:
            return []

        # Use DBSCAN to cluster points
        clustering = DBSCAN(eps=10, min_samples=1).fit(points)
        labels = clustering.labels_
        unique_labels = set(labels)

        count = 0
        found_clusters = []

        # Draw rectangles based on the clusters
        for label in unique_labels:
            points_in_cluster = points[labels == label]
            x_min, y_min = np.min(points_in_cluster, axis=0)
            x_max, y_max = np.max(points_in_cluster, axis=0)
            width, height = template_image.shape[1], template_image.shape[0]
            top_left = (int(x_min), int(y_min))
            bottom_right = (int(x_max + width), int(y_max + height))
            cv2.rectangle(main_image, top_left, bottom_right, (0, 255, 0), 2)
            count += 1
            found_clusters.append((x_min, y_min, x_max + width, y_max + height))

        # Convert color back to RGB for displaying in matplotlib
        main_image = cv2.cvtColor(main_image, cv2.COLOR_BGR2RGB)

        # Display using matplotlib
        plt.imshow(main_image)
        plt.title('Matched Image')
        plt.show()

        # print(f"{template_image_path} appears {count} times.")
        return found_clusters


    def iou(self, boxA, boxB):
        # Debugging: print boxes before processing
        # print("Calculating IoU for:", boxA, "and", boxB)

        # Validate input boxes
        if len(boxA) != 4 or len(boxB) != 4:
            raise ValueError("Both boxes must have exactly four elements (x_min, y_min, x_max, y_max).")

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea)
        print(iou)
        return iou

    def get_cards(self, main_image_path, image_directory):
        self.clusters = {}
        for filename in os.listdir(image_directory):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                image_path = os.path.join(image_directory, filename)
                new_clusters = self.count_appearance(main_image_path, image_path)
                card_name = filename.split('.')[0]

                for cluster in new_clusters:
                    replace = False
                    for existing_cluster, existing_name in list(self.clusters.items()):
                        if self.iou(cluster, existing_cluster) > 0.3:
                            print('replacing')
                            replace = True
                            # remove old 3 cluster
                            del self.clusters[existing_cluster]
                            # add 8 cluster
                            self.clusters[cluster] = card_name


                    if not replace:
                        self.clusters[cluster] = card_name
        print(self.clusters)
        cards = []
        for card in self.clusters.values():
            cards.append(card)
        return cards

class BlackjackPlayer:
    def __init__(self):
        self.game_coordinates = [(2360, 778), (3740, 1541)]
        self.card_checker = CardChecker()
        self.move_counts = 400

    def get_bj_image(self, upper_left_x, upper_left_y, lower_right_x, lower_right_y):
        width = lower_right_x - upper_left_x
        height = lower_right_y - upper_left_y

        # Take screenshot of the calculated region
        screenshot = pyautogui.screenshot(region=(upper_left_x, upper_left_y, width, height))

        # Save the screenshot
        screenshot.save("cards.png")
        self.split_image_horizontally('cards.png', 'dealer.png', 'player.png')

    def split_image_horizontally(self, image_path, save_path1, save_path2, split_percentage=0.4):
        # Open the image
        img = Image.open(image_path)

        # Calculate the dimensions to split the image by percentage
        width, height = img.size
        split_point = int(height * split_percentage)  # Calculate the height for the split

        # Split the image into top and bottom parts at the split point
        top_part = img.crop((0, 0, width, split_point))
        bottom_part = img.crop((0, split_point, width, height))

        # Save each part
        top_part.save(save_path1)
        bottom_part.save(save_path2)

    def start_game(self):
        for i in range(self.move_counts):
            self.get_bj_image(self.game_coordinates[0][0], self.game_coordinates[0][1],
                              self.game_coordinates[1][0], self.game_coordinates[1][1])

            self.make_move()
    def make_move(self):
        # check if respin or no action
        print('checking for respin/insurance')
        if self.is_round_end():
            self.click_image('actions/respin.png')
            print('Respin')
            return
        if self.is_insurance():
            self.click_image('actions/no.png')
            print('No')
            return
        print('finish check')
        dealer_hand = self.card_checker.get_cards('dealer.png', 'cards')
        print(dealer_hand)
        if len(dealer_hand) != 1:
            return
        player_hand = self.card_checker.get_cards('player.png', 'cards')
        print('dealer hand:', dealer_hand)
        print('player hand:', player_hand)
        if len(dealer_hand) == 0 or len(player_hand) < 2:
            print('error getting cards')
            return


        move = self.determine_move(player_hand, dealer_hand)
        print(move)

        self.click_move(move)
            # after clicking split, while loop to check right hand while
        if move == 'Split':
            self.play_split(dealer_hand)
            self.play_split(dealer_hand, right=False)
    def click_move(self, move):
        if move == 'Hit':
            self.click_image('actions/hit.png')
        if move == 'Stand':
            self.click_image('actions/stand.png')
        if move == 'Double':
            self.click_image('actions/double.png')
        if move == 'Split':
            self.click_image('actions/split.png')
            print('split triggered')
            time.sleep(9999999)
    def get_split(self, player_image_path):
        img = Image.open(player_image_path)
        width, height = img.size
        split_point = int(width * .48)

        left_part = img.crop((0, 0, split_point, height))
        right_part = img.crop((split_point, 0, width, height))

        left_part.save('left_hand.png')
        right_part.save('right_hand.png')

    def play_split(self, dealer_hand, right=True):
        if not right:
            print('playing left')
        print('playing split')
        if right:
            img = 'right_hand.png'
        else:
            img = 'left_hand.png'
        self.get_bj_image(self.game_coordinates[0][0], self.game_coordinates[0][1],
                          self.game_coordinates[1][0], self.game_coordinates[1][1])
        self.get_split('player.png')
        arrow = self.find_template_in_image(img, 'arrow.png', threshold=0.7)
        print(arrow)
        # while arrow in image
        while arrow:
            # get current image, check left and sides, if both left and right has number, its split
            self.get_bj_image(self.game_coordinates[0][0], self.game_coordinates[0][1],
                              self.game_coordinates[1][0], self.game_coordinates[1][1])
            self.get_split('player.png')
            if right:
                hand = self.card_checker.get_cards('right_hand.png', "cards")
            else:
                hand = self.card_checker.get_cards("left_hand.png", "cards")
            print(hand)
            move = self.determine_move(hand, dealer_hand)
            print(move)
            self.click_move(move)
            arrow = self.find_template_in_image(img, 'arrow.png', threshold=0.7)



    def find_template_in_image(self, main_image_path, template_image_path, threshold=.8):
        # Load the main image and template image
        main_image = cv2.imread(main_image_path)
        template = cv2.imread(template_image_path)

        # Convert images to grayscale
        main_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # Perform template matching
        result = cv2.matchTemplate(main_gray, template_gray, cv2.TM_CCOEFF_NORMED)

        # Get the location of the best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Calculate the center of the found template
        if max_val >= threshold:
            center_x = max_loc[0] + template.shape[1] // 2
            center_y = max_loc[1] + template.shape[0] // 2
            return center_x, center_y
        else:
            return None

    def get_region(self):
        width = self.game_coordinates[1][0] - self.game_coordinates[0][0]
        height = self.game_coordinates[1][1] - self.game_coordinates[0][1]
        region = (self.game_coordinates[0][0], self.game_coordinates[0][1], width, height)
        return region

    def click_image(self, image_path):
        # limit search region so it faster
        region = self.get_region()

        curr_location = pyautogui.position()
        location = pyautogui.locateCenterOnScreen(image_path, region=region, confidence=0.7)
        pyautogui.click(location)
        pyautogui.moveTo(curr_location[0], curr_location[1])


    def is_round_end(self):
        region = self.get_region()
        location = pyautogui.locateCenterOnScreen('actions/respin.png', region=region, confidence=0.8)
        if location:
            return True
        return False

    def is_insurance(self):
        region = self.get_region()
        location = pyautogui.locateCenterOnScreen('actions/no.png', region=region, confidence=0.8)

        if location:
            return True
        return False

    def determine_move(self, player_hand, dealer_hand, can_split=True):
        def calculate_total(hand):
            soft = False
            total = 0
            ace_count = 0
            for card in hand:
                if card in ['j', 'q', 'k']:
                    total += 10
                elif card == 'a':
                    ace_count += 1
                    total += 11  # Initially consider Ace as 11
                else:
                    total += int(card)
            if ace_count > 0 and total < 21:
                soft = True
            # Adjust Aces from 11 to 1 if total is over 21
            while total > 21 and ace_count:
                total -= 10
                ace_count -= 1
            return total, soft

        # Calculate totals
        player_total, soft = calculate_total(player_hand)
        dealer_total, _ = calculate_total(dealer_hand)

        # conditions for double
        if ((len(player_hand) == 2) and
                (((3 <= dealer_total <= 6) and player_total == 9)
                or ((2 <= dealer_total <= 9) and player_total == 10)
                or ((2 <= dealer_total <= 10) and player_total == 11)
                or ((5 <= dealer_total <= 6) and player_total in (13, 14) and soft)
                or ((4 <= dealer_total <= 6) and player_total in (15, 16) and soft)
                or ((3 <= dealer_total <= 6) and player_total in (17, 18) and soft))):
            return 'Double'
        # conditions for splitting
        if (len(player_hand) == 2) and (player_hand[0] == player_hand[1] and can_split):
            card = player_hand[0]
            if (card in ['a', '8']
                    or (card in ("2", "3") and dealer_total <= 7)
                    or card == '4' and dealer_total in [5, 6]
                    or (card == "6" and dealer_total <= 6)
                    or (card == "7" and dealer_total <= 7)
                    or (card == "9" and (dealer_total <= 6 or dealer_total in (8, 9)))):
                return 'Split'
        # conditions for Hit
        if (player_total <= 11
                or (player_total == 12 and dealer_total in (2, 3))
                or (player_total <= 16 and dealer_total >= 7)
                or (player_total <= 17 and soft)
                or (player_total == 18 and soft and dealer_total >= 9)):
            return 'Hit'

        return 'Stand'



blackjack_player = BlackjackPlayer()
blackjack_player.start_game()


