import os
import random

import cv2
import pyautogui
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import time
from humancursor import SystemCursor

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
        scale_range = np.linspace(0.8, 1.5, 10)  # Example range from 0.5x to 1.5x original size
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

        # # Display using matplotlib
        # plt.imshow(main_image)
        # plt.title('Matched Image')
        # plt.show()

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

        cards = []
        for card in self.clusters.values():
            cards.append(card)
        return cards

class BlackjackPlayer:
    def __init__(self):
        self.antibot = False
        self.cursor = SystemCursor()
        self.game_coordinates = [(0, 0), (0, 0)]
        self.card_checker = CardChecker()
        self.rounds = 400
        self.curr_round = 0

        self.board_img_path = "game_images/full_board.png"
        self.dealer_img_path = "game_images/dealer.png"
        self.player_img_path = "game_images/player.png"


    def start_game(self):
        self.get_board_coordinates()

        while self.curr_round < self.rounds:
            print('Round:', self.curr_round)
            self.update_game_images()
            self.play_round()

    def play_round(self):
        # check if respin or no insurance btn
        if self.click_image('actions/respin.png'):
            print('Respin | round + 1')
            self.curr_round += 1
            time.sleep(3.5)
            return
        if self.click_image('actions/no.png'):
            print('No insurance')
        dealer_hand = self.card_checker.get_cards(self.dealer_img_path, 'cards')
        print('dealer hand:', dealer_hand)
        if len(dealer_hand) != 1:
            return
        player_hand = self.card_checker.get_cards(self.player_img_path, 'cards')
        print('player hand:', player_hand)
        if len(dealer_hand) == 0 or len(player_hand) < 2:
            print('error getting cards')
            return

        move = self.determine_move(player_hand, dealer_hand)
        print(move)
        if move == 'Split':
            self.click_move(move)
            self.play_split(dealer_hand)
            time.sleep(2)
            self.play_split(dealer_hand, right=False)
        else:
            self.click_move(move)


    def get_board_coordinates(self):
        top_left_coords = self.find_image_on_screen("board_top_left.png")['top_left']
        bottom_right_coords = self.find_image_on_screen("board_bottom_right.png")['bottom_right']
        self.game_coordinates = [top_left_coords, bottom_right_coords]
        print("game coordinates:", self.game_coordinates)

    # finds an image on screen, in a certain region, or inside another image
    def find_image_on_screen(self, image_path, main_image_path=None, region=None, threshold=0.75):
        # Load the image from the path
        template = cv2.imread(image_path, 0)
        if template is None:
            raise ValueError("Image not found at the specified path")

        # Capture the screen or a specific region
        if region:
            screen = pyautogui.screenshot(region=region)
        else:
            screen = pyautogui.screenshot()

        if main_image_path:
            screen = cv2.imread(main_image_path)

        screen_np = np.array(screen)
        screen_gray = cv2.cvtColor(screen_np, cv2.COLOR_BGR2GRAY)

        scale_range = np.linspace(0.7, 1.5, 15)
        for scale in scale_range:
            resized_template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(screen_gray, resized_template, cv2.TM_CCOEFF_NORMED)

            if np.max(res) >= threshold:
                loc = np.where(res >= threshold)
                for pt in zip(*loc[::-1]):  # Switch x and y positions
                    # Calculate coordinates
                    top_left_x = pt[0]
                    top_left_y = pt[1]
                    bottom_right_x = pt[0] + resized_template.shape[1]
                    bottom_right_y = pt[1] + resized_template.shape[0]
                    center_x = pt[0] + resized_template.shape[1] // 2
                    center_y = pt[1] + resized_template.shape[0] // 2

                    # Adjust coordinates if a region was specified
                    if region:
                        top_left_x += region[0]
                        top_left_y += region[1]
                        bottom_right_x += region[0]
                        bottom_right_y += region[1]
                        center_x += region[0]
                        center_y += region[1]

                    # Return coordinates in a dictionary
                    return {
                        'top_left': (top_left_x, top_left_y),
                        'center': (center_x, center_y),
                        'bottom_right': (bottom_right_x, bottom_right_y)
                    }

        return {
            'top_left': None,
            'center': None,
            'bottom_right': None
        } # Return None if no matching area is found at any scale

    def update_game_images(self):
        cut_side_ratio = 0
        top_left_x = self.game_coordinates[0][0] * (1 + cut_side_ratio)
        top_left_y = self.game_coordinates[0][1]
        bottom_right_x = self.game_coordinates[1][0] * (1 - cut_side_ratio)
        bottom_right_y = self.game_coordinates[1][1]
        width = bottom_right_x - top_left_x
        height = bottom_right_y - top_left_y

        # Take screenshot of the calculated region
        screenshot = pyautogui.screenshot(region=(top_left_x, top_left_y, width, height))

        # Save the screenshot
        screenshot.save(self.board_img_path)
        self.save_player_dealer_images(self.board_img_path)

    def save_player_dealer_images(self, game_image_path, split_percentage=0.4):
        img = Image.open(game_image_path)

        # Calculate the dimensions to split the image by percentage
        width, height = img.size
        split_point = int(height * split_percentage)  # Calculate the height for the split

        # Split the image into top and bottom parts at the split point
        top_part = img.crop((0, 0, width, split_point))
        bottom_part = img.crop((0, split_point, width, height))

        # Save each part
        top_part.save(self.dealer_img_path)
        bottom_part.save(self.player_img_path)

    def click_move(self, move):
        if move == 'Hit':
            self.click_image('actions/hit.png')
        if move == 'Stand':
            self.click_image('actions/stand.png')
            return
        if move == 'Double':
            self.curr_round += 1
            self.click_image('actions/double.png')
        if move == 'Split':
            if self.click_image('actions/split.png'):
                self.curr_round += 1
            else:
                print('split error')
                time.sleep(15)

        time.sleep(1)  # ensures the cards are updated before updating image

    def update_splitted_hands(self, player_image_path):
        img = Image.open(player_image_path)
        width, height = img.size
        split_point = int(width * .47)

        left_part = img.crop((0, 0, split_point, height))
        right_part = img.crop((split_point, 0, width, height))

        left_part.save('game_images/left_hand.png')
        right_part.save('game_images/right_hand.png')


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


    def play_split(self, dealer_hand, right=True):
        if right:
            img = 'game_images/right_hand.png'
        else:
            img = 'game_images/left_hand.png'
        print(img)
        self.update_game_images()
        self.update_splitted_hands(self.player_img_path)
        arrow = self.find_image_on_screen('actions/arrow.png', img, threshold=0.85)['center']
        print(arrow)

        hand = self.card_checker.get_cards(img, "cards")
        while len(hand) > 2:
            print(' error with getting cards ')
            time.sleep(1)
        # while arrow in image
        while arrow:
            # get current image, check left and sides, if both left and right has number, its split
            if right:
                hand = self.card_checker.get_cards(img, "cards")
            else:
                hand = self.card_checker.get_cards(img, "cards")
            print(hand)
            move = self.determine_move(hand, dealer_hand, can_split=False)
            print(move)

            self.click_move(move)
            if move == 'Double' or move=='Stand':
                break
            time.sleep(1)
            self.update_game_images()
            self.update_splitted_hands(self.player_img_path)
            arrow = self.find_image_on_screen('actions/arrow.png', img, threshold=0.85)['center']
            print(arrow)

    def get_region(self):
        width = self.game_coordinates[1][0] - self.game_coordinates[0][0]
        height = self.game_coordinates[1][1] - self.game_coordinates[0][1]
        region = (self.game_coordinates[0][0], self.game_coordinates[0][1], width, height)
        return region

    def click_image(self, image_path):
        # limit search region so it faster
        region = self.get_region()

        location = self.find_image_on_screen(image_path, region=region)['center']
        curr_location = pyautogui.position()
        if not location:
            return False

        if self.antibot:
            random_x = location[0] + random.randint(-30, 30)
            random_y = location[1] + random.randint(-30, 30)
            self.cursor.move_to([random_x, random_y])
            pyautogui.click(random_x, random_y)
            self.cursor.move_to([curr_location[0], curr_location[1]])

        else:
            pyautogui.click(location)
            time.sleep(.1)
            pyautogui.moveTo(curr_location[0], curr_location[1])

        return True

    def is_round_end(self):
        region = self.get_region()
        location = self.find_image_on_screen('actions/respin.png', region=region)['center']
        if location:
            return True
        return False

    def is_insurance(self):
        region = self.get_region()
        location = self.find_image_on_screen('actions/no.png', region=region)['center']

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


