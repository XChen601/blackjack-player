import os
from PIL import Image


def convert_to_black_and_white(directory, threshold=128):
    # Check if the directory exists
    if not os.path.exists(directory):
        print("The specified directory does not exist.")
        return

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            # Construct the full file path
            file_path = os.path.join(directory, filename)

            # Open the image
            with Image.open(file_path) as img:
                # Convert the image to grayscale
                bw_img = img.convert('L')

                # Save the new black and white image, overwriting or to a new file
                bw_img.save(os.path.join(directory, f"bw_{filename}"))

                print(f"Converted {filename} to black and white and saved as bw_{filename}")


# Specify the directory containing images
image_directory = "cards"

# Call the function
convert_to_black_and_white(image_directory)
