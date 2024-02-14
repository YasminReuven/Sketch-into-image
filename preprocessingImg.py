import os
import shutil
import cv2
import numpy as np
import hashlib
from utils import connect_x_y
import torch

def organize_images_in_folder(sources_f: [], destination_f: str, counter: int):
    """
    Takes all the images that are in the source folders and moves them to the destination folder.
     In addition, the function changes the names of the images to a serial number according to the counter.
    :param sources_f:
    :param destination_f:
    :param counter:
    :return: Returns the serial number of the next image.
    """
    source_folders = sources_f
    destination_folder = destination_f
    counter = counter
    for folder in source_folders:
        for filename in os.listdir(folder):
            if os.path.isfile(os.path.join(folder, filename)):
                file_extension = os.path.splitext(filename)[1].lower()
                if file_extension in ['.jpg', '.jpeg', 'png']:
                    new_filename = str(counter) + file_extension
                    shutil.move(os.path.join(folder, filename), os.path.join(destination_folder, new_filename))
                    counter += 1
                    if int(counter) % int(10) == 0:
                        print(counter)
    return counter


def photo_into_sketch(image_path):
    """
    Takes an image path and turn the image into a black and white sketch.
    :param image_path:
    :return: None
    """
    # Load the image
    image = cv2.imread(image_path)
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    # Find contours in the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw the contours on the original image
    outlined_image = np.ones_like(image) * 255
    cv2.drawContours(outlined_image, contours, -1, (0, 0, 0), 2)
    # Save the outlined  in a new directory with the same name as the original
    output_directory = "./XXX"
    output_filename = os.path.splitext(os.path.basename(image_path))[0] + "_sketch.jpg"
    output_path = os.path.join(output_directory, output_filename)
    cv2.imwrite(output_path, outlined_image)
    
def delete_duplicate_files(folder):
    # Dictionary to store the file names and their corresponding paths
    file_hashes = {}

    # Iterate through all files in the folder
    for root, dirs, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)

            # Read the contents of the file in binary mode
            with open(file_path, 'rb') as f:
                # Calculate the MD5 hash of the file contents
                file_hash = hashlib.md5(f.read()).hexdigest()

            # Check if the file hash already exists in the dictionary
            if file_hash in file_hashes:
                duplicate_file_path = file_hashes[file_hash]
                print(f"Found duplicate file: {file_path}")
                print(f"Duplicate of: {duplicate_file_path}")
                # Delete the duplicate file
                os.remove(file_path)
                print(f"Deleted duplicate file: {file_path}")
            else:
                # Store the file hash and path in the dictionary
                file_hashes[file_hash] = file_path

    # Get the number of files left
    file_count = len(file_hashes)
    print(f"Number of files left: {file_count}")

def check_missing_files(folder_path, max_number):
    present_files = set()
    missing_files = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.jpg'):
            try:
                file_number = int(os.path.splitext(filename)[0])
                present_files.add(file_number)
            except ValueError:
                continue

    for i in range(1, max_number + 1):
        if i not in present_files:
            missing_files.append(i)

    num_missing_files = len(missing_files)
    num_present_files = len(present_files)

    print("Missing Files:")
    print(missing_files)
    print("Total Files Present:", num_present_files)
    print("Total Files Missing:", num_missing_files)


def main():
    # source = ["./flower", "./flowers", "./houses", "./natural images", "./tree", "./tree2"]
    # destinations = "./data"
    # 10872 pictures are already exists
    # print(organize_images_in_folder(source, destinations, 10872))
    # for index in range(2251, 15474):
    #     image_path = f"./trainDataHouseImg/{index}.jpg"
    #     photo_into_sketch(image_path)
    # print("Finished successfully!")
    # print("trainDataImg5:")
    # check_missing_files("./trainDataImg5", 2000)
    # delete_duplicate_files("./trainDataImg5")
    # print("trainDataSketch5:")
    # delete_duplicate_files("./trainDataSketch5")
    # print("testDataImg5:")
    # delete_duplicate_files("./testDataImg5")
    # print("testDataSketch5:")
    # delete_duplicate_files("./testDataSketch5")
    # print("Finish!")

    image_folder = "trainDataImg"
    sketch_folder = "trainDataSketch"
    output_folder = "train"

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over images in the folders and connect them
    image_files = os.listdir(image_folder)
    sketch_files = os.listdir(sketch_folder)

    for img_file, sketch_file in zip(image_files, sketch_files):
        img_path = os.path.join(image_folder, img_file)
        sketch_path = os.path.join(sketch_folder, sketch_file)

        image = cv2.imread(img_path)
        sketch = cv2.imread(sketch_path)

        # Check if image is None
        if image is None or sketch is None:
            print(f"Error: Unable to read image or sketch {img_file}. Skipping...")
            continue

        # Convert images to tensors
        image = np.transpose(image, (2, 0, 1)) if len(image.shape) == 3 else image
        sketch = np.transpose(sketch, (2, 0, 1)) if len(sketch.shape) == 3 else sketch

        image_tensor = torch.from_numpy(image)
        sketch_tensor = torch.from_numpy(sketch)

        connected_image = connect_x_y(image_tensor, sketch_tensor)

        # Save the connected image to the output folder
        output_path = os.path.join(output_folder, f"connected_{img_file}")
        cv2.imwrite(output_path, np.transpose(connected_image.numpy(), (1, 2, 0)))

    print("Images connected and saved to output folder.")


if __name__ == '__main__':
    main()
