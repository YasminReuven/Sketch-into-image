import os
import shutil
import cv2
import numpy as np


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
    output_directory = "sketches"
    output_filename = os.path.splitext(os.path.basename(image_path))[0] + "_sketch.jpg"
    output_path = os.path.join(output_directory, output_filename)
    cv2.imwrite(output_path, outlined_image)
    # Display the outlined image
    # cv2.imshow("Outlined Image", outlined_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def main():
    # source = ["./flower", "./flowers", "./houses", "./natural images", "./tree", "./tree2"]
    # destinations = "./data"
    # 10872 pictures are already exists
    # print(organize_images_in_folder(source, destinations, 10872))
    image_path = "./flower_0054.jpg"
    photo_into_sketch(image_path)


if __name__ == '__main__':
    main()
