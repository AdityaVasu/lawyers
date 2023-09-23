import cv2
import pymongo
import io
import requests
import numpy as np
from skimage import metrics

# Connect to your MongoDB database
client = pymongo.MongoClient("mongodb+srv://user:user@cluster0.a87ri9o.mongodb.net/?retryWrites=true&w=majority")
db = client["test"]
collection = db["lawyers"]

def get_user_image(barcouncil_no):
    # Query the MongoDB collection to retrieve user data based on Barcouncil number
    user_data = collection.find_one({"BarcouncilNO": barcouncil_no})
    if user_data is not None:
        image_url = user_data.get("profileImage")
        if image_url:
            response = requests.get(image_url)
            if response.status_code == 200:
                image_bytes = io.BytesIO(response.content)
                image = cv2.imdecode(np.frombuffer(image_bytes.read(), np.uint8), cv2.IMREAD_COLOR)
                return image
    return None

def capture_image_from_webcam():
    # Open the webcam
    cap = cv2.VideoCapture(0)

    success, img = cap.read()

    # Release the webcam
    cap.release()
    
    return img

def resize_image(image, target_width, target_height):
    return cv2.resize(image, (target_width, target_height))

def compare_images(image1, image2):
    # Ensure both images have the same dimensions
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same dimensions.")

    # Convert images to grayscale for SSIM comparison
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Calculate SSIM score
    ssim_score = metrics.structural_similarity(gray_image1, gray_image2)
    return ssim_score

def main():
    # Capture Barcouncil number as user input
    barcouncil_no = input("Enter your Barcouncil number: ")

    # Retrieve user image from the database
    user_image = get_user_image(barcouncil_no)

    if user_image is None:
        print("User not found in the database.")
        return

    # Capture an image from the webcam
    webcam_image = capture_image_from_webcam()

    # Resize the webcam image to match the dimensions of the user image
    webcam_image = resize_image(webcam_image, user_image.shape[1], user_image.shape[0])

    # Compare the images
    similarity_score = compare_images(user_image, webcam_image)
    print(f"Image similarity score: {similarity_score:.2f}")

if __name__ == "__main__":
    main()