import cv2
import face_recognition
import pymongo
import io
import requests
import numpy as np

# Connect to your MongoDB database
client = pymongo.MongoClient("mongodb+srv://user:user@cluster0.a87ri9o.mongodb.net/?retryWrites=true&w=majority")
db = client["test"]
collection = db["lawyers"]

def get_user_data(barcouncil_no):
    # Query the MongoDB collection to retrieve user data based on Barcouncil number
    user_data = collection.find_one({"BarcouncilNO": barcouncil_no})
    return user_data

def add_user(user_data):
    # Capture and store the user's face encoding
    image_url = user_data["profileImage"]
    response = requests.get(image_url)
    
    if response.status_code == 200:
        image_bytes = io.BytesIO(response.content)
        image_array = np.frombuffer(image_bytes.read(), np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        face_encodings = face_recognition.face_encodings(image)

        if len(face_encodings) > 0:
            user_data["encode"] = face_encodings[0].tolist()
            collection.insert_one(user_data)
            print("User added successfully.")
        else:
            print("No face found in the provided image.")
    else:
        print("Failed to retrieve user image from the URL")

def main():
    # Capture Barcouncil number as user input
    barcouncil_no = input("Enter your Barcouncil number: ")

    # Retrieve user data based on the Barcouncil number
    user_data = get_user_data(barcouncil_no)

    if user_data is None:
        print("User not found in the database")
        return

    # Load the image from the URL stored in the database
    image_url = user_data["profileImage"]
    response = requests.get(image_url)
    
    if response.status_code == 200:
        image_bytes = io.BytesIO(response.content)
        image_array = np.frombuffer(image_bytes.read(), np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        print("Failed to retrieve user image from the URL")
        return

    # Open the webcam
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        faces_cur_frame = face_recognition.face_locations(imgS)
        encodes_cur_frame = face_recognition.face_encodings(imgS, faces_cur_frame)

        for encode_face, face_loc in zip(encodes_cur_frame, faces_cur_frame):
            matches = face_recognition.compare_faces([encode_face], user_data["encode"])
            face_dis = face_recognition.face_distance([encode_face], user_data["encode"])

            match_index = np.argmin(face_dis)

            if matches[match_index]:
                name = user_data["BarcouncilNO"]
                y1, x2, y2, x1 = face_loc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Webcam', img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
