import numpy as np
import face_recognition
import cv2
import uvicorn
import shutil
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import os
import pickle
from pymongo import MongoClient
import base64


# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["user_info"]  # Your database
users_collection = db["user"]  # Collection for user data


# FastAPI app
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


# Load known face encodings and names from MongoDB
def load_known_faces():
    known_face_encodings = []
    known_face_names = []

    # Retrieve all users from the MongoDB collection
    users = users_collection.find()

    for user in users:
        # Deserialize embeddings and check if they are valid
        embeddings = pickle.loads(user['embeddings'])
        
        # Ensure embeddings are a non-empty list
        if embeddings and isinstance(embeddings, list) and len(embeddings) > 0:
            known_face_encodings.append(embeddings[0])  # Append the first encoding
            known_face_names.append(user['name'])

    return known_face_encodings, known_face_names


# Recognize faces in an image
def recognize_faces_in_image(image):
    # Load known faces from the database
    known_face_encodings, known_face_names = load_known_faces()

    # Find all the faces and face encodings in the uploaded image
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    # List to store recognized user names
    recognized_user_names = []

    if len(face_encodings) == 0:
        return recognized_user_names, image  # No faces found, return empty names

    # Loop through each face in the uploaded image
    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare the face with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Find the best match
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            recognized_user_names.append(name)

        # Draw a box around the face (optional)
        top, right, bottom, left = face_location
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face (optional)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    return recognized_user_names, image



@app.post("/register_new_user")
async def register_new_user(
    file: UploadFile = File(...),
    name: str = Form(...),
    age: int = Form(...),
    dob: str = Form(...),
    phone: str = Form(...),
    address: str = Form(...)
):
    contents = await file.read()  # Read the image file contents

    # Convert the image to NumPy array and extract face embeddings
    np_image = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_UNCHANGED)

    embeddings = face_recognition.face_encodings(image)

    if len(embeddings) == 0:
        raise HTTPException(status_code=400, detail="No face detected in the image")

    # Serialize face embeddings (pickle)
    pickle_data = pickle.dumps(embeddings)

    # Prepare the user data to be stored in MongoDB
    user_data = {
        "name": name,
        "age": age,
        "dob": dob,
        "phone": phone,
        "address": address,
        "image": contents,  # Store image as binary
        "embeddings": pickle_data  # Store embeddings as binary
    }

    users_collection.insert_one(user_data)  # Save user data in MongoDB

    return {'registration_status': 'Registered successfully'}




# API route to upload an image and recognize faces
@app.post("/recognize/")
async def recognize_faces(file: UploadFile = File(...)):
    try:
        # Read image from the uploaded file
        contents = await file.read()
        nnparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nnparr, cv2.IMREAD_UNCHANGED) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Recognize faces in the image
        recognized_faces, annotated_image = recognize_faces_in_image(img)

        if not recognized_faces:
            return JSONResponse(content={"message": "No recognized faces found."}, media_type="application/json")

        user_details = []

        for name in recognized_faces:
            # Retrieve user information from MongoDB based on recognized name
            user = users_collection.find_one({"name": name}, {"_id": 0, "embeddings": 0})  # Exclude _id and embeddings
            if user:
                # Convert image to base64 if it's bytes
                if 'image' in user and isinstance(user['image'], bytes):
                    user['image'] = base64.b64encode(user['image']).decode('utf-8')  # Convert bytes to base64 string

                user_details.append(user)

        return JSONResponse(content={"recognized_faces": user_details}, media_type="application/json")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")




if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
