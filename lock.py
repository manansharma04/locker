import cv2
import os
import numpy as np
import time
import ctypes
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import getpass

# Constants
DATA_DIR = "data"
LOCKED_FILE = "locked.txt"
PASSWORD_FILE = "password.txt"
FACE_MODEL = cv2.face_LBPHFaceRecognizer.create()

# Gmail credentials
GMAIL_USER = "email"  # Replace with your Gmail email
GMAIL_PASSWORD = "password"  # Replace with your Gmail password or App Password

# Function to lock the file using LBPH face recognition
def lock_file():
    a = os.stat(LOCKED_FILE).st_file_attributes
    if(a&2):
        print("the folder is already locked!!!\n try unlocking it first")
        return
    # Capture multiple images of the user's face for training
    captured_faces, user_name = capture_and_save_user_faces_and_name()

    # Train the LBPH face recognizer with the user's faces
    labels, faces = load_training_data()
    labels.extend([1] * len(captured_faces))  # Label 1 represents the user's face for all captured faces
    faces.extend(captured_faces)
    FACE_MODEL.train(faces, np.array(labels))

    # Save the LBPH model
    FACE_MODEL.save(os.path.join(DATA_DIR, "lbph_model.yml"))

    # Save the user's name
    with open(os.path.join(DATA_DIR, "user_name.txt"), "w") as user_name_file:
        user_name_file.write(user_name)

    # Hide the file
    hide_file(LOCKED_FILE)

    # Set up a password for unlocking
    password = getpass.getpass("Set up a password to unlock the file: ")
    with open(os.path.join(DATA_DIR, PASSWORD_FILE), "w") as password_file:
        password_file.write(password)

    # Ask the user for the recipient's email address
    recipient_email = input("Enter the recipient's email address: ")

    # Send an email with the folder password
    send_email("Folder Password", f"Your folder password: {password}", recipient_email, GMAIL_USER, GMAIL_PASSWORD)

    print("File is locked.")
def has_hidden_attribute(file_path):
        # Get the file attributes
        file_attributes = ctypes.windll.kernel32.GetFileAttributesW(file_path)

        # Check if the hidden attribute is set
        if file_attributes != -1 and (file_attributes & 2):
            return False  # Hidden attribute is set
        else:
            return True
# Function to unlock the file using LBPH face recognition and password
def unlock_file():
    # Check if the file is locked

    if has_hidden_attribute(LOCKED_FILE):
        print("lock the file first !!")
        return

    # Verify the password
    entered_password = getpass.getpass("Enter the password: ")
    with open(os.path.join(DATA_DIR, PASSWORD_FILE), "r") as password_file:
        saved_password = password_file.read()

    if entered_password != saved_password:
        print("Wrong password. Try again.")
        return

    # Load the LBPH model
    FACE_MODEL.read(os.path.join(DATA_DIR, "lbph_model.yml"))

    # Capture the user's face
    user_face = capture_user_face()

    # Recognize the user's face using LBPH
    label, confidence = FACE_MODEL.predict(user_face)

    if label == 1 and confidence < 70:  # Adjust the confidence threshold as needed
        unhide_file( LOCKED_FILE)
        print("File is unlocked.")
        time.sleep(5)  # Keep the video window open for 5 seconds
    else:
        print("Face not recognized or confidence too low. File remains locked.")

# Function to send an email with Gmail
def send_email(subject, body, to_email, gmail_user, gmail_password):
    try:
        # Create an SMTP server connection to Gmail
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        
        # Login to your Gmail account
        server.login(gmail_user, gmail_password)
        
        # Create the email message
        message = MIMEMultipart()
        message['From'] = gmail_user
        message['To'] = to_email
        message['Subject'] = subject
        message.attach(MIMEText(body, 'plain'))
        
        # Send the email
        server.sendmail(gmail_user, to_email, message.as_string())
        server.quit()
        print(f"Email sent to {to_email}.")
    except Exception as e:
        print(f"Error sending email: {str(e)}")

# Function to capture and save multiple images of the user's face and name
def capture_and_save_user_faces_and_name():
    # Initialize variables to store captured faces
    captured_faces = []
    user_name = input("Enter your name: ")

    # Capture multiple images of the user's face
    video_capture = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    image_count = 0
    while image_count < 100:  # Capture and save 100 images
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            x, y, w, h = faces[0]
            user_face = gray[y:y + h, x:x + w]
            captured_faces.append(user_face)

            # Save the captured face image
            image_filename = os.path.join(DATA_DIR, f"user_face_{image_count}.png")
            cv2.imwrite(image_filename, user_face)

            image_count += 1

        cv2.imshow("Capture Face", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    return captured_faces, user_name

# Function to hide a file (Windows-specific)
def hide_file(file_path):
    os.system(f"attrib +h {file_path}")

# Function to unhide a file (Windows-specific)
def unhide_file(file_path):
    os.system(f"attrib -h {file_path}")

# Function to capture the user's face during unlocking
def capture_user_face():
    video_capture = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            x, y, w, h = faces[0]
            user_face = gray[y:y + h, x:x + w]
            return user_face

        cv2.imshow("Capture Face", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Function to load training data (labels and faces)
def load_training_data():
    labels = []
    faces = []
    # Load labels and faces here
    for i in range(100):  # Load 100 saved images
        image_filename = os.path.join(DATA_DIR, f"user_face_{i}.png")
        if os.path.exists(image_filename):
            image = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
            labels.append(1)  # Label 1 represents the user's face
            faces.append(image)

    return labels, faces

# Main menu
while True:
    print("Locker System Menu:")
    print("1. Lock File")
    print("2. Unlock File")
    print("3. Exit")

    choice = input("Enter your choice: ")

    if choice == "1":
        lock_file()
    elif choice == "2":
        unlock_file()
    elif choice == "3":
        break
    else:
        print("Invalid choice. Please try again.")
