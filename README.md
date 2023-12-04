# Web-Based-Facial-Recognition-System
This project mainly works on facial and hand gesture recognition system to authenticate the user while registering or logging to the website.

# Aim
The primary aim of this project is to offer a user-friendly and accessible solution for facial and gesture recognition for login system which will add a layer of security above id and password. The key objectives include:
**Facial Recognition:** Identify individuals through facial features with high accuracy.
**Gesture Recognition:** Recognize and interpret hand gestures to enable interactive control.

# Models used
For facial redcognition - LBPH (Local Binary Pattern Histogram)
For hand gesture recognition - LSTM (Long short-term memory)


# How to use
The user interface primarily include two options - login and registration, and a notification segment for giving instruction to user

**Registration Phase**
- Firstly user needs to enter ID and password in registration part of the user interface.
- Then after clicking on submit, a live camera window will pop up with which gather facial features of the user and train a LBPH(local binary pattern histogram) model on it with help of haar cascade classifier.
- After face training is successful, a new pop up will open asking a hand gesture of a user that he want (The gestures to be used include ASL sign language). For gesture recognition we have used pre-trained LSTM model.
- When both face and gesture will be recognized, the data is stored for particular user and then a message 'Registration Successfull' will be shown to the user.

**Login Phase**
- Just like registration, user will first enter his/her ID and password.
- If id and password matches, then the face of the user will be authenticated. If it is successful, the hand gesture that the user used while regostration will be recognized.
- After successful authentication, a message 'Recognized successfully' will be shown to the user.

# Important modules used
CSV, Tkinter(for GUI interface), keras, tenserflow, pandas, numpy
