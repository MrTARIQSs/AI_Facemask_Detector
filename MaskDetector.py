import tkinter as tk
import cv2
from tkinter import filedialog
from tkinter import *
from PIL import Image
from PIL import ImageTk
from controller import process_image_frame
from controller import detect_mask_and_apply_modification_on
from controller import video_detection
from Load381model import *
from imutils.video import VideoStream
import imutils
import time
import os
root = tk.Tk()
countWearing = 0
countNotWearing = 0
mobilenet_v2_model = loadMobileNetModel()
our_model = loadOurOwnModel()
fileName = ""
placeholder = PhotoImage(file='black-600x338.png')

Selection = [
"Model 1",
"Model 2"
]
var = Selection[0]

def showVid():
    global leftPanel, rightPanel
    root.withdraw()
    if var.get() == "Model 1":
        image, detected_image, wearing, notWearing  = video_detection(mobilenet_v2_model)

        label.configure(text="\nNumber of people wearing a mask: " + str(wearing) + "\nNumber of people not wearing a mask: " + str(notWearing))
        leftPanel.configure(image=image)
        leftPanel.image = image
        rightPanel.configure(image=detected_image)
        rightPanel.image = detected_image
        root.deiconify()
    else:
        image, wearing, notWearing = video_detection(our_model)
        label.configure(text="\nNumber of people wearing a mask: " + str(wearing) + "\nNumber of people not wearing a mask: " + str(notWearing))
        root.deiconify()

def addPic():
    global leftPanel, rightPanel, fileName
    fileName = filedialog.askopenfilename(initialdir="/", title="Select an image", filetypes=(("Image files",("*.PNG", "*.JPEG", "*.JPG")), (("All Files", "*.*"))))
    if len(fileName) > 0: #insure an image was selected
        selection(var.get())

def selection(model):
    global leftPanel, rightPanel
    if len(fileName) > 0: #insure an image was selected
        image = cv2.imread(fileName) #read the selected image
        faces = process_image_frame(fileName)
        # model_name = menu.selected
        # if model_name == "model1":
        #     model =
        # Create the model to make prediction
        if model == "Model 1":
            detected_image, count_mask, count_none_mask = detect_mask_and_apply_modification_on(image.copy(), faces, mobilenet_v2_model)
        else:
            detected_image, count_mask, count_none_mask = detect_mask_and_apply_modification_on(image.copy(), faces, our_model)
        countWearing = count_mask
        countNotWearing = count_none_mask
        height = 300
        width = int(image.shape[1]*height/image.shape[0])
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA) #resizing picture to a more reasonable size
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #converting the BGR format of cv2 to RGB
        detected_image = cv2.resize(detected_image, (width, height), interpolation=cv2.INTER_AREA)
        detected_image = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        detected_image = Image.fromarray(detected_image)
        image = ImageTk.PhotoImage(image)
        detected_image = ImageTk.PhotoImage(detected_image)
        leftPanel.configure(image=image)
        rightPanel.configure(image=detected_image)
        leftPanel.image = image
        rightPanel.image = detected_image

        label.configure(text = "\nNumber of people wearing a mask: "+str(countWearing)+"\nNumber of people not wearing a mask: "+str(countNotWearing))


root.configure(background="#aaaaaa")

leftPanel = Label(image=placeholder)
leftPanel.image = placeholder
leftPanel.pack(side="left", padx=10, pady=120)
rightPanel = Label(image=placeholder)
rightPanel.image = placeholder
rightPanel.pack(side="right", padx=10, pady=120)


frame = tk.Frame(root, bg="black")
frame.place(relwidth="1.0", relheight="0.2", rely="0")

var = StringVar(frame)
var.set(Selection[0])

DD = OptionMenu(root, var, *Selection, command=selection)
DD.config(bg = "black", fg="white", activebackground="black", activeforeground="white")
DD["highlightthickness"]=0
DD.pack(pady=40)

importIcon = PhotoImage(file='import.png')
liveFeedIcon = PhotoImage(file='camera.png')

insertPic = tk.Button(frame, image=importIcon, bg="black", command=addPic, pady=10)
insertPic.pack(side="right", padx=30, pady=10)

vidFeed = tk.Button(frame, image=liveFeedIcon, bg="black", command=showVid, pady=10)
vidFeed.pack(side="left", padx=30, pady=10)

frame2 = tk.Frame(root, bg="black")
frame2.place(relwidth="1.0", relheight="0.2", rely="0.8")

label = Label(frame2, text="\nNumber of people wearing a mask: "+str(countWearing)+"\nNumber of people not wearing a mask: "+str(countNotWearing), bg="black", fg="white", font="none 20 bold")
label.pack()

root.resizable(0,0)
root.mainloop()
