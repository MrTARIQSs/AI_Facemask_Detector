import tkinter as tk
import cv2
from tkinter import filedialog
from tkinter import *
from PIL import Image
from PIL import ImageTk
from controller import process_image_frame
from controller import detect_mask_and_apply_modification_on
import os
root = tk.Tk()
countWearing = 0
countNotWearing = 0
def addPic():
    global leftPanel, rightPanel
    fileName = filedialog.askopenfilename(initialdir="/", title="Select Picture", filetypes=(("PNG", "*.PNG"), ("JPEG", "*.JPEG"), ("JPG", "*.JPG"), (("All Files", "*.*"))))
    if len(fileName) > 0: #insure an image was selected
        image = cv2.imread(fileName) #read the selected image
        faces = process_image_frame(fileName)
        # model_name = menu.selected
        # if model_name == "model1":
        #     model =
        # Create the model to make prediction
        detected_image, count_mask, count_none_mask = detect_mask_and_apply_modification_on(image.copy(), faces, None)
        countWearing = count_mask
        countNotWearing = count_none_mask
        width = 600
        height = 338
        image = cv2.resize(image, (width,height), interpolation=cv2.INTER_AREA) #resizing picture to a more reasonable size
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

        label.configure(text = "Wearing a mask: "+str(countWearing)+"\nNo wearing a mask: "+str(countNotWearing))

root.configure(background="#aaaaaa")



placeholder = PhotoImage(file='black-600x338.png')

leftPanel = Label(image=placeholder)
leftPanel.image = placeholder
leftPanel.pack(side="left", padx=10, pady=120)
rightPanel = Label(image=placeholder)
rightPanel.image = placeholder
rightPanel.pack(side="right", padx=10, pady=120)


frame = tk.Frame(root, bg="black")
frame.place(relwidth="1.0", relheight="0.2", rely="0")

icon = PhotoImage(file='addPhoto.png')

insertPic = tk.Button(frame, image=icon, bg="black", command=addPic, pady=10)
insertPic.pack()

frame2 = tk.Frame(root, bg="black")
frame2.place(relwidth="1.0", relheight="0.2", rely="0.8")

label = Label(frame2, text="Wearing a mask: "+str(countWearing)+"\nNo wearing a mask: "+str(countNotWearing), bg="black", fg="white", font="none 20 bold")
label.pack()

root.resizable(0,0)
root.mainloop()
