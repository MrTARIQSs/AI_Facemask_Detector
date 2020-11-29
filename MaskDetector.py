import tkinter as tk
import cv2
from tkinter import filedialog
from tkinter import *
from PIL import Image
from PIL import ImageTk
import os

root = tk.Tk()

def addPic():
    global leftPanel, rightPanel
    fileName = filedialog.askopenfilename(initialdir="/", title="Select Picture", filetypes=(("PNG", "*.PNG"), ("JPEG", "*.JPEG"), ("JPG", "*.JPG"), (("All Files", "*.*"))))
    if len(fileName) > 0: #insure an image was selected
        image = cv2.imread(fileName) #read the selected image
        width = 600
        height = 338
        image = cv2.resize(image, (width,height), interpolation=cv2.INTER_AREA) #resizing picture to a more reasonable size
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #converting the colored BGR format of cv2 to grayscale
        edge = cv2.Canny(grayscale, 60, 120)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #converting the BGR format of cv2 to RGB
        image = Image.fromarray(image)
        edge = Image.fromarray(edge)
        image = ImageTk.PhotoImage(image)
        edge = ImageTk.PhotoImage(edge)
        leftPanel.configure(image=image)
        rightPanel.configure(image=edge)
        leftPanel.image = image
        rightPanel.image = edge

root.configure(background="#aaaaaa")

countWearing = 0
countNotWearing = 0

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
