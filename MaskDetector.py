# Created by Tariq
# Credits: https://www.pyimagesearch.com/2016/05/23/opencv-with-tkinter/, https://www.youtube.com/watch?v=jE-SpRI3K5g&t=770s, https://stackoverflow.com/
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

root = tk.Tk()  # The UI window
countWearing = 0  # number of people wearing
countNotWearing = 0  # number of people not wearing
mobilenet_v2_model = loadMobileNetModel()  # Bring the mobile net model
our_model = loadOurOwnModel()  # bring our model
fileName = ""  # initialized file name
placeholder = cv2.imread('UI_Images/Black-600x338.png')  # black background for the panels when opening the system
height = int(root.winfo_screenheight() / 2.5)  # adaptive height based on screen resolution
width = int(placeholder.shape[1] * height / placeholder.shape[0])  # width based on the height to preserve aspect ratio
placeholder = cv2.resize(placeholder, (width, height),
                         interpolation=cv2.INTER_AREA)  # resizing picture to a more reasonable size
placeholder = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)  # converting the BGR format of cv2 to RGB
placeholder = Image.fromarray(placeholder)  # Creates an image memory
placeholder = ImageTk.PhotoImage(placeholder)  # display image

Selection = [
    "Model 1",
    "Model 2"
]  # dropdown selection elements

var = Selection[0]  # default to the first choice of the dropdown


def showVid():
    global leftPanel, rightPanel  # use the global left and right panel variables which are the panels that show the content
    root.withdraw()  # close the ui to display video feed
    if var.get() == "Model 1":  # if the model 1 drop down option was selected
        image, detected_image, wearing, notWearing = video_detection(
            mobilenet_v2_model)  # insert the mobile net v2 model, and get the return values from the video_detection method and assigning them to the 4 variables
        label.configure(
            text="\nNumber of people wearing a mask: " + str(wearing) + "\nNumber of people not wearing a mask: " + str(
                notWearing))  # change the count to the returned count after processing
        leftPanel.configure(image=image)  # reconfigure the left panel to contain the original image
        leftPanel.image = image  # show the changed left panel
        rightPanel.configure(image=detected_image)  # reconfigure the right panel to contain the processed image
        rightPanel.image = detected_image  # show the changed right panel
    else:  # if the model 2 drop down option was selected
        image, detected_image, wearing, notWearing = video_detection(
            our_model)  # insert our model, and get the return values from the video_detection method and assigning them to the 4 variables
        label.configure(
            text="\nNumber of people wearing a mask: " + str(wearing) + "\nNumber of people not wearing a mask: " + str(
                notWearing))  # change the count to the returned count after processing
        leftPanel.configure(image=image)  # reconfigure the left panel to contain the original image
        leftPanel.image = image  # show the changed left panel
        rightPanel.configure(image=detected_image)  # reconfigure the right panel to contain the processed image
        rightPanel.image = detected_image  # show the changed right panel
    root.deiconify()  # reopen the UI after closing the video


def addPic():
    global leftPanel, rightPanel, fileName  # use the global left and right panel variables which are the panels that show the content
    fileName = filedialog.askopenfilename(initialdir="/", title="Select an image", filetypes=(
    ("Image files", ("*.PNG", "*.JPEG", "*.JPG")),
    (("All Files", "*.*"))))  # open the file explorer and assign the file name and location to the fileName variable
    if len(fileName) > 0:  # insure an image was selected
        selection(var.get())  # get the selected model and pass it to the selection method


def selection(model):
    global leftPanel, rightPanel
    if len(fileName) > 0:  # insure an image was selected
        image = cv2.imread(fileName)  # read the selected image
        faces = process_image_frame(fileName)  # pass the file location to the processing method and get the face data
        if model == "Model 1":
            detected_image, count_mask, count_none_mask = detect_mask_and_apply_modification_on(image.copy(), faces,
                                                                                                mobilenet_v2_model)  # return the output using model 1
        else:
            detected_image, count_mask, count_none_mask = detect_mask_and_apply_modification_on(image.copy(), faces,
                                                                                                our_model)  # return the output using model 2
        countWearing = count_mask  # assign the variable to the returned count
        countNotWearing = count_none_mask  # assign the variable to the returned count
        height = int(root.winfo_screenheight() / 2.5)  # adaptive height based on screen resolution
        width = int(image.shape[1] * height / image.shape[0])  # width based on the height to preserve aspect ratio
        image = cv2.resize(image, (width, height),
                           interpolation=cv2.INTER_AREA)  # resizing picture to a more reasonable size
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # converting the BGR format of cv2 to RGB
        detected_image = cv2.resize(detected_image, (width, height),
                                    interpolation=cv2.INTER_AREA)  # resize to appropriate size
        detected_image = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)  # converting the BGR format of cv2 to RGB
        image = Image.fromarray(image)  # Creates an image memory
        detected_image = Image.fromarray(detected_image)  # Creates an image memory
        image = ImageTk.PhotoImage(image)  # display image
        detected_image = ImageTk.PhotoImage(detected_image)  # display image
        leftPanel.configure(image=image)  # reconfigure the left panel to contain the original image
        rightPanel.configure(image=detected_image)  # reconfigure the right panel to contain the processed image
        leftPanel.image = image  # show the changed left panel
        rightPanel.image = detected_image  # show the changed left panel

        label.configure(text="\nNumber of people wearing a mask: " + str(
            countWearing) + "\nNumber of people not wearing a mask: " + str(
            countNotWearing))  # change the count to the returned count after processing


root.configure(background="#aaaaaa")  # the base background color for the UI window

leftPanel = Label(image=placeholder)  # assign default background to the variable
leftPanel.image = placeholder
leftPanel.pack(side="left", padx=10, pady=150)  # Show the the default background on the left panel
rightPanel = Label(image=placeholder)  # assign default background to the variable
rightPanel.image = placeholder
rightPanel.pack(side="right", padx=10, pady=150)  # Show the the default background on the left panel

frame = tk.Frame(root, bg="black")  # top frame
frame.place(relwidth="1.0", relheight="0.2", rely="0")  # resize and paint black

var = StringVar(frame)  # the content of the drop down menu
var.set(Selection[0])  # set the default value of the drop down menu

DD = OptionMenu(root, var, *Selection, command=selection)  # assign the drop down menu to a variable
DD.config(bg="black", fg="white", activebackground="black", activeforeground="white")  # recolor and place
DD["highlightthickness"] = 0  # recolor the highlighted state to black
DD.pack(pady=40)  # show the drop down menu

importIcon = PhotoImage(file='UI_Images/import.png')  # icon of the importing file option
liveFeedIcon = PhotoImage(file='UI_Images/camera.png')  # icon of the live video option

insertPic = tk.Button(frame, image=importIcon, bg="black", command=addPic, pady=10)  # new button
insertPic.pack(side="right", padx=30, pady=10)  # display button with icon

vidFeed = tk.Button(frame, image=liveFeedIcon, bg="black", command=showVid, pady=10)  # new button
vidFeed.pack(side="left", padx=30, pady=10)  # display button with icon

frame2 = tk.Frame(root, bg="black")  # bottom frame
frame2.place(relwidth="1.0", relheight="0.2", rely="0.8")  # paint black and resize

label = Label(frame2, text="\nNumber of people wearing a mask: " + str(
    countWearing) + "\nNumber of people not wearing a mask: " + str(countNotWearing), bg="black", fg="white",
              font="none 20 bold")  # shows the count of people wearing and not wearing, default count values are 0
label.pack()  # Show label

root.title("Face Mask Detector")  # window title
root.resizable(0, 0)  # disable resizing
root.iconbitmap(default="UI_Images/maskico.ico")  # the icon of the window
root.mainloop()  # the end of the ui code
