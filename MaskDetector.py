import tkinter as tk
from tkinter import filedialog, Text
import os

root = tk.Tk()

def addPic():
    fileName = filedialog.askopenfilename(initialdir="/", title="Select Picture", filetypes=(("JPEG", "*.JPEG"), ("PNG", "*.PNG"), ("JPG", "*.JPG"), (("All Files", "*.*"))))

canvas = tk.Canvas(root, height=600, width=800, bg="#000000")
canvas.pack()

frame = tk.Frame(root, bg="silver")
frame.place(relwidth="1.0", relheight="0.6", rely="0.2")

insertPic = tk.Button(frame, text="Insert Photo", fg="white", bg="black", command=addPic)
insertPic.pack()

root.mainloop()
