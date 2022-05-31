from TSR import TrafficSignDetection
from time import time
import numpy as np
import cv2
import sv_ttk
from tkinter import *
from tkinter.ttk import *
import tkinter
import PIL.Image
import PIL.ImageTk


class GUI(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.core = TrafficSignDetection(out_file='vid.avi', path_to_weight='./models/TSR.pt', cam_mode=1)

        # Variables
        self.camera_image = None
        self.after_id = None

        # Window
        self.parent.title("Traffic Sign Recognition")
        self.parent.geometry("680x600")

        # Camera view
        canvas_w = self.core.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        canvas_h = self.core.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.camera_canvas = Canvas(self.parent, width=canvas_w, height=canvas_h, bg="grey")
        self.camera_canvas.pack(pady=(20,0))

        # Button start camera
        self.button_start = Button(windows, text="Start", command=self.start_camera)
        self.button_start.pack(side='left', anchor='e', expand=True, padx=(0,10))

        # Button stop camera
        self.button_stop = Button(windows, text="Stop", command=self.stop_camera)
        self.button_stop.pack(side='right', anchor='w', expand=True)

    def start_camera(self):
        start_time = time()

        # Read from camera
        ret, frame = self.core.camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Processing frame with YOLOv5 model and labeling
        results = self.core.score_frame(frame)
        frame = self.core.plot_boxes(results, frame)
        end_time = time()
        fps = 1/np.round(end_time - start_time, 3)
        print(f"Frames Per Second : {fps}")

        # Show camera
        self.camera_image = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
        self.camera_canvas.create_image(0, 0, image=self.camera_image, anchor=tkinter.NW)
        self.after_id = self.parent.after(15, self.start_camera)

    def stop_camera(self):
        if self.after_id:
            self.parent.after_cancel(self.after_id)
            self.camera_canvas.delete("all")


# Start app GUI
windows = Tk()
sv_ttk.set_theme("light")
app = GUI(windows)
windows.mainloop()





