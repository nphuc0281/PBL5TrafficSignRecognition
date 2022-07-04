from TSR import TrafficSignDetection
import cv2
from tkinter import *
from tkinter.ttk import *
import tkinter
import PIL.Image
import PIL.ImageTk
from playsound import playsound
from multiprocessing.pool import ThreadPool
from threading import Thread
# from gtts import gTTS
# import os


class GUI(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.core = TrafficSignDetection(path_to_weight='./models/TSR.pt', cam_mode=0)  # 0 for pc, 1 for jetson nano

        # Variables
        self.camera_image = None
        self.after_id = None
        self.ts_labels = {
            'CamDiNguocChieu': 'Cấm đi ngược chiều',
            'CamDo': 'Cấm đỗ',
            'CamQuayDau': 'Cấm quay đầu',
            'CamReTrai': 'Cấm rẽ trái',
            'CamRePhai': 'Cấm rẽ phải',
            'CamBopCoi': 'Cấm bóp còi',
            'GiaoNhauVoiDuongUuTien': 'Giao nhau với đường ưu tiên',
            'GiaoNhauVoiDuongKhongUuTien': 'Giao nhau với đường không ưu tiên',
            'GiaoNhauChayTheoVongXuyen': 'Giao nhau chạy theo vòng xuyến',
            'GiaoNhauVoiDuongSatCoRaoChan': 'Giao nhau với đường sắt có rào chắn',
            'TreEmQuaDuong': 'Trẻ em qua đường',
            'BenhVien': 'Bệnh viện',
        }
        self.current_labels = []

        # Window
        self.parent.title("Traffic Sign Recognition")
        self.parent.geometry("1300x800")

        # Camera view
        canvas_w = self.core.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        canvas_h = self.core.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.camera_canvas = Canvas(self.parent, width=canvas_w, height=canvas_h, bg="grey")
        self.camera_canvas.pack(side='top', expand=True)

        # Label
        self.lblResults = Label(windows, text="Không phát hiện biển báo", foreground='red', font=("Roboto", 20))
        self.lblResults.pack(side='top', anchor='n', expand=True)

        # Button start camera
        self.button_start = Button(windows, text="Start", command=self.start_camera)
        self.button_start.pack(side='left', anchor='e', expand=True, pady=(0, 150), padx=(0, 20))

        # Button stop camera
        self.button_stop = Button(windows, text="Stop", command=self.stop_camera, state='disable')
        self.button_stop.pack(side='right', anchor='w', expand=True, pady=(0, 150), padx=(20, 0))

    def start_camera(self):
        self.button_start['state'] = 'disable'
        self.button_stop['state'] = 'normal'

        # Read from camera
        _, frame = self.core.camera.read()

        # Processing frame with YOLOv5 model and labeling
        pool = ThreadPool(processes=1)
        async_result = pool.apply_async(self.recognition_process, (frame,))
        frame, label_keys = async_result.get()

        # Change labels
        label_keys = list(set(label_keys))
        label_keys.sort()
        labels = [self.ts_labels[label_keys[i]] for i in range(len(label_keys))]

        if len(label_keys) == 0:
            self.lblResults['text'] = 'Không phát hiện biển báo'
        elif self.current_labels != label_keys or len(self.current_labels) == 0:
            intersect = [x for x in label_keys if x not in self.current_labels]
            if len(labels) > 1:
                self.lblResults['text'] = '\n'.join(labels)
                thread = Thread(target=self.speech, args=(False, intersect))
                thread.daemon = True
                thread.start()
                self.current_labels = label_keys
            elif len(intersect) == 1:
                self.lblResults['text'] = self.ts_labels[intersect[0]]
                thread = Thread(target=self.speech, args=(intersect[0], False))
                thread.daemon = True
                thread.start()
                self.current_labels = label_keys
            elif len(labels) == 1:
                self.lblResults['text'] = labels[0]
            else:
                self.lblResults['text'] = 'Không phát hiện biển báo'
        else:
            if len(labels) > 1:
                self.lblResults['text'] = '\n'.join(labels)
            elif len(labels) == 1:
                self.lblResults['text'] = labels[0]
            else:
                self.lblResults['text'] = 'Không phát hiện biển báo'

        # Show camera
        self.camera_image = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
        self.camera_canvas.create_image(0, 0, image=self.camera_image, anchor=tkinter.NW)
        self.after_id = self.parent.after(15, self.start_camera)

    def recognition_process(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.core.score_frame(frame)
        frame, label_keys = self.core.plot_boxes(results, frame)
        return frame, label_keys

    def stop_camera(self):
        self.button_start['state'] = 'normal'
        self.button_stop['state'] = 'disable'
        if self.after_id:
            self.parent.after_cancel(self.after_id)
            self.camera_canvas.delete("all")

    @staticmethod
    def speech(key=None, keys=None):
        if keys:
            for k in keys:
                # if not os.path.exists("sounds/"+k+".mp3"):
                #     tts = gTTS(self.ts_labels[k], tld="com.vn", lang="vi")
                #     tts.save("%s.mp3" % os.path.join("sounds", k))
                playsound("sounds/"+k+".mp3", block=True)
        elif key:
            # if not os.path.exists("sounds/" + key + ".mp3"):
            #     tts = gTTS(self.ts_labels[key], tld="com.vn", lang="vi")
            #     tts.save("%s.mp3" % os.path.join("sounds", key))
            playsound("sounds/" + key + ".mp3", block=True)


# Start app GUI
windows = Tk()
app = GUI(windows)
windows.mainloop()
