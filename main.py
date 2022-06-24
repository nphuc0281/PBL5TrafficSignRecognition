from TSR import TrafficSignDetection
import cv2
from tkinter import *
from tkinter.ttk import *
import tkinter
import PIL.Image
import PIL.ImageTk
from playsound import playsound
from multiprocessing.pool import ThreadPool
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

        # Window
        self.parent.title("Traffic Sign Recognition")
        self.parent.geometry("1300x800")

        # Camera view
        canvas_w = self.core.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        canvas_h = self.core.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.camera_canvas = Canvas(self.parent, width=canvas_w, height=canvas_h, bg="grey")
        self.camera_canvas.pack(side='top', expand=True)

        # Label
        self.lblResults = Label(windows, text="Không phát hiện biển báo", foreground='red', font=("Arial", 20))
        self.lblResults.pack(side='top', anchor='n', expand=True)

        # Button start camera
        self.button_start = Button(windows, text="Start", command=self.start_camera)
        self.button_start.pack(side='left', anchor='e', expand=True, pady=(0, 20), padx=(0, 20))

        # Button stop camera
        self.button_stop = Button(windows, text="Stop", command=self.stop_camera)
        self.button_stop.pack(side='right', anchor='w', expand=True, pady=(0, 20), padx=(20, 0))


    def start_camera(self):
        # Read from camera
        _, frame = self.core.camera.read()

        # Processing frame with YOLOv5 model and labeling
        pool = ThreadPool(processes=4)
        async_result = pool.apply_async(self.recognition_process, (frame,))
        frame, label_keys = async_result.get()

        # Change labels
        label_keys = list(set([self.core.class_to_label(label_keys[i]) for i in range(len(label_keys))]))
        labels = [self.ts_labels[label_keys[i]] for i in range(len(label_keys))]
        if len(labels) > 1:
            label = '\n'.join(labels)
            if self.lblResults['text'] != label:
                self.speech(keys=label_keys)
        elif len(labels) == 1:
            label = labels[0]
            if self.lblResults['text'] != label:
                self.speech(key=label_keys[0])
        else:
            label = 'Không phát hiện biển báo'

        self.lblResults['text'] = label

        # Show camera
        self.camera_image = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
        self.camera_canvas.create_image(0, 0, image=self.camera_image, anchor=tkinter.NW)
        self.after_id = self.parent.after(10, self.start_camera)

    def recognition_process(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.core.score_frame(frame)
        frame = self.core.plot_boxes(results, frame)
        return frame, results[0]

    def stop_camera(self):
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
                playsound("sounds/"+k+".mp3", block=False)
        else:
            # if not os.path.exists("sounds/" + key + ".mp3"):
            #     tts = gTTS(self.ts_labels[key], tld="com.vn", lang="vi")
            #     tts.save("%s.mp3" % os.path.join("sounds", key))
            playsound("sounds/" + key + ".mp3", block=False)


# Start app GUI
windows = Tk()
app = GUI(windows)
windows.mainloop()
