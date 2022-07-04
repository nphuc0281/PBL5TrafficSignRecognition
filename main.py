from TSR import TrafficSignDetection
import cv2
from tkinter import *
from tkinter.ttk import *
import tkinter
import PIL.Image
import PIL.ImageTk
from playsound import playsound
import threading
import asyncio
import queue
import nest_asyncio
nest_asyncio.apply()
# from gtts import gTTS
# import os


class GUI:
    def __init__(self, max_data):
        self.root = Tk()
        self.core = TrafficSignDetection(path_to_weight='./models/TSR.pt', cam_mode=1)  # 0 for pc, 1 for jetson nano

        # thread-safe data storage
        self.the_queue = queue.Queue()

        # Variables
        self.camera_image = None
        self.after_id = None
        self.max_data = max_data
        self.thread = None
        self.data = []
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

        self.loop = asyncio.get_event_loop()

        # Window
        self.root.title("Traffic Sign Recognition")
        self.root.geometry("1300x800")

        # Camera view
        canvas_w = self.core.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        canvas_h = self.core.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.camera_canvas = Canvas(self.root, width=canvas_w, height=canvas_h, bg="grey")
        self.camera_canvas.pack(side='top', expand=True)

        # Label
        self.lblResults = Label(self.root, text="Không phát hiện biển báo", foreground='red', font=("Roboto", 20))
        self.lblResults.pack(side='top', anchor='n', expand=True)
        self.current_labels = []

        # Button start camera
        self.button_start = Button(self.root, text="Start", command=lambda: self.do_asyncio())
        self.button_start.pack(side='left', anchor='e', expand=True, pady=(0, 150), padx=(0, 20))

        # Button stop camera
        self.button_stop = Button(self.root, text="Stop", command=self.stop_asyncio)
        self.button_stop.pack(side='right', anchor='w', expand=True, pady=(0, 150), padx=(20, 0))

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

    def refresh_data(self):
        # do nothing if the aysyncio thread is dead
        # and no more data in the queue
        if not self.thread.is_alive() and self.the_queue.empty():
            self.do_asyncio()
            return

        # refresh the GUI with new data from the queue
        while not self.the_queue.empty():
            frame, label_keys = self.the_queue.get()

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
                    thread = threading.Thread(target=self.speech, args=(False, intersect))
                    thread.daemon = True
                    thread.start()
                    self.current_labels = label_keys
                elif len(intersect) == 1:
                    self.lblResults['text'] = self.ts_labels[intersect[0]]
                    thread = threading.Thread(target=self.speech, args=(intersect[0], False))
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

        print('RefreshData...')
        self.after_id = self.root.after(200, self.refresh_data)

    def do_asyncio(self):
        # create thread object
        self.thread = AsyncioThread(self.the_queue, self.max_data, self.core, self.loop)

        # Timer to refresh data
        self.after_id = self.root.after(200, self.refresh_data)

        # start the thread
        self.thread.start()

    def stop_asyncio(self):
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.camera_canvas.delete("all")
        self.thread.stop()


class AsyncioThread(threading.Thread):
    def __init__(self, the_queue, max_data, core, loop):
        self.asyncio_loop = loop
        self.core = core
        self.the_queue = the_queue
        self.max_data = max_data
        self._stop_event = threading.Event()
        threading.Thread.__init__(self)

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        self.asyncio_loop.run_until_complete(self.do_data())

    async def do_data(self):
        """ Creating and starting 'maxData' asyncio-tasks. """
        tasks = [self.create_dummy_data() for key in range(self.max_data)]
        await asyncio.wait(tasks)

    async def create_dummy_data(self):
        """ Create data and store it in the queue. """
        # Read from camera
        _, frame = self.core.camera.read()

        # Processing frame with YOLOv5 model
        frame, label_keys = await self.recognition_process(frame)

        self.the_queue.put((frame, label_keys))

    async def recognition_process(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.core.score_frame(frame)
        frame, labels = self.core.plot_boxes(results, frame)
        return frame, labels


# Start app GUI
window = GUI(50)
window.root.mainloop()
