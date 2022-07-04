import cv2


class Camera:
    def __init__(self, mode=None):
        if mode == 0:
            self.cam = cv2.VideoCapture(0)
        elif mode == 1:
            self.cam = cv2.VideoCapture(self.gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

        print('Finish initialize camera.')

    def __del__(self):
        self.cam.release()

    @staticmethod
    def gstreamer_pipeline(
            capture_width=1920,
            capture_height=1080,
            display_width=1920,
            display_height=1080,
            framerate=60,
            flip_method=0,
    ):
        return (
                "nvarguscamerasrc ! "
                "video/x-raw(memory:NVMM), "
                "width=(int)%d, height=(int)%d, "
                "format=(string)NV12, framerate=(fraction)%d/1 ! "
                "nvvidconv flip-method=%d ! "
                "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
                "videoconvert ! "
                "video/x-raw, format=(string)BGR ! appsink"
                % (
                    capture_width,
                    capture_height,
                    framerate,
                    flip_method,
                    display_width,
                    display_height,
                )
        )

    def show_camera(self):
        if self.cam.isOpened():
            window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
            while cv2.getWindowProperty("CSI Camera", 0) >= 0:
                ret_val, img = self.cam.read()
                cv2.imshow("CSI Camera", img)
                keyCode = cv2.waitKey(30) & 0xFF
                # Press ESC to exit
                if keyCode == 27:
                    break
            self.cam.release()
            cv2.destroyAllWindows()
        else:
            print("Không thể  mở Camera")