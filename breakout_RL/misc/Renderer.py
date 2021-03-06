import cv2


class Renderer(object):

    def __init__(self, width=600, height=600, window_name='obs', delay=1, video=False):
        self.window_name = window_name
        self.delay = delay
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width, height)

        self.video = video
        if video:
            self.vid = cv2.VideoWriter('demo.avi', cv2.VideoWriter_fourcc(*"XVID"), float(30), (160, 210), False)


    def update(self, screen):
        cv2.imshow(self.window_name, screen)
        cv2.waitKey(self.delay)
        if self.video:
            self.vid.write(screen)

    def release(self):
        if self.video:
            self.vid.release()