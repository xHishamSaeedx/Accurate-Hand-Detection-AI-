import os
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
import subprocess

from final_app2 import detect_hands, load_model
from graph import draw_graph
import pandas as pd
import random

hands_open = "hands open"
hands_close = "hands close"
filename = "output.csv"
class WebcamCaptureApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Webcam Capture")
        self.setGeometry(100, 100, 640, 480)

        self.layout = QVBoxLayout()

       
        # Add input text boxes
        self.handsopen_input = QLineEdit(self)
        self.handsopen_input.setPlaceholderText("Enter for Hands Open")
        self.layout.addWidget(self.handsopen_input)

        self.handsclose_input = QLineEdit(self)
        self.handsclose_input.setPlaceholderText("Enter for Hands Close")
        self.layout.addWidget(self.handsclose_input)

        self.filename_input = QLineEdit(self)
        self.filename_input.setPlaceholderText("File name")
        self.layout.addWidget(self.filename_input)

        self.grid_input = QLineEdit(self)
        self.grid_input.setPlaceholderText("grids")
        self.layout.addWidget(self.grid_input)

        # Add a button to trigger the disappearance of input boxes
        self.input_button = QPushButton("Submit Input")
        self.input_button.clicked.connect(self.hide_input)
        self.layout.addWidget(self.input_button)

        self.image_label = QLabel(self)
        self.layout.addWidget(self.image_label)

        self.captured_image_label = QLabel(self)
        self.layout.addWidget(self.captured_image_label)
        self.screenshot_taken = False

        self.graph_label = QLabel(self)
        self.layout.addWidget(self.graph_label)

        self.start_button = QPushButton("Start")
        self.capture_button = QPushButton("Capture")
        self.layout.addWidget(self.start_button)
        self.layout.addWidget(self.capture_button)
        self.restart_button = QPushButton("Restart")
        self.layout.addWidget(self.restart_button)

        self.start_button.clicked.connect(self.start_webcam)
        self.capture_button.clicked.connect(self.capture_image)
        self.restart_button.clicked.connect(self.restart_program)

        self.capture_button.setEnabled(False)
        self.restart_button.setEnabled(False)
        self.is_capturing = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.video_capture = None

        self.setLayout(self.layout)

        self.start_button.setEnabled(False)

        # Initialize a variable to track whether the input has been submitted
        self.input_submitted = False

    def hide_input(self):

        global hands_open
        global hands_close
        global filename
        # Get input values and hide the input boxes
        hands_open = 'hands open'
        hands_close = 'hands close'
        a=self.handsopen_input.text()
        b=self.handsclose_input.text()
        if a!='':
            hands_open=a
        if b!='':
            hands_close=b
        filename = self.filename_input.text()
        numbers_str = self.grid_input.text()
        number_strings = numbers_str.split(',')
        self.n = [int(num) for num in number_strings]
        self.start_button.setEnabled(True)



    def start_webcam(self):
        if not self.is_capturing:
            self.video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            self.is_capturing = True
            self.capture_button.setEnabled(True)
            self.restart_button.setEnabled(False)
            self.start_button.setText("Stop")
            self.timer.start(20)  # Update frame every 20ms

        else:
            self.video_capture.release()
            self.is_capturing = False
            self.capture_button.setEnabled(False)
            self.restart_button.setEnabled(True)
            self.start_button.setText("Start")
            self.image_label.clear()
            self.timer.stop()

    def update_frame(self):
        if self.is_capturing:
            ret, frame = self.video_capture.read()
            frame = cv2.flip(frame,1)
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channel = frame.shape
                self.grid=True
                if self.grid==True:
                    # Draw a nxn grid for all values of n
                    for i in self.n:
                        random.seed(i*20)
                        color = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
                        for j in range(0, width, int(width/i)):
                            cv2.line(frame, (j, 0), (j, height), color, 2, 1)
                        for j in range(0, height, int(height/i)):
                            cv2.line(frame, (0, j), (width, j), color, 2, 1)
                            


                bytes_per_line = 3 * width
                q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                self.image_label.setPixmap(pixmap)

    def capture_image(self):
        global hands_open
        global hands_close

        if self.is_capturing and not self.screenshot_taken:
            ret, frame = self.video_capture.read()
            frame = cv2.flip(frame,1)
            if ret:
                cv2.imwrite("captured_image.png", frame)
                detect_hands(self.n)

                draw_graph(hands_open, hands_close)

                df = pd.read_csv("output.csv")

                df = df.rename(columns= {'Hand Open': hands_open , 'Hand Closed': hands_close})

                df.to_csv(filename, index=False)

                self.show_combined_images("processed_image.jpg", "bar_graph.png")  # Display side by side
                # self.stop_webcam()
                # self.screenshot_taken = True

    def stop_webcam(self):
        if self.is_capturing:
            self.video_capture.release()
            self.is_capturing = False
            self.capture_button.setEnabled(False)
            self.restart_button.setEnabled(True)
            self.start_button.setEnabled(False)
            self.start_button.setText("Start")
            self.image_label.clear()
            self.timer.stop()

    def display_captured_image(self, image_path):
        pixmap = QPixmap(image_path)
        self.captured_image_label.setPixmap(pixmap)


    def display_graph_image(self, image_path):
        pixmap = QPixmap(image_path)
        self.graph_label.setPixmap(pixmap)



    def show_combined_images(self, screenshot_path, processed_image_path):
        # Load the captured screenshot
        screenshot = cv2.imread(screenshot_path)
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Load the processed image
        processed_image = cv2.imread(processed_image_path)
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Ensure both images have the same height for side-by-side display
        height = max(screenshot.shape[0], processed_image.shape[0])

        if screenshot.shape[0] < height:
            screenshot = cv2.copyMakeBorder(screenshot, 0, height - screenshot.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        if processed_image.shape[0] < height:
            processed_image = cv2.copyMakeBorder(processed_image, 0, height - processed_image.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        # Combine the two images side by side
        combined_image = np.hstack((screenshot, processed_image))

        # Display the combined image in the captured_image_label
        height, width, channel = combined_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(combined_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.captured_image_label.setPixmap(pixmap)

    def restart_program(self):
        app.quit()  # Quit the current application
        python = sys.executable
        os.execl(python, python, *sys.argv)  # Start a new instance of the program



if __name__ == '__main__':
    load_model()
    app = QApplication(sys.argv)
    window = WebcamCaptureApp()
    window.show()
    sys.exit(app.exec_())