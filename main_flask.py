import json
import logging
import os  # Import os at the top
import sys
import threading
import time

import cv2
import numpy as np
from flask import Flask, request
from PIL import Image  # For PIL conversion
from PyQt5 import QtCore
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow

debug = False


class Signals(QObject):
    game_state_signal = pyqtSignal()  # Signal to trigger image cropping
    hide_signal = pyqtSignal()        # Signal to hide the overlay


class OverlayWindow(QMainWindow):
    def __init__(self, objects):
        super().__init__()
        self.objects = objects  # Store the imported objects
        self.image_dict = {}    # Dictionary to hold the cropped images
        # Store the full best_key for each object in matched_results
        self.matched_results = {}
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.WindowTransparentForInput
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.hide()  # Start hidden

        # Define unique colors for each tier
        self.tier_colors = {
            "Lowtier": QColor(157, 157, 157),      # gray
            "Situational": QColor(255, 255, 255),    # white
            "Tier1": QColor(255, 128, 0),            # orange legendary color
            "Tier2": QColor(163, 53, 238),           # Purple
            "Tier3": QColor(0, 112, 221),            # Blue
            "Tier4": QColor(30, 255, 0)              # green
        }

        # Define border thickness based on tier importance (lower to higher)
        self.tier_border_thickness = {
            "Lowtier": 1,
            "Situational": 2,
            "Tier4": 3,
            "Tier3": 4,
            "Tier2": 5,
            "Tier1": 6
        }

        # Load the tiny icon images for shard and aghs
        icon_size = 16
        self.shard_pixmap = QPixmap("dota2_ability_icons/shard.png")
        if not self.shard_pixmap.isNull():
            self.shard_pixmap = self.shard_pixmap.scaled(
                icon_size, icon_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        else:
            print("[ERROR] shard.png not found.")

        self.aghs_pixmap = QPixmap("dota2_ability_icons/aghs.png")
        if not self.aghs_pixmap.isNull():
            self.aghs_pixmap = self.aghs_pixmap.scaled(
                icon_size, icon_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        else:
            print("[ERROR] aghs.png not found.")

    def cropper(self):
        """Crop images and match them with IconMatcher without saving images to disk."""
        print("[DEBUG] cropper() called.")
        try:
            from classes import IconMatcher, ImageCropper

            # 1) Crop images using ImageCropper
            cropper_instance = ImageCropper("cord_dict.txt", screenshot=True)
            print("[DEBUG] ImageCropper initialized.")
            cropper_instance.crop_images()
            self.image_dict = cropper_instance.get_image_dict()
            print(f"[DEBUG] Cropped images: {list(self.image_dict.keys())}")

            # 2) Create an instance of IconMatcher
            matcher = IconMatcher(
                base_library_dir="dota2_ability_icons", input_size=(224, 224))

            # 3) For each cropped image, convert it to a PIL image and match it.
            for obj_name, cropped_image in self.image_dict.items():
                if isinstance(cropped_image, np.ndarray):
                    # Convert OpenCV's image (BGR) to RGB for PIL
                    rgb_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_image)

                    best_key, dist = matcher.match_icon(pil_img)
                    # Instead of only storing the tier, we now store the full best_key.
                    self.matched_results[obj_name] = best_key
                    print(
                        f"[INFO] {obj_name} matched with {best_key} (distance: {dist:.4f})")
                else:
                    print(
                        f"[WARNING] Skipping {obj_name}: unexpected image type.")

            self.showFullScreen()
            self.update()

        except Exception as e:
            print(f"[ERROR] Error in cropper: {e}")

    def hide_overlay(self):
        """Hide the overlay."""
        print("[DEBUG] Hiding the overlay.")
        self.hide()

    def paintEvent(self, event):
        """Custom paint event to draw the overlay."""
        painter = QPainter(self)

        # Set font for text (if needed)
        painter.setFont(QFont("Arial", 12))

        # For each object, draw its border and any extra icons.
        for obj_name, obj in self.objects.items():
            # Get the full best_key that was stored.
            best_key = self.matched_results.get(obj_name, "")
            # Extract the tier name from the best_key.
            # The best_key is of the form "TierName/filename.png".
            tier_name = best_key.split(os.path.sep)[
                0] if best_key else "Unknown"

            color = self.tier_colors.get(tier_name, QColor(255, 255, 255))
            border_thickness = self.tier_border_thickness.get(tier_name, 1)

            pen = painter.pen()
            pen.setColor(color)
            pen.setWidth(border_thickness)
            painter.setPen(pen)

            rect = QtCore.QRect(obj["x"], obj["y"], obj["w"], obj["h"])
            painter.drawRect(rect)

            # --- Draw small icons if best_key indicates "shard" and/or "aghs" ---
            icons_to_draw = []
            if "shard" in best_key.lower():
                icons_to_draw.append(self.shard_pixmap)
            if "aghs" in best_key.lower():
                icons_to_draw.append(self.aghs_pixmap)

            if icons_to_draw:
                padding = 2
                icon_width = self.shard_pixmap.width()  # Assuming both icons are same size
                total_icons_width = len(
                    icons_to_draw) * icon_width + (len(icons_to_draw) - 1) * padding

                start_x = rect.x() + rect.width() - total_icons_width - 2
                start_y = rect.y() + rect.height() - icon_width - 2

                for icon in icons_to_draw:
                    if not icon.isNull():
                        painter.drawPixmap(start_x, start_y, icon)
                    start_x += icon_width + padding


class GameServer:
    def __init__(self, overlay_widget=None, signals=None):
        self.app = Flask(__name__)
        self.game_state = None
        self.running = True  # Flag to control the threads
        self.clock_time = 0
        self.overlay_widget = overlay_widget
        self.signals = signals

        self.data_processing_thread = threading.Thread(
            target=self.process_data, daemon=True
        )

        self.app.add_url_rule('/', 'handle_post',
                              self.handle_post, methods=['POST'])

    def handle_post(self):
        """Handle incoming POST requests from Flask."""
        try:
            data = request.get_json()
            if data:
                if 'map' in data:
                    if 'game_state' in data['map']:
                        self.clock_time = data['map'].get('clock_time', 0)
                        self.game_state = data['map']['game_state']
                        return f"game_state: {self.game_state}, clock_time: {self.clock_time}"
                    else:
                        self.game_state = None
                        return "game_state is undefined"
                else:
                    self.game_state = None
                    return "map is undefined"
            else:
                self.game_state = None
                return "No JSON data received"
        except Exception as e:
            self.game_state = None
            return f"Error processing request: {str(e)}"

    def start_threads(self):
        flask_thread = threading.Thread(target=self.run)
        flask_thread.daemon = True
        flask_thread.start()
        self.data_processing_thread.start()

    def process_data(self):
        already_emitted = False
        already_hidden = False

        while self.running:
            if self.game_state == "DOTA_GAMERULES_STATE_HERO_SELECTION" and self.clock_time > -54:
                if not already_emitted and self.signals:
                    self.signals.game_state_signal.emit()
                    already_emitted = True
                    already_hidden = False
            elif self.game_state != "DOTA_GAMERULES_STATE_HERO_SELECTION":
                if not already_hidden and self.signals:
                    self.signals.hide_signal.emit()
                    already_hidden = True
                    already_emitted = False
            time.sleep(0.5)

    def run(self):
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        self.app.run(port=3000, threaded=True)

    def stop(self):
        self.running = False
        self.data_processing_thread.join()


def main():
    with open("cord_dict.txt", "r") as file:
        objects = json.load(file)

    app = QApplication(sys.argv)
    overlay = OverlayWindow(objects)

    signals = Signals()
    signals.game_state_signal.connect(overlay.cropper)
    signals.hide_signal.connect(overlay.hide_overlay)

    server = GameServer(overlay_widget=overlay, signals=signals)
    server.start_threads()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
