import cv2
import numpy as np
import mediapipe as mp
import os
import time
import urllib.request
from datetime import datetime

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded!")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS


class HandDrawingApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

        self.prev_x, self.prev_y = 0, 0
        self.draw_color = (0, 255, 255)
        self.brush_thickness = 8
        self.is_drawing = False
        self.selected_color_idx = 2
        self.is_pinching = False

        # Eraser
        self.is_eraser = False
        self.eraser_thickness = 40
        self.all_fingers_start_time = None
        self.all_fingers_hold_sec = 2.0

        # Object pick and move
        self.is_picking = False
        self.pick_offset_x, self.pick_offset_y = 0, 0
        self.pick_anchor_x, self.pick_anchor_y = 0, 0
        self.picked_region = None
        self.pick_mask = None
        self.pick_bbox = (0, 0, 0, 0)
        self.was_picking = False

        self.colors = [
            (255, 255, 255),
            (255, 100, 100),
            (0, 255, 255),
            (255, 0, 0),
            (0, 255, 0),
            (255, 0, 255),
            (0, 165, 255),
            (128, 128, 255),
        ]
        self.color_names = ["White", "Red", "Cyan", "Blue", "Green", "Magenta", "Orange", "Pink"]

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.hand_landmarker = HandLandmarker.create_from_options(options)

        self.color_panel_x = 30
        self.color_panel_y = 180
        self.color_box_size = 45
        self.color_gap = 12

        self.save_button_rect = (30, 620, 150, 55)
        self.clear_button_rect = (30, 550, 150, 55)

        self.brush_sizes = [4, 8, 14, 22]
        self.brush_size_idx = 1

        self.points_buffer = []
        self.smoothing = 5
        self.frame_count = 0

        self.mouse_x, self.mouse_y = 0, 0
        self.mouse_clicked = False

    def lm_to_pixel(self, lm, w, h):
        return int(lm.x * w), int(lm.y * h)

    def fingers_up(self, landmarks, is_right_hand=False):
        fingers = []
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        # Thumb: tip extended away from palm
        if is_right_hand:
            fingers.append(1 if thumb_tip.x < thumb_ip.x - 0.02 else 0)
        else:
            fingers.append(1 if thumb_tip.x > thumb_ip.x + 0.02 else 0)
        # Other fingers: tip clearly above PIP joint
        for tip_id, pip_id in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            fingers.append(1 if landmarks[tip_id].y < landmarks[pip_id].y - 0.01 else 0)
        return fingers

    def all_fingers_up(self, fingers):
        return sum(fingers) >= 4

    def is_pinch(self, landmarks, threshold=0.07):
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        dist = np.sqrt(
            (thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2
        )
        return dist < threshold

    def is_thumb_index_touch(self, landmarks):
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        thumb_ip = landmarks[3]
        index_pip = landmarks[6]
        
        dist_tip = np.sqrt(
            (thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2
        )
        
        # Tips must be very close
        if dist_tip > 0.08:
            return False
        
        # Both fingers must be extended (not curled)
        thumb_extended = thumb_tip.y < thumb_ip.y
        index_extended = index_tip.y < index_pip.y
        
        return thumb_extended and index_extended

    def has_content_at(self, x, y, radius=40):
        h, w = 720, 1280
        x1 = max(0, x - radius)
        y1 = max(0, y - radius)
        x2 = min(w, x + radius)
        y2 = min(h, y + radius)
        region = self.canvas[y1:y2, x1:x2]
        return np.any(region > 0)

    def pick_region(self, cx, cy):
        h, w = 720, 1280

        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        target_label = 0
        if 0 <= cy < h and 0 <= cx < w:
            target_label = labels[cy, cx]

        # Search nearby if exact pixel missed
        if target_label == 0:
            for r in range(5, 50, 5):
                for dy in range(-r, r+1, 5):
                    for dx in range(-r, r+1, 5):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and labels[ny, nx] > 0:
                            target_label = labels[ny, nx]
                            break
                    if target_label > 0:
                        break
                if target_label > 0:
                    break

        if target_label == 0:
            return False

        # Create mask for this component only
        shape_mask = (labels == target_label).astype(np.uint8) * 255

        coords = cv2.findNonZero(shape_mask)
        if coords is None or len(coords) < 10:
            return False

        x, y, bw, bh = cv2.boundingRect(coords)
        pad = 10
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)

        self.pick_anchor_x, self.pick_anchor_y = cx, cy
        self.pick_offset_x, self.pick_offset_y = 0, 0

        self.picked_region = self.canvas[y1:y2, x1:x2].copy()
        self.pick_mask = shape_mask[y1:y2, x1:x2].copy()
        self.pick_bbox = (x1, y1, x2, y2)

        # Clear shape from canvas
        self.canvas[shape_mask > 0] = [0, 0, 0]

        self.is_picking = True
        self.was_picking = False
        print(f"Picked shape at ({cx}, {cy})")
        return True

    def move_picked(self, cx, cy):
        if self.picked_region is None:
            return
        # Smooth movement with lerp
        target_x = cx - self.pick_anchor_x
        target_y = cy - self.pick_anchor_y
        self.pick_offset_x = int(self.pick_offset_x * 0.5 + target_x * 0.5)
        self.pick_offset_y = int(self.pick_offset_y * 0.5 + target_y * 0.5)

    def drop_picked(self):
        if self.picked_region is None:
            return

        h, w = 720, 1280
        x1 = self.pick_bbox[0] + self.pick_offset_x
        y1 = self.pick_bbox[1] + self.pick_offset_y
        x2 = self.pick_bbox[2] + self.pick_offset_x
        y2 = self.pick_bbox[3] + self.pick_offset_y

        sx1 = max(0, x1)
        sy1 = max(0, y1)
        sx2 = min(w, x2)
        sy2 = min(h, y2)

        src_x1 = sx1 - x1
        src_y1 = sy1 - y1
        src_x2 = src_x1 + (sx2 - sx1)
        src_y2 = src_y1 + (sy2 - sy1)

        if src_x2 > src_x1 and src_y2 > src_y1:
            region = self.picked_region[src_y1:src_y2, src_x1:src_x2]
            mask = self.pick_mask[src_y1:src_y2, src_x1:src_x2]
            mask_3ch = cv2.merge([mask, mask, mask])
            roi = self.canvas[sy1:sy2, sx1:sx2]
            roi_clean = cv2.bitwise_and(roi, cv2.bitwise_not(mask_3ch))
            self.canvas[sy1:sy2, sx1:sx2] = cv2.bitwise_or(roi_clean, region)

        print("Dropped")
        self.picked_region = None
        self.pick_mask = None
        self.is_picking = False

    def draw_ui(self, frame):
        if self.is_eraser:
            cv2.putText(frame, "ERASER MODE", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        elif self.is_picking:
            cv2.putText(frame, "MOVING OBJECT", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 165, 0), 3)
        else:
            cv2.putText(frame, "Hand Tracking Drawing", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)

        if self.is_eraser:
            cv2.putText(frame, "5 fingers 2s to toggle", (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 1)
        elif self.is_picking:
            cv2.putText(frame, "Drag to move, Release to drop", (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        else:
            cv2.putText(frame, f"Color: {self.color_names[self.selected_color_idx]}", (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.putText(frame, f"Brush: {self.brush_sizes[self.brush_size_idx]}px", (30, 110),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        if self.is_eraser:
            cv2.putText(frame, f"Eraser: {self.eraser_thickness}px", (30, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
        elif not self.is_picking:
            cv2.putText(frame, "Index=Draw | 5fingers 2s=Eraser | Thumb+Index=Pick", (30, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        if self.all_fingers_start_time is not None and not self.is_eraser:
            elapsed = time.time() - self.all_fingers_start_time
            progress = min(elapsed / self.all_fingers_hold_sec, 1.0)
            bar_x, bar_y, bar_w, bar_h = 200, 140, 200, 15
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + bar_h), (0, 200, 255), -1)

        for i, color in enumerate(self.colors):
            x = self.color_panel_x
            y = self.color_panel_y + i * (self.color_box_size + self.color_gap)
            cv2.rectangle(frame, (x, y), (x + self.color_box_size, y + self.color_box_size), color, -1)
            cv2.rectangle(frame, (x, y), (x + self.color_box_size, y + self.color_box_size), (80, 80, 80), 2)
            if i == self.selected_color_idx:
                cv2.rectangle(frame, (x - 4, y - 4),
                    (x + self.color_box_size + 4, y + self.color_box_size + 4), (0, 255, 0), 3)

        sx, sy, sw, sh = self.clear_button_rect
        cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (50, 50, 50), -1)
        cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (100, 100, 100), 2)
        cv2.putText(frame, "CLEAR", (sx + 25, sy + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        sx, sy, sw, sh = self.save_button_rect
        cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (0, 0, 180), -1)
        cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
        cv2.putText(frame, "SAVE", (sx + 35, sy + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        bx, by = 30, 690
        for i, size in enumerate(self.brush_sizes):
            cv2.circle(frame, (bx + 25 + i * 45, by), max(2, size // 2), (150, 150, 150), -1)
            if i == self.brush_size_idx:
                cv2.circle(frame, (bx + 25 + i * 45, by), max(4, size // 2 + 2), (0, 255, 0), 2)

    def check_ui_interaction(self, x, y):
        for i in range(len(self.colors)):
            cx = self.color_panel_x
            cy = self.color_panel_y + i * (self.color_box_size + self.color_gap)
            if cx <= x <= cx + self.color_box_size and cy <= y <= cy + self.color_box_size:
                self.selected_color_idx = i
                self.draw_color = self.colors[i]
                print(f"Color: {self.color_names[i]}")
                return True

        sx, sy, sw, sh = self.clear_button_rect
        if sx <= x <= sx + sw and sy <= y <= sy + sh:
            self.canvas.fill(0)
            print("Canvas cleared")
            return True

        sx, sy, sw, sh = self.save_button_rect
        if sx <= x <= sx + sw and sy <= y <= sy + sh:
            os.makedirs("drawings", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join("drawings", f"drawing_{ts}.png")
            cv2.imwrite(path, self.canvas)
            print(f"Saved: {path}")
            return True

        bx, by = 30, 690
        for i in range(len(self.brush_sizes)):
            btx = bx + 25 + i * 45
            if abs(x - btx) < 20 and abs(y - by) < 20:
                self.brush_size_idx = i
                self.brush_thickness = self.brush_sizes[i]
                return True

        return False

    def smooth_point(self, x, y):
        self.points_buffer.append((x, y))
        if len(self.points_buffer) > self.smoothing:
            self.points_buffer.pop(0)
        avg_x = int(np.mean([p[0] for p in self.points_buffer]))
        avg_y = int(np.mean([p[1] for p in self.points_buffer]))
        return avg_x, avg_y

    def draw_hand_skeleton(self, frame, hand_landmarks, is_right=False):
        h, w = 720, 1280
        line_color = (0, 0, 200) if self.is_eraser else ((255, 140, 0) if is_right else (0, 180, 180))
        dot_color = (0, 0, 255) if self.is_eraser else ((255, 180, 0) if is_right else (0, 220, 220))

        for connection in HAND_CONNECTIONS:
            x1 = int(hand_landmarks[connection.start].x * w)
            y1 = int(hand_landmarks[connection.start].y * h)
            x2 = int(hand_landmarks[connection.end].x * w)
            y2 = int(hand_landmarks[connection.end].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), line_color, 2)

        for lm in hand_landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(frame, (x, y), 4, dot_color, -1)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_clicked = True
            self.mouse_x, self.mouse_y = x, y
            print(f"Mouse click at ({x}, {y})")

    def run(self):
        cv2.namedWindow("Hand Drawing App")
        cv2.setMouseCallback("Hand Drawing App", self.mouse_callback)

        print("=" * 50)
        print("  Hand Tracking Drawing App")
        print("=" * 50)

        while True:
            success, frame = self.cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            self.frame_count += 1
            timestamp_ms = int(self.frame_count * 33)

            result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)

            display = self.canvas.copy()

            if result.hand_landmarks:
                handedness = result.handedness if result.handedness else []

                for idx, hand_landmarks in enumerate(result.hand_landmarks):
                    is_right = False
                    if idx < len(handedness) and len(handedness[idx]) > 0:
                        is_right = handedness[idx][0].category_name == "Right"

                    self.draw_hand_skeleton(display, hand_landmarks, is_right)

                    fingers = self.fingers_up(hand_landmarks, is_right)
                    index_tip = self.lm_to_pixel(hand_landmarks[8], 1280, 720)
                    thumb_tip = self.lm_to_pixel(hand_landmarks[4], 1280, 720)
                    all_up = self.all_fingers_up(fingers)
                    pinch = self.is_pinch(hand_landmarks)
                    thumb_index_touch = self.is_thumb_index_touch(hand_landmarks)
                    
                    # Debug: print pinch distance
                    thumb_tip_lm = hand_landmarks[4]
                    index_tip_lm = hand_landmarks[8]
                    pinch_dist = np.sqrt(
                        (thumb_tip_lm.x - index_tip_lm.x)**2 + 
                        (thumb_tip_lm.y - index_tip_lm.y)**2
                    )

                    # Eraser toggle: 5 fingers for 2 sec
                    if all_up and not self.is_eraser and not self.is_picking:
                        if self.all_fingers_start_time is None:
                            self.all_fingers_start_time = time.time()
                            print("Hold 5 fingers... 2 sec to activate eraser")
                        elif time.time() - self.all_fingers_start_time >= self.all_fingers_hold_sec:
                            self.is_eraser = True
                            self.all_fingers_start_time = None
                            self.is_drawing = False
                            self.points_buffer = []
                            print("Eraser ON")
                    elif not all_up:
                        if self.is_eraser:
                            self.is_eraser = False
                            print("Eraser OFF")
                        self.all_fingers_start_time = None

                    if all_up and not self.is_eraser and not pinch:
                        self.is_drawing = False
                        self.points_buffer = []
                        continue

                    # Pinch handling
                    if pinch and not self.is_pinching:
                        self.is_pinching = True

                        # Toggle eraser off with pinch
                        if self.is_eraser:
                            self.is_eraser = False
                            print("Eraser OFF")
                            continue

                        # PICK: thumb+index touch near content
                        if self.has_content_at(index_tip[0], index_tip[1], radius=40):
                            if self.pick_region(index_tip[0], index_tip[1]):
                                continue

                        # UI interaction (only if no content nearby)
                        if self.check_ui_interaction(index_tip[0], index_tip[1]):
                            continue

                    elif not pinch:
                        # Release pinch = drop picked object
                        if self.is_picking:
                            self.drop_picked()
                        self.is_pinching = False

                    if self.is_picking and self.is_pinching:
                        self.move_picked(index_tip[0], index_tip[1])
                        continue

                    if self.is_picking:
                        continue

                    if self.is_pinching:
                        self.is_drawing = False
                        self.points_buffer = []
                        continue

                    # Eraser cursor
                    if self.is_eraser:
                        cv2.circle(display, index_tip, self.eraser_thickness, (0, 0, 255), 2)

                    # Hover preview: show highlight when near pickable content
                    if not self.is_eraser and not self.is_picking and not self.is_pinching:
                        if self.has_content_at(index_tip[0], index_tip[1], radius=40):
                            cv2.circle(display, index_tip, 25, (0, 255, 255), 2)
                            cv2.putText(display, "PINCH TO PICK", (index_tip[0]+30, index_tip[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        else:
                            cv2.circle(display, index_tip, 6, self.draw_color, -1)

                    # Draw or Erase mode
                    can_erase = self.is_eraser and fingers[1] == 1
                    can_draw = not self.is_eraser and fingers[1] == 1 and fingers[2] == 0

                    if can_erase or can_draw:
                        sx, sy = self.smooth_point(index_tip[0], index_tip[1])
                        if not self.is_drawing:
                            self.prev_x, self.prev_y = sx, sy
                            self.is_drawing = True
                        else:
                            if self.is_eraser:
                                cv2.line(display, (self.prev_x, self.prev_y), (sx, sy), (0, 0, 0), self.eraser_thickness)
                                cv2.line(self.canvas, (self.prev_x, self.prev_y), (sx, sy), (0, 0, 0), self.eraser_thickness)
                            else:
                                cv2.line(display, (self.prev_x, self.prev_y), (sx, sy), self.draw_color, self.brush_thickness)
                                cv2.line(self.canvas, (self.prev_x, self.prev_y), (sx, sy), self.draw_color, self.brush_thickness)
                            self.prev_x, self.prev_y = sx, sy
                    else:
                        self.is_drawing = False
                        self.points_buffer = []

            # Draw picked region with glow effect
            if self.is_picking and self.picked_region is not None:
                h, w = 720, 1280
                x1 = self.pick_bbox[0] + self.pick_offset_x
                y1 = self.pick_bbox[1] + self.pick_offset_y
                x2 = self.pick_bbox[2] + self.pick_offset_x
                y2 = self.pick_bbox[3] + self.pick_offset_y

                sx1 = max(0, x1)
                sy1 = max(0, y1)
                sx2 = min(w, x2)
                sy2 = min(h, y2)

                src_x1 = sx1 - x1
                src_y1 = sy1 - y1
                src_x2 = src_x1 + (sx2 - sx1)
                src_y2 = src_y1 + (sy2 - sy1)

                if src_x2 > src_x1 and src_y2 > src_y1:
                    region = self.picked_region[src_y1:src_y2, src_x1:src_x2]
                    mask = self.pick_mask[src_y1:src_y2, src_x1:src_x2]
                    mask_3ch = cv2.merge([mask, mask, mask])
                    roi = display[sy1:sy2, sx1:sx2]
                    roi_clean = cv2.bitwise_and(roi, cv2.bitwise_not(mask_3ch))
                    display[sy1:sy2, sx1:sx2] = cv2.bitwise_or(roi_clean, region)

                # Glow outline
                glow_color = (0, 200, 255)
                cv2.rectangle(display, (sx1-2, sy1-2), (sx2+2, sy2+2), glow_color, 2)
                cv2.rectangle(display, (sx1-4, sy1-4), (sx2+4, sy2+4), (0, 100, 150), 1)

            # Handle mouse click FIRST
            if self.mouse_clicked:
                self.check_ui_interaction(self.mouse_x, self.mouse_y)
                self.mouse_clicked = False

            self.draw_ui(display)
            cv2.imshow("Hand Drawing App", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                self.canvas.fill(0)
                print("Canvas cleared")
            elif key == ord("s"):
                os.makedirs("drawings", exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(os.path.join("drawings", f"drawing_{ts}.png"), self.canvas)
                print("Saved")
            elif key == 82:
                self.brush_size_idx = min(len(self.brush_sizes) - 1, self.brush_size_idx + 1)
                self.brush_thickness = self.brush_sizes[self.brush_size_idx]
            elif key == 84:
                self.brush_size_idx = max(0, self.brush_size_idx - 1)
                self.brush_thickness = self.brush_sizes[self.brush_size_idx]

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = HandDrawingApp()
    app.run()
