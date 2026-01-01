import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
from datetime import datetime

# Load pretrained YOLO model (downloads automatically first time)
model = YOLO('yolov8n.pt')

class BorderSurveillance:
    def __init__(self):
        self.ssim_threshold = 0.6      # Large changes [web:56]
        self.min_contour_area = 500    # Human/vehicle size [web:62]
        self.human_temp_threshold = 37 # Body heat °C [web:68]
        
    def preprocess_image(self, img):
        """Resize + grayscale for consistent comparison"""
        img = cv2.resize(img, (640, 480))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray, img
    
    def detect_changes_ssim(self, historical_gray, latest_gray):
        """SSIM change detection with contours"""
        score, diff = ssim(historical_gray, latest_gray, full=True)
        diff = (diff * 255).astype(np.uint8)
        contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        change_regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.min_contour_area:
                x, y, w, h = cv2.boundingRect(cnt)
                change_regions.append((x, y, w, h, area))
                cv2.rectangle(latest_gray, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        return score, diff, change_regions, latest_gray
    
    def yolo_detection(self, img):
        """Detect humans/vehicles in change regions"""
        results = model(img, conf=0.5)
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls in [0, 2]:  # 0=person, 2=car [web:65]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detections.append((cls, conf, (int(x1), int(y1), int(x2-x1), int(y2-y1))))
        return detections
    
    def thermal_anomaly_check(self, thermal_img, bbox):
        """Check body heat in bounding box"""
        x, y, w, h = bbox
        roi = thermal_img[y:y+h, x:x+w]
        if roi.size > 0:
            avg_temp = np.mean(roi)  # Assuming calibrated thermal [web:68]
            return avg_temp > self.human_temp_threshold
        return False
    
    def analyze_pair(self, historical_rgb, historical_thermal, latest_rgb, latest_thermal, gps_coord):
        """Full analysis pipeline"""
        # Preprocess
        hist_rgb_gray, hist_rgb = self.preprocess_image(historical_rgb)
        hist_thermal_gray, hist_thermal = self.preprocess_image(historical_thermal)
        latest_rgb_gray, latest_rgb_annotated = self.preprocess_image(latest_rgb)
        latest_thermal_gray, latest_thermal_annotated = self.preprocess_image(latest_thermal)
        
        # Step 1: SSIM change detection [web:56]
        rgb_ssim, rgb_diff, rgb_changes, rgb_annotated = self.detect_changes_ssim(hist_rgb_gray, latest_rgb_gray)
        thermal_ssim, thermal_diff, thermal_changes, thermal_annotated = self.detect_changes_ssim(hist_thermal_gray, latest_thermal_gray)
        
        alert = False
        alert_msg = []
        
        # Step 2: Check significant changes
        if rgb_ssim < self.ssim_threshold or thermal_ssim < self.ssim_threshold:
            print(f"🚨 CHANGE DETECTED at GPS {gps_coord}: RGB SSIM={rgb_ssim:.3f}, Thermal SSIM={thermal_ssim:.3f}")
            
            # Step 3: YOLO on change regions [web:65]
            detections = self.yolo_detection(latest_rgb)
            if detections:
                for cls, conf, bbox in detections:
                    class_name = "PERSON" if cls == 0 else "VEHICLE"
                    print(f"  → {class_name} detected (conf={conf:.2f})")
                    
                    # Step 4: Thermal confirmation [web:68]
                    if self.thermal_anomaly_check(latest_thermal, bbox):
                        alert = True
                        alert_msg.append(f"HUMAN at GPS {gps_coord} ({class_name}, {conf:.2f})")
                    else:
                        print(f"  → Heat signature not human-level")
        
        return {
            'alert': alert,
            'alert_msg': alert_msg,
            'rgb_ssim': rgb_ssim,
            'thermal_ssim': thermal_ssim,
            'annotated_rgb': rgb_annotated,
            'annotated_thermal': thermal_annotated
        }
    
    def visualize_results(self, results, gps_coord):
        """Show before/after with annotations"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes[0,0].imshow(results['annotated_rgb'], cmap='gray')
        axes[0,0].set_title(f'Latest RGB - GPS {gps_coord}
SSIM: {results["rgb_ssim"]:.3f}')
        axes[0,0].axis('off')
        
        axes[0,1].imshow(results['annotated_thermal'], cmap='hot')
        axes[0,1].set_title(f'Latest Thermal - GPS {gps_coord}
SSIM: {results["thermal_ssim"]:.3f}')
        axes[0,1].axis('off')
        
        axes[1,0].text(0.5, 0.5, 'HISTORICAL (Baseline)', ha='center', va='center', transform=axes[1,0].transAxes)
        axes[1,0].axis('off')
        axes[1,1].axis('off')
        
        if results['alert']:
            axes[1,1].text(0.1, 0.5, '🚨 SUSPICIOUS ACTIVITY
' + '
'.join(results['alert_msg']), 
                          fontsize=14, color='red')
        else:
            axes[1,1].text(0.1, 0.5, '✅ Normal Activity', fontsize=14, color='green')
        
        plt.tight_layout()
        plt.savefig(f'alert_gps_{gps_coord}_{datetime.now().strftime("%H%M%S")}.png')
        plt.show()

# ====================================
# USAGE: Replace with your image paths [conversation_history:6][web:36]
# ====================================
surveillance = BorderSurveillance()

# Waypoint 1: Download sample from FLIR dataset or use your data/baseline/
historical_rgb = cv2.imread('data/baseline/border_rgb.jpg')
historical_thermal = cv2.imread('data/baseline/border_thermal.jpg')
latest_rgb = cv2.imread('data/latest/border_rgb.jpg')
latest_thermal = cv2.imread('data/latest/border_thermal.jpg')

if all([historical_rgb, historical_thermal, latest_rgb, latest_thermal]) is not None:
    gps_coord = "28.6139,77.2090"  # Sample Agra border coord
    results = surveillance.analyze_pair(historical_rgb, historical_thermal, 
                                       latest_rgb, latest_thermal, gps_coord)
    surveillance.visualize_results(results, gps_coord)
    
    if results['alert']:
        print("🚨 REPORT SENT TO AUTHENTICATOR:", results['alert_msg'])
    else:
        print("✅ All clear at", gps_coord)
else:
    print("❌ Load images first! Paths: data/baseline/* and data/latest/*")
