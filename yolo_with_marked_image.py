import cv2
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import os
import glob


class CrimeSceneBatchDetector:
    def __init__(self, model_weights='yolov8l.pt'):
        """
        Initialize the detector with the Large YOLO model.
        """
        print(f"[INIT] Loading forensic model: {model_weights}...")
        self.model = YOLO(model_weights)

        # --- CONFIGURATION ---
        self.CONFIDENCE_CUTOFF = 0.45

        # Evidence Classes
        self.evidence_classes = {
            0: "Person (Suspect/Victim)",
            24: "Backpack", 26: "Handbag", 28: "Suitcase",
            39: "Bottle", 40: "Wine Glass", 41: "Cup",
            43: "Knife", 67: "Cell Phone", 76: "Scissors", 73: "Laptop"
        }

        # Professional Colors
        self.colors = {
            "background": (66, 44, 26),  # Midnight Navy
            "weapon": (44, 42, 93),  # Deep Burgundy
            "digital": (97, 77, 59),  # Slate Blue
            "general": (80, 62, 44),  # Evergreen
            "text": (255, 255, 255)  # White
        }

    def process_directory(self, input_dir, output_root='evidence_reports'):
        """
        Iterates through a directory of images and processes them one by one.
        """
        # 1. Setup Output Directory Structure
        visuals_dir = os.path.join(output_root, "visuals")
        official_dir = os.path.join(output_root, "official_logs")
        debug_dir = os.path.join(output_root, "debug_data")

        for d in [visuals_dir, official_dir, debug_dir]:
            os.makedirs(d, exist_ok=True)

        # 2. Find All Images
        # Looks for jpg, jpeg, png (case in-sensitive usually, but explicit here)
        image_files = glob.glob(input_dir)

        print(f"[INFO] Found {len(image_files)} images in '{input_dir}'")

        # 3. Process Each Image
        for i, img_path in enumerate(image_files):
            print(f"\n[{i + 1}/{len(image_files)}] Processing: {os.path.basename(img_path)}...")
            self._analyze_single_image(img_path, visuals_dir, official_dir, debug_dir)

        print(f"\n[COMPLETE] Batch processing finished. Results in '{output_root}/'")

    def _analyze_single_image(self, image_path, visuals_dir, official_dir, debug_dir):
        """
        Internal helper to process one image and save its 3 specific output files.
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"[ERROR] Skipped corrupt file: {image_path}")
            return

        # Get filename without extension for saving
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # --- INFERENCE (Get Everything) ---
        results = self.model.predict(source=img, conf=0.001, iou=0.5, verbose=False)[0]

        debug_log_all = []
        official_evidence_log = []
        annotated_img = img.copy()
        class_names = results.names

        # --- PROCESSING ---
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            raw_label = class_names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            is_evidence_type = cls_id in self.evidence_classes
            evidence_label = self.evidence_classes.get(cls_id, raw_label)

            # STREAM A: DEBUG LOG (All Detections)
            debug_log_all.append({
                "Source_Image": base_name,
                "Object_Type": raw_label,
                "Confidence": conf,
                "Is_Evidence_Class": is_evidence_type,
                "Box": [x1, y1, x2, y2]
            })

            # STREAM B: OFFICIAL LOGIC (>45% & Evidence Class)
            if is_evidence_type and conf > self.CONFIDENCE_CUTOFF:

                # Add to CSV
                official_evidence_log.append({
                    "Timestamp": datetime.now().isoformat(),
                    "Source_Image": base_name,
                    "Evidence_Type": evidence_label,
                    "Confidence": f"{conf:.2%}",
                    "Location": [x1, y1, x2, y2]
                })

                # Draw Visuals
                if "Weapon" in evidence_label or "Knife" in evidence_label:
                    c = self.colors["weapon"]
                elif "Digital" in evidence_label or "Phone" in evidence_label:
                    c = self.colors["digital"]
                else:
                    c = self.colors["general"]

                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), c, 2)

                # Label
                lbl_text = f"{evidence_label.split('(')[0]} {conf:.0%}"
                (w, h), _ = cv2.getTextSize(lbl_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                cv2.rectangle(annotated_img, (x1, y1 - h - 10), (x1 + w + 10, y1), self.colors["background"], -1)
                cv2.putText(annotated_img, lbl_text, (x1 + 5, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.colors["text"], 3)

        # --- SAVING ---

        # 1. Save Visual Image (Only if evidence found, or always? Let's save always to prove we checked)
        out_img_path = os.path.join(visuals_dir, f"{base_name}_VISUAL.jpg")
        cv2.imwrite(out_img_path, annotated_img)

        # 2. Save Official CSV (Only if evidence found)
        if official_evidence_log:
            df_off = pd.DataFrame(official_evidence_log)
            out_csv_path = os.path.join(official_dir, f"{base_name}_EVIDENCE.csv")
            df_off.to_csv(out_csv_path, index=False)
            print(f"   -> Evidence Found! Saved report to official logs.")
        else:
            print(f"   -> Clean scene (No high-confidence evidence).")

        # 3. Save Debug CSV (Always)
        if debug_log_all:
            df_debug = pd.DataFrame(debug_log_all)
            df_debug = df_debug.sort_values(by="Confidence", ascending=False)
            out_debug_path = os.path.join(debug_dir, f"{base_name}_DEBUG.csv")
            df_debug.to_csv(out_debug_path, index=False)


# --- EXECUTION ---
if __name__ == "__main__":
    # Initialize
    app = CrimeSceneBatchDetector(model_weights='yolov8l.pt')

    # DEFINE YOUR INPUT DIRECTORY HERE
    # Make sure this folder exists and contains images
    INPUT_FOLDER = "crime_scenes/*"

    try:
        app.process_directory(INPUT_FOLDER)
    except Exception as e:
        print(f"Error: {e}")