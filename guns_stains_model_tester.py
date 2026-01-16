import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os


class CustomModelTester:
    def __init__(self, weights_path):
        """
        Initialize with the path to your custom trained .pt file
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Custom weights not found at: {weights_path}")

        print(f"[INIT] Loading custom model from: {weights_path}...")
        self.model = YOLO(weights_path)

        # Define colors for your specific classes
        # Red for Gun, Dark Red/Maroon for Blood
        self.colors = {
            "gun": (255, 0, 0),  # Bright Red
            "blood_stain": (139, 0, 0)  # Dark Red
        }

    def test_image(self, image_path, conf_threshold=0.4):
        """
        Runs inference on a single image and displays the result.
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"[ERROR] Could not load image: {image_path}")
            return

        # Convert to RGB for Matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run Prediction
        # conf=0.4 filters out weak detections
        results = self.model.predict(source=img, conf=conf_threshold, verbose=False)[0]

        # Prepare Plot
        plt.figure(figsize=(12, 8))
        plt.imshow(img_rgb)
        ax = plt.gca()
        plt.axis('off')

        detected_count = 0

        print(f"\n--- TESTING REPORT for {os.path.basename(image_path)} ---")

        # Iterate through detections
        for box in results.boxes:
            # Get Class ID and Name
            cls_id = int(box.cls[0])
            class_name = results.names[cls_id]  # e.g., 'gun' or 'blood_stain'
            conf = float(box.conf[0])

            # Get Coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Select Color (Default to Yellow if unknown class)
            color_rgb = self.colors.get(class_name, (255, 255, 0))
            # Normalize color for Matplotlib (0-1 range)
            color_plot = [c / 255.0 for c in color_rgb]

            # 1. Draw Bounding Box
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, color=color_plot, linewidth=3
            )
            ax.add_patch(rect)

            # 2. Draw Label
            label_text = f"{class_name.upper()} {conf:.1%}"
            ax.text(
                x1, y1 - 5, label_text,
                color='white', fontsize=10, fontweight='bold',
                bbox=dict(facecolor=color_plot, alpha=0.8, edgecolor='none', pad=2)
            )

            print(f" > Detected: {class_name} ({conf:.1%})")
            detected_count += 1

        if detected_count == 0:
            print(" > No guns or blood stains detected.")
            plt.title(f"Result: No Detections (Threshold {conf_threshold})", fontsize=14, color='red')
        else:
            plt.title(f"Result: Detected {detected_count} Item(s)", fontsize=14, color='green')

        plt.show()


# --- EXECUTION ---
if __name__ == "__main__":
    # 1. PATH TO YOUR TRAINED WEIGHTS
    # Update this to point to your actual .pt file
    CUSTOM_WEIGHTS = "runs/detect/gun_blood_model4/weights/best.pt"

    # 2. PATH TO TEST IMAGE
    TEST_IMAGE = "crime_scenes/scene_with_gun.png"

    # Run Test
    try:
        # If you haven't trained it yet and just want to test the code structure,
        # you can temporarily swap 'best.pt' with 'yolov8n.pt', but it won't detect guns/blood.
        tester = CustomModelTester(CUSTOM_WEIGHTS)
        tester.test_image(TEST_IMAGE, conf_threshold=0.35)  # Lower threshold for testing
    except Exception as e:
        print(f"System Error: {e}")