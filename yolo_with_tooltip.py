import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


class CrimeSceneInteractive:
    def __init__(self, model_weights='yolov8l.pt'):
        print(f"[INIT] Loading model: {model_weights}...")
        self.model = YOLO(model_weights)

        # Evidence Configuration
        self.evidence_classes = {
            0: "Person", 24: "Backpack", 26: "Handbag", 28: "Suitcase",
            39: "Bottle", 40: "Wine Glass", 41: "Cup", 43: "Knife",
            67: "Cell Phone", 76: "Scissors", 73: "Laptop"
        }

        # Colors (Normalized 0-1 for Matplotlib)
        # Red for Weapons, Blue for Digital, Yellow for General
        self.color_map = {
            "Weapon": (1.0, 0.2, 0.2),  # Red
            "Digital": (0.2, 0.6, 1.0),  # Blue
            "General": (1.0, 0.8, 0.0)  # Yellow
        }

    def analyze_and_view(self, image_path):
        """
        Runs inference and opens an interactive Matplotlib window.
        """
        # 1. Load and Prep Image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image not found.")
        # Convert BGR (OpenCV) to RGB (Matplotlib)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. Run Inference (Low confidence to find small details)
        results = self.model.predict(source=img, conf=0.45, verbose=False)[0]

        # 3. Extract Data for Plotting
        x_coords = []
        y_coords = []
        colors = []
        labels = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id in self.evidence_classes:
                # Calculate Center Point (The "Pin")
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                label_name = self.evidence_classes[cls_id]
                conf = float(box.conf[0])

                # Determine Category for Color
                if any(x in label_name for x in ["Knife", "Scissors", "Weapon"]):
                    c = self.color_map["Weapon"]
                elif any(x in label_name for x in ["Phone", "Laptop"]):
                    c = self.color_map["Digital"]
                else:
                    c = self.color_map["General"]

                x_coords.append(cx)
                y_coords.append(cy)
                colors.append(c)
                labels.append(f"{label_name}\n({conf:.1%})")

        # 4. Create Interactive Plot
        self._create_plot(img_rgb, x_coords, y_coords, colors, labels)

    def _create_plot(self, img, x, y, colors, labels):
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(img)
        ax.axis('off')  # Hide axes numbers

        # Plot the "Pins" (Scatter plot)
        # 'v' marker looks like a pin pointing down
        sc = ax.scatter(x, y, c=colors, s=150, marker='v', edgecolors='white', linewidths=1.5)

        # Create the Annotation (Hidden by default)
        annot = ax.annotate("", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="black", ec="white", alpha=0.8),
                            color='white', weight='bold',
                            arrowprops=dict(arrowstyle="->", color='white'))
        annot.set_visible(False)

        # --- INTERACTIVITY LOGIC ---
        def update_annot(ind):
            """Updates the tooltip content based on the index of the hovered point."""
            # Get the index of the point being hovered
            idx = ind["ind"][0]

            # Position the tooltip at the point
            pos = sc.get_offsets()[idx]
            annot.xy = pos

            # Set the text
            text = labels[idx]
            annot.set_text(text)

            # Match tooltip color to pin color (optional, looks nice)
            # annot.get_bbox_patch().set_facecolor(colors[idx])

        def hover(event):
            """Event handler for mouse movement."""
            vis = annot.get_visible()
            if event.inaxes == ax:
                # Check if mouse is over a point
                cont, ind = sc.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()

        # Connect the hover event to the figure
        fig.canvas.mpl_connect("motion_notify_event", hover)

        print("[INFO] Viewer opened. Hover over pins to see details.")
        plt.title("Interactive Evidence Map (Hover for Info)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


# --- EXECUTION ---
if __name__ == "__main__":
    # 1. Initialize
    app = CrimeSceneInteractive(model_weights='yolov8l.pt')

    # 2. Run on your image
    # Replace 'image_14d3de.png' with your actual file path
    try:
        app.analyze_and_view("./crime_scenes/1.jpeg")
    except Exception as e:
        print(f"Error: {e}")