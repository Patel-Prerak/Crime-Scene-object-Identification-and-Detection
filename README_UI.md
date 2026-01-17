# 🔍 Crime Scene Evidence Analyzer

A professional, AI-powered forensic evidence detection and classification system with a stunning modern UI.

## ✨ Features

### 🎨 **Ultra-Professional UI Design**
- **Animated Gradient Headers** with glassmorphism effects
- **Interactive Stat Boxes** with hover animations
- **Color-Coded Evidence Badges** with pulsing animations for weapons
- **Modern Card Layouts** with depth and shadows
- **Smooth Transitions** and micro-animations throughout
- **Custom Gradient Scrollbars** matching the theme
- **Responsive Design** that looks great on all screen sizes

### 🤖 **AI Detection Capabilities**
- **Dual Model System:**
  - YOLOv8 Large (Standard Model) - General object detection
  - Custom Forensic Model - Specialized weapon & blood detection
  
- **Detectable Evidence:**
  - 👤 Persons & Suspects
  - 🔫 Weapons (Guns, Knives)
  - 🩸 Blood Stains
  - 📱 Digital Evidence (Phones, Laptops)
  - 🎒 Personal Items (Backpacks, Handbags, Suitcases)
  - 🍾 Bottles & Containers

### 📊 **Advanced Analysis**
- Real-time evidence detection with confidence scores
- Visual annotations on detected objects
- Comprehensive evidence summary dashboard
- Detailed statistical metrics
- Exportable CSV reports
- Downloadable annotated images

## 🚀 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/paarth-birla/Crime-Scene-Evidence-Classification
cd "crimse scene objects identifier and classifier"
```

2. **Install dependencies:**
```bash
pip install streamlit ultralytics opencv-python pandas pillow numpy
```

3. **Run the application:**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 How to Use

1. **Upload Image:** Click the upload button and select a crime scene photograph (JPG, JPEG, or PNG)
2. **Analyze:** Click the "🔍 Analyze Evidence" button to process the image
3. **Review Results:** View detected evidence with confidence scores and visual annotations
4. **Download Reports:** Export detailed CSV reports and annotated images

## 🎨 UI Highlights

### **Color Scheme**
- **Primary Gradient:** Purple to violet (#667eea → #764ba2)
- **Weapon Badges:** Red gradient with pulsing animation
- **Person Badges:** Blue gradient
- **Digital Evidence:** Cyan to pink gradient
- **General Items:** Green to blue gradient

### **Animations**
- Header pulsing glow effect
- Rotating background patterns
- Hover scale and lift effects
- Smooth color transitions
- Pulsing weapon badges for critical alerts

### **Typography**
- **Font:** Inter (Google Fonts)
- **Headers:** 800 weight, large sizing
- **Body:** 400-600 weight for readability

## 🔧 Configuration

- **Confidence Threshold:** 30% (adjustable in code)
- **Visual Cutoff:** Only detections above 30% are displayed
- **All detections logged:** Even low-confidence detections are saved in reports

## 📊 Evidence Summary Dashboard

The results page includes:
- **Total Evidence Count** with high-confidence filter
- **Weapons Detected** with critical alert
- **Persons Identified** count
- **Average Confidence Score** across all detections
- **Color-Coded Badges** for each evidence type
- **Detailed Table** with model source and confidence
- **Download Options** for CSV and annotated images

## ⚠️ Important Notes

- This system uses advanced AI models for evidence detection
- Results should be verified by forensic professionals
- Minimum confidence threshold is set to 30% for visualization
- All detections are logged regardless of confidence level

## 🛠️ Technical Stack

- **Frontend:** Streamlit with custom CSS
- **AI Models:** YOLOv8 (Ultralytics)
- **Image Processing:** OpenCV, PIL
- **Data Handling:** Pandas, NumPy
- **Styling:** Custom CSS with gradients and animations

## 📝 License

This project is part of the Crime Scene Evidence Classification system.

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- Streamlit framework
- Google Fonts (Inter)

## 👥 Contributors

- **devhl** - *Contributor*
