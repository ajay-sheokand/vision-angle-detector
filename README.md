# Robust Diagram Angle Detector 📐✨

A computer vision-powered Streamlit web application that isolates schematic or diagrammatic regions within scanned documents or images, mathematically extrapolates intersecting line segments, and computes precise interior and exterior angles.

Built with **OpenCV**, **Streamlit**, and **Matplotlib**, this tool is explicitly engineered to handle real-world scanned artifacts, page borders, and low-contrast lines dynamically via an interactive calibration interface.

---

## 🚀 Key Features

* **Document-Agnostic Input:** Seamlessly upload and process multi-page PDFs, PNGs, JPGs, or JPEGs.
* **Dimension-Agnostic Processing:** Utilizes array-flattening strategies to remain fully compatible across varying local and cloud versions of OpenCV (`opencv-python` vs `opencv-python-headless`).
* **Advanced Geometry Pipeline:** 
  * Locates and isolates the primary diagram using automated ellipse-fitting contour segmentation.
  * Automatically filters out page margins, text blocks, and document scanning frames.
  * Extrapolates line segments mathematically to resolve hidden vertex intersection points.
* **Reflex Angle Analytics:** Visualizes both the acute interior sweep and the full-circle exterior remainder (Exterior Angle = 360° - Interior Angle).
* **Cascade Fallback Engineering:** If severe artifacting or noise prevents line segment parsing, the app gracefully falls back to displaying your localized boundary ellipse overlay instead of throwing a fatal execution error.
* **Interactive Calibration Panel:** Real-time sidebar sliders for tuning edge-detection thresholds, morphological line-stitching, and boundary size limits.
* **Integrated Data Logs:** Features an on-the-fly table displaying evaluated segment lengths and normalized headings alongside optional Tesseract-powered OCR text blocks.

---

## 🛠️ System Requirements & Prerequisites

Because this application relies on low-level system bindings for rendering documents and processing text, you must ensure the following non-Python binary dependencies are available on your host environment:

1. **Poppler:** Required by `pdf2image` to unpack and rasterize PDF pages.
2. **Tesseract OCR (Optional):** Required by `pytesseract` to view document text layouts.

### Local Environment Setup

#### 🍏 macOS (via Homebrew)
```bash
brew install poppler tesseract
🐧 Linux (Ubuntu/Debian)
Bash
sudo apt-get update
sudo apt-get install -y poppler-utils tesseract-ocr
🪟 Windows
Download a pre-compiled Windows binary package for Poppler and Tesseract.

Extract the folders to a permanent location (e.g., C:\Program Files\poppler).

Add the path to the internal bin/ directories to your Windows System Environment PATH Variable.

💻 Local Installation & Deployment
Follow these quick steps to set up and run the app locally on your machine:

Clone the Repository:

Bash
git clone [https://github.com/YOUR_USERNAME/vision-angle-detector.git](https://github.com/YOUR_USERNAME/vision-angle-detector.git)
cd vision-angle-detector
Install Python Libraries:

Bash
pip install -r requirements.txt
Run the Streamlit Application:

Bash
streamlit run app.py
☁️ Deploying to Streamlit Community Cloud
When hosting this repository on the web using Streamlit Community Cloud, system binary dependencies must be declared so the server environment can provision them automatically.

Your repository folder layout should contain these files in the root directory:

Plaintext
├── app.py
├── requirements.txt
└── packages.txt
1. requirements.txt
Ensure your core Python packages are explicitly listed:

Plaintext
opencv-python-headless
numpy
matplotlib
streamlit
pillow
pdf2image
pytesseract
2. packages.txt
Create a file exactly named packages.txt in the root folder to handle background Linux system packages:

Plaintext
poppler-utils
tesseract-ocr
🎛️ Contour Calibration Guide
If a specific document layout skips the targeted diagram or accidentally picks up text lines, use the interactive Sidebar Calibration Panel to tune the computer vision engine in real time:

Canny Thresholds (Low/High): Lower these values if your target lines are faint or low-contrast. Raise them to ignore light background grids or artifact noise.

Line Stitching Thickness (Morphology): Increasing this parameter applies a mathematical MORPH_CLOSE operation, blending broken ink layers or micro-gaps together into continuous contours.

Max Allowed Contour Area (%): Set this below 100% (e.g., 90%) to instantly force the engine to ignore contours that match the exact page borders, dropping straight onto the internal diagram assets.