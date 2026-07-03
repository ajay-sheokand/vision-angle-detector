# Robust Diagram Angle Detector 📐✨

A computer vision-powered Streamlit web application that isolates schematic or diagrammatic regions within scanned documents or images, mathematically extrapolates intersecting line segments, and computes precise interior and exterior angles.

Built with **OpenCV**, **Streamlit**, and **Matplotlib**, this tool is explicitly engineered to handle real-world scanned artifacts, page borders, and low-contrast lines dynamically via an interactive calibration interface.


Below is the link for the Streamlit App:
https://vision-angle-detector.streamlit.app/

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