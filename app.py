import os
import tempfile
import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    from pdf2image import convert_from_path, pdfinfo_from_path
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helper Geometric & Mathematical Functions
# ---------------------------------------------------------------------------

def get_line_properties(line):
    """Calculates the length and 0-180 normalized angle of a line segment."""
    # Defensively flatten to guarantee a 1D array of 4 elements
    flat_line = np.ravel(line)
    if len(flat_line) != 4:
        return 0.0, 0.0
        
    x1, y1, x2, y2 = flat_line
    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
    return length, angle

def get_angle_distance(a1, a2):
    """Finds the absolute acute angular distance between two angles (0-180)."""
    diff = abs(a1 - a2)
    return min(diff, 180 - diff)

def find_intersection(line1, line2):
    """
    Finds the mathematical intersection point (x, y) of two infinite lines.
    Returns None if the lines are perfectly parallel.
    """
    flat1 = np.ravel(line1)
    flat2 = np.ravel(line2)
    if len(flat1) != 4 or len(flat2) != 4:
        return None

    x1, y1, x2, y2 = flat1
    x3, y3, x4, y4 = flat2

    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1 * x1 + B1 * y1

    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2 * x3 + B2 * y3

    det = A1 * B2 - A2 * B1

    if abs(det) < 1e-5:
        return None  # Parallel lines

    x = (B2 * C1 - B1 * C2) / det
    y = (A1 * C2 - A2 * C1) / det
    return int(round(x)), int(round(y))


# ---------------------------------------------------------------------------
# Core Image Processing Pipeline
# ---------------------------------------------------------------------------

def calculate_and_plot_angle_final(img, canny_low, canny_high, morph_kernel_size, max_area_pct):
    """
    Finds the diagram's oval contour, detects structural line fragments,
    mathematically extrapolates them, maps vectors to the line midpoints,
    and draws targeted interior/exterior arcs. Falls back gracefully to showing
    the localized ellipse if line tracking stages fail.
    """
    if img is None:
        return None, None, None, "Error: Image not found."

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    total_page_area = w * h

    # 1. Edge detection with user-tuned threshold profiles
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    
    # Apply morphological closing to bridge broken lines in the diagram boundary
    if morph_kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, None, "No structural diagram contours detected on this page with current thresholds."

    # Filter out contours that are too tiny (noise) or represent the page border/frame
    valid_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        area_pct = (area / total_page_area) * 100
        if area > 100 and area_pct < max_area_pct:
            valid_contours.append(c)

    if not valid_contours:
        best_contour = max(contours, key=cv2.contourArea)
    else:
        best_contour = max(valid_contours, key=cv2.contourArea)

    if len(best_contour) < 5:
        return None, None, None, "Selected boundary contour has too few coordinate points to fit an ellipse."

    ellipse = cv2.fitEllipse(best_contour)

    # INITIALIZE PLOT EARLY: Draw background and ellipse so it is always ready to display
    fig, ax = plt.subplots(figsize=(8, 8))
    vis_out = img_rgb.copy()
    cv2.ellipse(vis_out, ellipse, (0, 255, 0), 4)
    ax.imshow(vis_out)
    ax.axis('on')

    # 2. Mask out everything outside the ellipse to suppress background noise/text
    mask = np.zeros_like(gray)
    cv2.ellipse(mask, ellipse, (255), thickness=-1)
    masked_roi = cv2.bitwise_and(gray, gray, mask=mask)

    # Find edges specifically inside our isolated region of interest
    roi_edges = cv2.Canny(masked_roi, 50, 100)

    # 3. Dynamic Hough Line Detection
    lines = cv2.HoughLinesP(
        roi_edges, 
        rho=1,
        theta=np.pi/180,
        threshold=40,        
        minLineLength=35,    
        maxLineGap=20        
    )

    # FALLBACK 1: Ellipse isolated, but zero lines found
    if lines is None or len(lines) < 2:
        ax.set_title("Boundary Ellipse Localized (No Lines Detected)", fontsize=12, fontweight='bold')
        return fig, None, None, "Ellipse isolated successfully, but could not detect enough straight lines inside the area."

    # 4. Extract all segments and calculate properties
    all_lines = []
    for idx, raw_line in enumerate(lines):
        # FIX: Defensively flatten the array to prevent structure unpacking errors across versions
        coords = np.ravel(raw_line)
        if len(coords) != 4:
            continue
            
        length, angle = get_line_properties(coords)
        all_lines.append({
            'id': idx + 1,
            'coords': coords, 
            'length': length, 
            'angle': angle
        })
    
    # Sort descending by length (longest/strongest line strokes first)
    all_lines.sort(key=lambda x: x['length'], reverse=True)

    # Select the longest detected segment as Line 1
    line1_data = all_lines[0]
    line2_data = None

    # Find the longest remaining segment that represents a different trajectory (> 15 degrees apart)
    for candidate in all_lines[1:]:
        if get_angle_distance(candidate['angle'], line1_data['angle']) > 15:
            line2_data = candidate
            break

    # Plot Line 1 segment on the canvas since it was verified
    coords1 = line1_data['coords']
    ax.plot([coords1[0], coords1[2]], [coords1[1], coords1[3]], 
            color='red', linewidth=4, label=f"Line 1 Segment ({line1_data['angle']:.1f}°)")

    # FALLBACK 2: Found line fragments, but they are parallel
    if not line2_data:
        ax.set_title("Ellipse & Fragments Localized (Parallel/No Pair)", fontsize=12, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0, fontsize=9)
        return fig, None, all_lines, "Discovered lines, but they are completely parallel or too close to calculate an intersection."

    # Plot Line 2 segment on the canvas
    coords2 = line2_data['coords']
    ax.plot([coords2[0], coords2[2]], [coords2[1], coords2[3]], 
            color='orange', linewidth=4, label=f"Line 2 Segment ({line2_data['angle']:.1f}°)")

    # 5. Extrapolate paths to calculate where they intersect
    intersection = find_intersection(coords1, coords2)

    # FALLBACK 3: Infinite lines map to perfectly parallel data points
    if not intersection:
        ax.set_title("Trajectories Evaluated (No Stable Intersection)", fontsize=12, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0, fontsize=9)
        return fig, None, all_lines, "Line paths could not resolve to a stable geometric intersection vertex."

    # --- SUCCESS PATH: Plot tracking guidelines, markers, and arcs ---
    ix, iy = intersection
    
    # Plot infinite tracking guidelines
    first_guideline = True
    for data in [line1_data, line2_data]:
        c = data['coords']
        dx, dy = c[2] - c[0], c[3] - c[1]
        mag = np.sqrt(dx**2 + dy**2)
        ux, uy = dx / mag, dy / mag
        
        ext_x1, ext_y1 = int(ix + ux * 700), int(iy + uy * 700)
        ext_x2, ext_y2 = int(ix - ux * 700), int(iy - uy * 700)
        
        label_text = "Extrapolated Trajectories" if first_guideline else ""
        ax.plot([ext_x1, ext_x2], [ext_y1, ext_y2], color='cyan', linestyle='--', linewidth=1.5, alpha=0.5, label=label_text)
        first_guideline = False

    # --- TARGETED ARC SECTOR CALCULATION ---
    mx1, my1 = (coords1[0] + coords1[2]) / 2, (coords1[1] + coords1[3]) / 2
    mx2, my2 = (coords2[0] + coords2[2]) / 2, (coords2[1] + coords2[3]) / 2

    ang1 = np.degrees(np.arctan2(my1 - iy, mx1 - ix)) % 360
    ang2 = np.degrees(np.arctan2(my2 - iy, mx2 - ix)) % 360

    diff = (ang2 - ang1) % 360
    if diff <= 180:
        start_angle = ang1
        end_angle = ang1 + diff
        interior_angle = diff
    else:
        start_angle = ang2
        end_angle = ang2 + (360 - diff)
        interior_angle = 360 - diff

    exterior_angle = 360.0 - interior_angle

    # Draw Interior Arc
    radius_int = 75
    angles_int = np.linspace(np.radians(start_angle), np.radians(end_angle), 60)
    arc_x_int = ix + radius_int * np.cos(angles_int)
    arc_y_int = iy + radius_int * np.sin(angles_int)
    ax.plot(arc_x_int, arc_y_int, color='magenta', linewidth=3.5, label=f"Interior Arc ({interior_angle:.1f}°)")
    
    # Draw Exterior Arc
    radius_ext = 110
    angles_ext = np.linspace(np.radians(end_angle), np.radians(start_angle + 360), 120)
    arc_x_ext = ix + radius_ext * np.cos(angles_ext)
    arc_y_ext = iy + radius_ext * np.sin(angles_ext)
    ax.plot(arc_x_ext, arc_y_ext, color='lime', linewidth=3.5, label=f"Exterior Arc ({exterior_angle:.1f}°)")
    
    # Plot intersection star marker
    ax.scatter(ix, iy, color='gold', s=200, marker='*', zorder=6, 
               label=f"Vertex Intersection\nInt: {interior_angle:.1f}° | Ext: {exterior_angle:.1f}°")

    ax.set_title("Extrapolated Trajectory Vector Analysis", fontsize=12, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0, fontsize=9)

    return fig, interior_angle, all_lines, None


def process_pdf_page(pdf_path, page_number=0):
    pages = convert_from_path(pdf_path, first_page=page_number + 1, last_page=page_number + 1)
    open_cv_image = np.array(pages[0])
    img = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    return img


# ---------------------------------------------------------------------------
# Streamlit Web Application Interface
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Vision Angle Detector", layout="wide")
st.title("Vision Angle Detector")

# --- SIDEBAR CALIBRATION ENVIRONMENT ---
st.sidebar.header("🛠️ Contour Calibration Panel")
st.sidebar.markdown("Use these configurations if the green ellipse chooses page borders or misses the diagram shape entirely.",
                    "Please do not change it until and unless it not detecting anything.")

canny_low = st.sidebar.slider("Canny Threshold (Low)", min_value=10, max_value=150, value=30, step=5)
canny_high = st.sidebar.slider("Canny Threshold (High)", min_value=50, max_value=300, value=100, step=10)
morph_kernel_size = st.sidebar.slider("Line Stitching Thickness (Morphology)", min_value=1, max_value=15, value=5, step=2, help="Bridges structural breaks or tiny gaps in lines.")
max_area_pct = st.sidebar.slider("Max Allowed Contour Area (%)", min_value=50.0, max_value=100.0, value=90.0, step=1.0, help="Ignores contours that fill the whole page to kill page-border tracking bugs.")

uploaded_file = st.file_uploader("Upload an image or PDF file", type=["png", "jpg", "jpeg", "pdf"])

if uploaded_file is None:
    st.info("Please upload a supported image or document file to begin.")
    st.stop()

is_pdf = uploaded_file.type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf")
img_bgr = None

if is_pdf:
    if not PDF_AVAILABLE:
        st.error("pdf2image dependency isn't available. Please check system configurations.")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_pdf_path = tmp_file.name

    try:
        info = pdfinfo_from_path(tmp_pdf_path)
        num_pages = info["Pages"]
        st.success(f"Successfully read document — {num_pages} available page(s).")
        
        page_number_1indexed = st.selectbox(
            "Select target page to analyze",
            options=list(range(1, num_pages + 1)),
            format_func=lambda n: f"Page {n}",
        )

        with st.spinner(f"Extracting visual canvas from Page {page_number_1indexed}..."):
            img_bgr = process_pdf_page(tmp_pdf_path, page_number=page_number_1indexed - 1)
            
    except Exception as e:
        st.error(f"Error parsing document structure: {e}")
        st.stop()
    finally:
        if os.path.exists(tmp_pdf_path):
            os.remove(tmp_pdf_path)
else:
    pil_img = Image.open(uploaded_file).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

if img_bgr is None:
    st.error("Failed to construct a matrix array from the uploaded asset.")
    st.stop()


# ---------------------------------------------------------------------------
# Execution and Layout Rendering Block
# ---------------------------------------------------------------------------

with st.spinner("Computing contour geometry matrices and projecting intersecting vectors..."):
    fig, interior_angle, evaluated_lines, error = calculate_and_plot_angle_final(
        img_bgr, canny_low, canny_high, morph_kernel_size, max_area_pct
    )

st.subheader("Side-by-Side Diagram Comparison")
view_col1, view_col2 = st.columns(2)

with view_col1:
    st.markdown("**Original Page Viewport**")
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

with view_col2:
    st.markdown("**Calculated Mathematical Viewport**")
    if fig is not None:
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)  
        
        if error:
            st.warning(error)
        else:
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric(label="Interior Angle (Facing Lines)", value=f"{interior_angle:.2f}°")
            with m_col2:
                st.metric(label="Exterior Angle (Full Circle Remainder)", value=f"{(360.0 - interior_angle):.2f}°")
    else:
        st.error(error)

st.divider()
col_data1, col_data2 = st.columns(2)

with col_data1:
    with st.expander("📝 View Document Content Analysis (OCR)"):
        if OCR_AVAILABLE:
            try:
                page_text = pytesseract.image_to_string(img_bgr)
                if page_text.strip():
                    st.text_area("Discovered text blocks:", page_text, height=180)
                else:
                    st.caption("No discernible text layers identified.")
            except Exception:
                st.warning("Tesseract runtime execution environment binary missing or unconfigured.")
        else:
            st.caption("pytesseract package libraries are not configured.")

with col_data2:
    with st.expander("📊 View Data Log: All Line Segment Angles Evaluated"):
        if evaluated_lines is None:
            st.caption("No line vectors generated for evaluation.")
        else:
            formatted_table_data = []
            for line in evaluated_lines:
                formatted_table_data.append({
                    "Line ID": f"Segment #{line['id']}",
                    "Length": f"{line['length']:.1f} px",
                    "Angle (0-180°)": f"{line['angle']:.2f}°"
                })
            st.dataframe(formatted_table_data, use_container_width=True, height=180)