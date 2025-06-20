from pathlib import Path
from PIL import Image, ImageOps, UnidentifiedImageError
import logging
import cv2 # OpenCV for advanced image processing
import numpy as np # For numerical operations with OpenCV

logger = logging.getLogger(__name__)


def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders a list of 4 points (quadrilateral corners) in top-left, 
    top-right, bottom-right, bottom-left order.
    """
    # Initialize a list of coordinates that will be ordered
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def _find_slide_corners_and_warp(img_np: np.ndarray, ratio_thresh=0.1, area_thresh_factor=0.1):
    """
    Tries to find the four corners of a slide in an image and returns the
    perspective-corrected (warped) image of the slide content.

    Args:
        img_np: Image as a NumPy array (from cv2.imread or PIL conversion).
        ratio_thresh: Threshold for aspect ratio deviation from common slide ratios.
        area_thresh_factor: Minimum area of the contour relative to image area.

    Returns:
        A NumPy array of the warped slide image if successful, otherwise None.
    """
    if img_np is None:
        logger.error("[image_utils_warp] Input image is None for corner finding.")
        return None

    img_h_orig, img_w_orig = img_np.shape[:2]
    # Make a copy for operations that might modify the image (like drawing contours, though not done here)
    # and ensure it's BGR for OpenCV
    if len(img_np.shape) == 2 or img_np.shape[2] == 1: # Grayscale image
        image_for_processing = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    elif img_np.shape[2] == 4: # RGBA image
        image_for_processing = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
    else:
        image_for_processing = img_np.copy()


    # Preprocessing
    gray = cv2.cvtColor(image_for_processing, cv2.COLOR_BGR2GRAY)
    # Experiment with blur settings
    blur = cv2.GaussianBlur(gray, (5, 5), 0) 
    
    # Experiment with Canny edge detection thresholds
    edged = cv2.Canny(blur, 50, 150)
    # Optional: Dilate and Erode to close gaps in edges
    # edged = cv2.dilate(edged, None, iterations=1)
    # edged = cv2.erode(edged, None, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        logger.warning("[image_utils_warp] No contours found.")
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10] # Check top 10 largest

    slide_screen_cnt = None
    min_area_threshold = img_w_orig * img_h_orig * area_thresh_factor

    for c in contours:
        peri = cv2.arcLength(c, True)
        # Adjust epsilon for approxPolyDP; 0.02 is a common start, can be tuned
        approx_poly = cv2.approxPolyDP(c, 0.02 * peri, True) 

        if len(approx_poly) == 4 and cv2.isContourConvex(approx_poly) and \
           cv2.contourArea(approx_poly) > min_area_threshold:
            
            # Basic check for aspect ratio of the bounding box of the contour
            # This helps filter out very elongated or oddly shaped quadrilaterals
            x_br, y_br, w_br, h_br = cv2.boundingRect(approx_poly)
            if w_br == 0 or h_br == 0: continue
            aspect_ratio_br = float(w_br) / h_br
            
            # Define typical slide aspect ratios and allowed deviation
            common_ratios = [4/3, 16/9, 16/10, 3/2] # Add more if needed
            is_good_ratio = False
            for r_common in common_ratios:
                if abs(aspect_ratio_br - r_common) < ratio_thresh:
                    is_good_ratio = True
                    break
            
            if not is_good_ratio:
                # logger.debug(f"[image_utils_warp] Contour skipped due to bounding box aspect ratio: {aspect_ratio_br:.2f}")
                continue
            
            slide_screen_cnt = approx_poly
            logger.info(f"[image_utils_warp] Found potential slide contour with {len(slide_screen_cnt)} points and area {cv2.contourArea(slide_screen_cnt):.0f}")
            break # Found a good candidate

    if slide_screen_cnt is None:
        logger.warning("[image_utils_warp] No suitable 4-point contour found for slide.")
        return None

    # Order the points of the contour
    ordered_screen_pts = _order_points(slide_screen_cnt.reshape(4, 2))
    
    (tl, tr, br, bl) = ordered_screen_pts
    
    # Calculate width of the new warped image
    width_A = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2)) # Bottom edge
    width_B = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2)) # Top edge
    max_width = max(int(width_A), int(width_B))

    # Calculate height of the new warped image
    height_A = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2)) # Right edge
    height_B = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2)) # Left edge
    max_height = max(int(height_A), int(height_B))

    if max_width <= 0 or max_height <= 0:
        logger.warning(f"[image_utils_warp] Calculated max_width ({max_width}) or max_height ({max_height}) for warp is zero or negative.")
        return None

    # Destination points for the perspective transform
    dst_pts = np.array([
        [0, 0],                  # Top-left
        [max_width - 1, 0],      # Top-right
        [max_width - 1, max_height - 1], # Bottom-right
        [0, max_height - 1]],    # Bottom-left
        dtype="float32")

    # Compute the perspective transform matrix and apply it
    # Use the original color image (or a BGR copy of it) for warping
    transform_matrix = cv2.getPerspectiveTransform(ordered_screen_pts, dst_pts)
    warped_slide = cv2.warpPerspective(image_for_processing, transform_matrix, (max_width, max_height))
    
    logger.info(f"[image_utils_warp] Slide warped to dimensions: {warped_slide.shape[1]}x{warped_slide.shape[0]}")
    return warped_slide


def _calculate_resize_dimensions(original_width, original_height, target_width, target_height):
    """
    Calculates new dimensions to fit an image within target_width and target_height
    while maintaining its aspect ratio.
    """
    original_ratio = original_width / original_height
    target_ratio = target_width / target_height

    if target_ratio > original_ratio:
        # Target is wider than original (relative to height), so fit to height
        new_height = target_height
        new_width = int(original_ratio * new_height)
    else:
        # Target is narrower than original, or same aspect ratio, so fit to width
        new_width = target_width
        new_height = int(new_width / original_ratio)

    return new_width, new_height

def prepare_slide_fullscreen(
    image_path: Path,
    output_dir: Path,
    target_width: int,
    target_height: int,
    canvas_bg_color: tuple = (0, 0, 0) # Default black
) -> Path | None:
    """
    Processes a slide image:
    1. (Placeholder for cropping) Crops the image to isolate slide content.
    2. Resizes the (cropped) slide to fit within target_width and target_height, maintaining aspect ratio.
    3. Creates a new canvas of target_width x target_height with canvas_bg_color.
    4. Pastes the resized slide onto the center of the canvas.
    5. Saves the resulting image to output_dir.

    Args:
        image_path: Path to the input image.
        output_dir: Directory to save the processed image.
        target_width: Width of the output canvas.
        target_height: Height of the output canvas.
        canvas_bg_color: Background color for the canvas (RGB tuple).

    Returns:
        Path to the saved processed image, or None if processing fails.
    """
    if not image_path.is_file():
        logger.error(f"[image_utils] Image file not found: {image_path}")
        return None

    pil_img = None
    try:
        pil_img = Image.open(image_path)
        pil_img = pil_img.convert("RGB") # Ensure image is in RGB for consistency
    except UnidentifiedImageError:
        logger.error(f"[image_utils] Cannot identify image file (possibly corrupt or unsupported format): {image_path}")
        return None
    except Exception as e:
        logger.error(f"[image_utils] Error opening image {image_path.name} with Pillow: {e}")
        return None

    # Convert PIL image to OpenCV format for cropping
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # --- 1. Cropping (OpenCV-based implementation with perspective warp) ---
    logger.info(f"[image_utils] Attempting to find and warp slide content for {image_path.name}")
    # _find_slide_corners_and_warp expects a BGR image for OpenCV processing
    warped_cv_slide = _find_slide_corners_and_warp(cv_img) 
    
    source_for_resize_pil = None
    if warped_cv_slide is not None and warped_cv_slide.size > 0:
        # Convert the (color) warped OpenCV image back to PIL Image for resizing and pasting
        try:
            source_for_resize_pil = Image.fromarray(cv2.cvtColor(warped_cv_slide, cv2.COLOR_BGR2RGB))
            logger.info(f"[image_utils] Successfully warped {image_path.name}. Warped dimensions: {source_for_resize_pil.size}")
        except Exception as e_conv:
            logger.error(f"[image_utils] Error converting warped CV image to PIL for {image_path.name}: {e_conv}. Falling back.")
            source_for_resize_pil = pil_img # Fallback to original PIL image (opened at the start)
    else:
        logger.warning(f"[image_utils] Perspective warp failed for {image_path.name}. Using entire image for resize/padding.")
        source_for_resize_pil = pil_img # Fallback to original PIL image (opened at the start)

    # --- 2. Resize (maintaining aspect ratio) ---
    # Calculate dimensions to fit source_for_resize_pil within target_width/target_height
    current_width, current_height = source_for_resize_pil.size
    new_width, new_height = _calculate_resize_dimensions(
        current_width, current_height, target_width, target_height
    )
    
    try:
        # Using LANCZOS (formerly ANTIALIAS) for high-quality downscaling
        resized_slide_content = source_for_resize_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
    except Exception as e:
        logger.error(f"[image_utils] Error resizing image {image_path.name}: {e}")
        return None

    # --- 3. Create new canvas ---
    try:
        canvas = Image.new("RGB", (target_width, target_height), canvas_bg_color)
    except Exception as e:
        logger.error(f"[image_utils] Error creating new image canvas for {image_path.name}: {e}")
        return None

    # --- 4. Paste resized slide onto center of canvas ---
    # Calculate position to paste (center)
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    
    try:
        canvas.paste(resized_slide_content, (paste_x, paste_y))
    except Exception as e:
        logger.error(f"[image_utils] Error pasting image {image_path.name} onto canvas: {e}")
        return None

    # --- 5. Save the resulting image ---
    try:
        output_filename = image_path.name # Retain original filename
        output_path = output_dir / output_filename
        canvas.save(output_path)
        logger.info(f"[image_utils] Successfully processed and saved {image_path.name} to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"[image_utils] Error saving processed image {output_path}: {e}")
        return None

if __name__ == '__main__':
    # Example Usage (for testing this module directly)
    # Create dummy directories and a dummy image for testing
    print("Running image_utils.py example...")

    # Setup basic logging for standalone test
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    project_root_test = Path(__file__).resolve().parent.parent.parent
    test_materials_dir = project_root_test / "test_materials"
    test_slides_dir = test_materials_dir / "test_slides"
    test_output_dir = project_root_test / "test_working_data" / "test_cropped_slides"

    test_slides_dir.mkdir(parents=True, exist_ok=True)
    test_output_dir.mkdir(parents=True, exist_ok=True)

    # Create a dummy image
    dummy_image_path = test_slides_dir / "dummy_slide.jpg"
    try:
        dummy_img = Image.new("RGB", (800, 600), "blue")
        dummy_img.save(dummy_image_path)
        print(f"Created dummy image: {dummy_image_path}")

        # Test the function
        TARGET_W, TARGET_H = 1920, 1080
        BG_COLOR = (30, 30, 30) # Dark grey

        result_path = prepare_slide_fullscreen(
            dummy_image_path,
            test_output_dir,
            TARGET_W,
            TARGET_H,
            BG_COLOR
        )

        if result_path:
            print(f"Test successful. Processed image saved to: {result_path}")
            # You can open the image to verify
            # Image.open(result_path).show()
        else:
            print("Test failed. See logs for errors.")

    except Exception as e:
        print(f"Error during test setup or execution: {e}")
    finally:
        # Clean up dummy files/dirs if needed, or leave for inspection
        # For a real test suite, you'd use unittest and proper teardown.
        pass
