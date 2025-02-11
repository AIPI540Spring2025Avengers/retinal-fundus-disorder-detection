'''
script to train model and predict
'''
import cv2
import numpy as np

def classify_fundus_image(image):
    """Extracts features and performs rule-based classification on a retinal fundus image."""

    if image is None:
        return "Error - Unable to Load"

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    #Feature Extraction
    edges = cv2.Canny(blurred, 10, 80)
    edge_density = np.sum(edges) / edges.size

    _, bright_regions = cv2.threshold(blurred, 130, 255, cv2.THRESH_BINARY)
    bright_pixel_count = cv2.countNonZero(bright_regions)

    mean_intensity = np.mean(gray)

    _, optic_disc = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    optic_disc_size = np.sum(optic_disc) / gray.size

    vessel_skeleton = cv2.bitwise_and(edges, edges, mask=bright_regions)
    vessel_tortuosity = np.sum(vessel_skeleton) / bright_pixel_count if bright_pixel_count > 0 else 0

    #Rule-Based Classification
    if mean_intensity > 125 and bright_pixel_count > 5000 and edge_density < 25:
        return "Dry AMD"
    elif bright_pixel_count > 2500 and mean_intensity > 115 and edge_density < 30:
        return "Wet AMD"
    elif 2000 < bright_pixel_count < 10000 and edge_density > 4 and vessel_tortuosity > 0.25:
        return "Mild DR"
    elif 1500 < bright_pixel_count < 8000 and 6 < edge_density < 35:
        return "Moderate DR"
    elif 800 < bright_pixel_count < 6000 and edge_density > 10:
        return "Severe DR"
    elif edge_density > 12 and bright_pixel_count < 4000 and vessel_tortuosity > 0.3:
        return "Proliferate DR"
    elif mean_intensity < 105 and edge_density < 6:
        return "Cataract"
    elif optic_disc_size > 0.018 and mean_intensity < 120:
        return "Glaucoma"
    elif bright_pixel_count < 2200 and edge_density > 8:
        return "Pathological Myopia"
    elif edge_density < 18 and mean_intensity > 125:
        return "Normal Fundus"
    else:
        return "Uncertain - Further Review Needed"