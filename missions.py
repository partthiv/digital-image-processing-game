# Mission Mode - Educational Scenarios

MISSIONS = {
    "medical_xray": {
        "title": "Medical Imaging: Bone Fracture Detection",
        "description": "A doctor needs to see bone fractures clearly in this X-ray image.",
        "goal": "Enhance the edges to make fractures visible.",
        "hint": "Edge detection filters highlight boundaries and cracks.",
        "required_categories": ["Edge Detection"],
        "suggested_filters": ["Sobel Edge", "Prewitt Edge", "Laplacian"],
        "difficulty": 2,
        "context": "In medical imaging, edge detection helps identify fractures, tumors, and abnormalities."
    },
    
    "cctv_cleanup": {
        "title": "Security: CCTV Footage Enhancement",
        "description": "This CCTV footage is grainy and noisy. Clean it up for investigation.",
        "goal": "Remove noise while preserving important details.",
        "hint": "Smoothing filters reduce noise. Median filter is best for salt-and-pepper noise.",
        "required_categories": ["Smoothing"],
        "suggested_filters": ["Median Filter", "Gaussian Blur"],
        "difficulty": 2,
        "context": "Security footage often needs noise reduction to identify suspects or read license plates."
    },
    
    "satellite_contrast": {
        "title": "Satellite Imaging: Terrain Analysis",
        "description": "This satellite image has poor contrast. Enhance it for terrain mapping.",
        "goal": "Improve contrast to see terrain features clearly.",
        "hint": "Point operations adjust brightness and contrast.",
        "required_categories": ["Point Ops"],
        "suggested_filters": ["Contrast Stretch", "Gamma Correction"],
        "difficulty": 2,
        "context": "Satellite images often need contrast enhancement to identify geographical features."
    },
    
    "document_scan": {
        "title": "Document Processing: Text Extraction",
        "description": "This scanned document needs to be converted to pure black and white for OCR.",
        "goal": "Create a binary image with clear text.",
        "hint": "Threshold converts grayscale to binary.",
        "required_categories": ["Point Ops"],
        "suggested_filters": ["Threshold"],
        "difficulty": 2,
        "context": "Document scanning uses thresholding to separate text from background for OCR systems."
    },
    
    "photo_restoration": {
        "title": "Photo Restoration: Old Photo Enhancement",
        "description": "This old photo is dark and has low contrast. Restore it.",
        "goal": "Brighten the image and enhance details.",
        "hint": "Combine brightness adjustment with contrast enhancement.",
        "required_categories": ["Point Ops"],
        "suggested_filters": ["Log Transform", "Gamma Correction", "Contrast Stretch"],
        "difficulty": 3,
        "context": "Photo restoration uses multiple point operations to recover details from old photographs."
    },
    
    "fingerprint_analysis": {
        "title": "Forensics: Fingerprint Enhancement",
        "description": "Enhance this fingerprint for pattern matching.",
        "goal": "Make ridge patterns clear and distinct.",
        "hint": "First enhance contrast, then detect edges.",
        "required_categories": ["Point Ops", "Edge Detection"],
        "suggested_filters": ["Contrast Stretch", "Sobel Edge"],
        "difficulty": 3,
        "context": "Fingerprint analysis requires contrast enhancement followed by edge detection."
    },
    
    "astronomy_nebula": {
        "title": "Astronomy: Nebula Detail Enhancement",
        "description": "This telescope image of a nebula has faint details. Enhance them.",
        "goal": "Reveal hidden structures in the nebula.",
        "hint": "Log transform enhances dark regions. High pass filter reveals fine details.",
        "required_categories": ["Point Ops", "Frequency Domain"],
        "suggested_filters": ["Log Transform", "DFT High Pass"],
        "difficulty": 3,
        "context": "Astronomical images use log transform to reveal faint objects and frequency filters for detail."
    },
    
    "quality_control": {
        "title": "Manufacturing: Defect Detection",
        "description": "Inspect this product for surface defects.",
        "goal": "Highlight any scratches, dents, or imperfections.",
        "hint": "Smooth first to reduce texture noise, then detect edges to find defects.",
        "required_categories": ["Smoothing", "Edge Detection"],
        "suggested_filters": ["Gaussian Blur", "Laplacian"],
        "difficulty": 3,
        "context": "Quality control systems use smoothing to reduce texture, then edge detection to find defects."
    },
    
    "facial_recognition": {
        "title": "Security: Facial Recognition Preprocessing",
        "description": "Prepare this face image for recognition system.",
        "goal": "Normalize lighting and enhance features.",
        "hint": "Adjust brightness, enhance contrast, then smooth slightly.",
        "required_categories": ["Point Ops", "Smoothing"],
        "suggested_filters": ["Gamma Correction", "Contrast Stretch", "Gaussian Blur"],
        "difficulty": 4,
        "context": "Facial recognition systems preprocess images to normalize lighting and reduce noise."
    },
    
    "microscopy_cells": {
        "title": "Biology: Cell Boundary Detection",
        "description": "Identify cell boundaries in this microscopy image.",
        "goal": "Clearly separate individual cells.",
        "hint": "Enhance contrast, smooth noise, detect edges, then threshold.",
        "required_categories": ["Point Ops", "Smoothing", "Edge Detection"],
        "suggested_filters": ["Contrast Stretch", "Median Filter", "Sobel Edge", "Threshold"],
        "difficulty": 4,
        "context": "Microscopy analysis uses multiple steps to segment and count cells."
    }
}

def get_mission_by_difficulty(difficulty):
    """Get missions suitable for a given difficulty level"""
    return {k: v for k, v in MISSIONS.items() if v["difficulty"] == difficulty}

def get_all_missions():
    """Get all available missions"""
    return MISSIONS

def get_mission(mission_id):
    """Get a specific mission by ID"""
    return MISSIONS.get(mission_id)
