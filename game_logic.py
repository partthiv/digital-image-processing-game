import numpy as np
import cv2
import random

class ImageProcessor:
    @staticmethod
    def to_gray(img):
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    @staticmethod
    def to_rgb(img):
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    @staticmethod
    def normalize(img):
        img = img.astype(float)
        img -= img.min()
        if img.max() != 0:
            img /= img.max()
        return (img * 255).astype(np.uint8)

    # --- Filters ---
    @staticmethod
    def negative(img, params=None):
        return 255 - img

    @staticmethod
    def log_transform(img, params=None):
        c = params.get('c', 1.0) if params else 1.0
        img_f = img.astype(float)
        return ImageProcessor.normalize(c * np.log1p(img_f))

    @staticmethod
    def gamma_correction(img, params=None):
        gamma = params.get('gamma', 0.6) if params else 0.6
        c = params.get('c', 1.0) if params else 1.0
        img_f = img.astype(float) / 255.0
        return ImageProcessor.normalize(c * (img_f ** gamma))

    @staticmethod
    def contrast_stretch(img, params=None):
        low_p = params.get('low_percentile', 2) if params else 2
        high_p = params.get('high_percentile', 98) if params else 98
        out = np.zeros_like(img)
        channels = 1 if len(img.shape) == 2 else 3
        
        if channels == 1:
            lo = np.percentile(img, low_p)
            hi = np.percentile(img, high_p)
            out = np.clip((img - lo) * 255 / (hi - lo + 1e-5), 0, 255)
        else:
            for c in range(3):
                lo = np.percentile(img[:,:,c], low_p)
                hi = np.percentile(img[:,:,c], high_p)
                out[:,:,c] = np.clip((img[:,:,c]-lo)*255/(hi-lo+1e-5), 0, 255)
        return out.astype(np.uint8)

    @staticmethod
    def threshold(img, params=None):
        thresh_value = params.get('threshold', 127) if params else 127
        g = ImageProcessor.to_gray(img)
        _, th = cv2.threshold(g, thresh_value, 255, cv2.THRESH_BINARY)
        return ImageProcessor.to_rgb(th)

    @staticmethod
    def averaging_blur(img, params=None):
        k = int(params.get('kernel_size', 5)) if params else 5
        # Ensure kernel size is odd and at least 1
        if k % 2 == 0: k += 1
        if k < 1: k = 1
        return cv2.blur(img, (k, k))

    @staticmethod
    def gaussian_blur(img, params=None):
        k = int(params.get('kernel_size', 5)) if params else 5
        sigma = params.get('sigma', 1.0) if params else 1.0
        # Ensure kernel size is odd and at least 1
        if k % 2 == 0: k += 1
        if k < 1: k = 1
        return cv2.GaussianBlur(img, (k, k), sigma)

    @staticmethod
    def median_filter(img, params=None):
        k = int(params.get('kernel_size', 5)) if params else 5
        # Ensure kernel size is odd and at least 1
        if k % 2 == 0: k += 1
        if k < 1: k = 1
        return cv2.medianBlur(img, k)

    @staticmethod
    def sobel(img, params=None):
        g = ImageProcessor.to_gray(img)
        dx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
        return ImageProcessor.to_rgb(ImageProcessor.normalize(np.sqrt(dx*dx + dy*dy)))

    @staticmethod
    def prewitt(img, params=None):
        g = ImageProcessor.to_gray(img)
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        dx = cv2.filter2D(g, cv2.CV_64F, kernel_x)
        dy = cv2.filter2D(g, cv2.CV_64F, kernel_y)
        return ImageProcessor.to_rgb(ImageProcessor.normalize(np.sqrt(dx*dx + dy*dy)))

    @staticmethod
    def laplacian(img, params=None):
        g = ImageProcessor.to_gray(img)
        lap = cv2.Laplacian(g, cv2.CV_64F)
        return ImageProcessor.to_rgb(ImageProcessor.normalize(np.abs(lap)))

    @staticmethod
    def dft_low_pass(img, params=None):
        cutoff = params.get('cutoff', 30) if params else 30
        g = ImageProcessor.to_gray(img)
        dft = cv2.dft(np.float32(g), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
        rows, cols = g.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create low pass filter mask
        mask = np.zeros((rows, cols, 2), np.uint8)
        r = int(cutoff)
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
        mask[mask_area] = 1
        
        # Apply mask
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        
        return ImageProcessor.to_rgb(ImageProcessor.normalize(img_back))

    @staticmethod
    def dft_high_pass(img, params=None):
        cutoff = params.get('cutoff', 30) if params else 30
        g = ImageProcessor.to_gray(img)
        dft = cv2.dft(np.float32(g), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
        rows, cols = g.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create high pass filter mask
        mask = np.ones((rows, cols, 2), np.uint8)
        r = int(cutoff)
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
        mask[mask_area] = 0
        
        # Apply mask
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        
        return ImageProcessor.to_rgb(ImageProcessor.normalize(img_back))

# Metadata Definition
FILTER_METADATA = {
    "Negative": {
        "func": ImageProcessor.negative, 
        "category": "Point Ops", 
        "params": {},
        "formula": "output = 255 - input"
    },
    "Log Transform": {
        "func": ImageProcessor.log_transform, 
        "category": "Point Ops", 
        "params": {"c": {"min": 0.1, "max": 3.0, "default": 1.0, "step": 0.1}},
        "formula": "output = c × log(1 + input)"
    },
    "Gamma Correction": {
        "func": ImageProcessor.gamma_correction, 
        "category": "Point Ops",
        "params": {"gamma": {"min": 0.1, "max": 3.0, "default": 0.6, "step": 0.1},
                   "c": {"min": 0.1, "max": 3.0, "default": 1.0, "step": 0.1}},
        "formula": "output = c × (input/255)^γ × 255"
    },
    "Contrast Stretch": {
        "func": ImageProcessor.contrast_stretch, 
        "category": "Point Ops",
        "params": {"low_percentile": {"min": 0, "max": 10, "default": 2, "step": 1},
                   "high_percentile": {"min": 90, "max": 100, "default": 98, "step": 1}},
        "formula": "output = (input - min) × 255 / (max - min)"
    },
    "Threshold": {
        "func": ImageProcessor.threshold, 
        "category": "Point Ops",
        "params": {"threshold": {"min": 0, "max": 255, "default": 127, "step": 1}},
        "formula": "output = 255 if input > T else 0"
    },
    "Averaging Blur": {
        "func": ImageProcessor.averaging_blur, 
        "category": "Smoothing",
        "params": {"kernel_size": {"min": 3, "max": 15, "default": 5, "step": 2}},
        "formula": "output = mean(neighborhood)"
    },
    "Gaussian Blur": {
        "func": ImageProcessor.gaussian_blur, 
        "category": "Smoothing",
        "params": {"kernel_size": {"min": 3, "max": 15, "default": 5, "step": 2},
                   "sigma": {"min": 0.1, "max": 5.0, "default": 1.0, "step": 0.1}},
        "formula": "G(x,y) = (1/2πσ²)e^(-(x²+y²)/2σ²)"
    },
    "Median Filter": {
        "func": ImageProcessor.median_filter, 
        "category": "Smoothing",
        "params": {"kernel_size": {"min": 3, "max": 15, "default": 5, "step": 2}},
        "formula": "output = median(neighborhood)"
    },
    "Sobel Edge": {
        "func": ImageProcessor.sobel, 
        "category": "Edge Detection", 
        "params": {},
        "formula": "G = √(Gx² + Gy²)"
    },
    "Prewitt Edge": {
        "func": ImageProcessor.prewitt, 
        "category": "Edge Detection", 
        "params": {},
        "formula": "G = √(Gx² + Gy²) with Prewitt kernels"
    },
    "Laplacian": {
        "func": ImageProcessor.laplacian, 
        "category": "Edge Detection", 
        "params": {},
        "formula": "∇²f = ∂²f/∂x² + ∂²f/∂y²"
    },
    "DFT Low Pass": {
        "func": ImageProcessor.dft_low_pass, 
        "category": "Frequency Domain",
        "params": {"cutoff": {"min": 10, "max": 100, "default": 30, "step": 5}},
        "formula": "F(u,v) × H(u,v) where H=1 if D≤D₀"
    },
    "DFT High Pass": {
        "func": ImageProcessor.dft_high_pass, 
        "category": "Frequency Domain",
        "params": {"cutoff": {"min": 10, "max": 100, "default": 30, "step": 5}},
        "formula": "F(u,v) × H(u,v) where H=0 if D≤D₀"
    }
}

class GameEngine:
    def __init__(self):
        self.original_image = None
        self.pipeline = []
        self.pipeline_params = []
        self.intermediates = []
        self.current_step = 0
        self.difficulty = 3
        
        # Filter compatibility rules
        self.incompatible_sequences = [
            # Don't put threshold before edge detection (creates black images)
            ("Threshold", "Sobel Edge"),
            ("Threshold", "Prewitt Edge"),
            ("Threshold", "Laplacian"),
            # Don't put threshold before frequency domain
            ("Threshold", "DFT Low Pass"),
            ("Threshold", "DFT High Pass"),
            # Don't put edge detection before threshold (meaningless)
            ("Sobel Edge", "Threshold"),
            ("Prewitt Edge", "Threshold"),
            ("Laplacian", "Threshold"),
        ]
        
        # Categories that shouldn't repeat back-to-back
        self.no_repeat_categories = ["Point Ops"]
    
    def start_game(self, image_path, difficulty, max_retries=10):
        try:
            self.difficulty = difficulty
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image from {image_path}")
            self.original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Try to generate a valid pipeline with retries
            for retry in range(max_retries):
                # Generate Pipeline with constraints
                self.pipeline = self._generate_valid_pipeline(difficulty)
                self.pipeline_params = []
                self.intermediates = []
                
                temp = self.original_image.copy()
                pipeline_valid = True
                
                for filter_name in self.pipeline:
                    meta = FILTER_METADATA[filter_name]
                    params = {}
                    for pname, pinfo in meta["params"].items():
                        params[pname] = pinfo["default"]
                    
                    self.pipeline_params.append(params)
                    temp = meta["func"](temp, params)
                    # Ensure temp is RGB
                    if len(temp.shape) == 2:
                        temp = cv2.cvtColor(temp, cv2.COLOR_GRAY2RGB)
                    
                    # Validate output isn't completely black/white
                    std_dev = np.std(temp)
                    if std_dev < 1.0:  # Nearly uniform image
                        print(f"Warning: {filter_name} produced near-uniform image (std={std_dev:.2f}), retry {retry+1}/{max_retries}")
                        pipeline_valid = False
                        break
                    
                    self.intermediates.append(temp)
                
                # If pipeline is valid, we're done
                if pipeline_valid:
                    self.current_step = 0
                    return
            
            # If we exhausted retries, raise an error
            raise ValueError(f"Could not generate a valid pipeline after {max_retries} attempts. Try a different image or lower difficulty.")
                
        except Exception as e:
            print(f"Error in start_game: {e}")
            raise
    
    def _generate_valid_pipeline(self, difficulty, max_attempts=50):
        """Generate a valid pipeline that avoids problematic filter combinations"""
        all_filters = list(FILTER_METADATA.keys())
        
        for attempt in range(max_attempts):
            pipeline = random.sample(all_filters, min(difficulty, len(all_filters)))
            
            # Check for incompatible sequences
            valid = True
            for i in range(len(pipeline) - 1):
                current_filter = pipeline[i]
                next_filter = pipeline[i + 1]
                
                # Check incompatible pairs
                if (current_filter, next_filter) in self.incompatible_sequences:
                    valid = False
                    break
                
                # Check no-repeat categories
                current_category = FILTER_METADATA[current_filter]["category"]
                next_category = FILTER_METADATA[next_filter]["category"]
                
                if current_category in self.no_repeat_categories and current_category == next_category:
                    valid = False
                    break
            
            if valid:
                return pipeline
        
        # Fallback: return a safe default pipeline
        print("Warning: Could not generate valid pipeline, using safe defaults")
        safe_filters = ["Gaussian Blur", "Sobel Edge", "Contrast Stretch"]
        return safe_filters[:difficulty]

    def get_current_input(self):
        try:
            if self.current_step == 0:
                return self.original_image.copy() if self.original_image is not None else None
            if self.current_step - 1 < len(self.intermediates):
                return self.intermediates[self.current_step - 1].copy()
            return None
        except Exception as e:
            print(f"Error in get_current_input: {e}")
            return None

    def get_target_for_step(self):
        try:
            if self.current_step < len(self.intermediates):
                return self.intermediates[self.current_step].copy()
            return None
        except Exception as e:
            print(f"Error in get_target_for_step: {e}")
            return None

    def get_final_target(self):
        """Get the final target image (last intermediate) - this never changes during gameplay"""
        try:
            if len(self.intermediates) > 0:
                return self.intermediates[-1].copy()
            return None
        except Exception as e:
            print(f"Error in get_final_target: {e}")
            return None

    def check_guess(self, filter_name, result_image):
        try:
            expected_filter = self.pipeline[self.current_step]
            expected_image = self.intermediates[self.current_step]
            
            is_name_correct = (filter_name == expected_filter)
            
            # Ensure both images are in the same format
            if len(result_image.shape) == 2:
                result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2RGB)
            if len(expected_image.shape) == 2:
                expected_image = cv2.cvtColor(expected_image, cv2.COLOR_GRAY2RGB)
            
            # Calculate SSIM for better similarity check
            from skimage.metrics import structural_similarity as ssim
            
            # Convert to grayscale for SSIM
            result_gray = cv2.cvtColor(result_image, cv2.COLOR_RGB2GRAY)
            expected_gray = cv2.cvtColor(expected_image, cv2.COLOR_RGB2GRAY)
            
            similarity = ssim(result_gray, expected_gray)
            
            # Also calculate MSE as fallback
            err = np.sum((result_image.astype("float") - expected_image.astype("float")) ** 2)
            err /= float(result_image.shape[0] * result_image.shape[1])
            
            # Image is correct if SSIM > 0.95 OR MSE < 50
            is_image_correct = (similarity > 0.95) or (err < 50)
            
            # Creative solution bonus: If image matches but filter name doesn't
            is_creative_solution = (not is_name_correct) and (similarity > 0.98)
            
            return is_name_correct, is_image_correct, is_creative_solution, similarity
        except Exception as e:
            print(f"Error in check_guess: {e}")
            # Fallback to MSE only
            try:
                err = np.sum((result_image.astype("float") - expected_image.astype("float")) ** 2)
                err /= float(result_image.shape[0] * result_image.shape[1])
                is_image_correct = err < 50
                return is_name_correct, is_image_correct, False, 0.0
            except:
                return False, False, False, 0.0
    
    def get_parameter_distance(self, filter_name, user_params):
        """Calculate how close user parameters are to expected (0.0 = perfect, 1.0 = far)"""
        try:
            if self.current_step >= len(self.pipeline):
                return 1.0
            
            expected_filter = self.pipeline[self.current_step]
            if filter_name != expected_filter:
                return 1.0
            
            expected_params = self.pipeline_params[self.current_step]
            if not expected_params:
                return 0.0  # No parameters to match
            
            # Calculate normalized distance for each parameter
            meta = FILTER_METADATA[filter_name]
            distances = []
            
            for param_name, expected_value in expected_params.items():
                if param_name not in user_params:
                    distances.append(1.0)
                    continue
                
                user_value = user_params[param_name]
                param_info = meta["params"][param_name]
                
                # Normalize to 0-1 range
                param_range = param_info["max"] - param_info["min"]
                if param_range == 0:
                    distances.append(0.0)
                    continue
                
                normalized_distance = abs(user_value - expected_value) / param_range
                distances.append(min(normalized_distance, 1.0))
            
            # Return average distance
            return sum(distances) / len(distances) if distances else 0.0
            
        except Exception as e:
            print(f"Error in get_parameter_distance: {e}")
            return 1.0