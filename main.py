import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                             QComboBox, QSlider, QFrame, QMessageBox, QScrollArea, 
                             QSizePolicy, QCheckBox, QTextEdit, QDialog, QDialogButtonBox,
                             QGridLayout)
from PyQt6.QtCore import Qt, pyqtSignal as Signal, QTimer, QPropertyAnimation, QEasingCurve, QRect, QSize, QThread
from PyQt6.QtGui import QPixmap, QImage, QIcon

from game_logic import GameEngine, FILTER_METADATA
import styling
from missions import MISSIONS, get_mission_by_difficulty

class FilterWorker(QThread):
    """Background thread for filter processing"""
    finished = Signal(object)  # Emits the processed image
    
    def __init__(self, filter_func, image, params):
        super().__init__()
        self.filter_func = filter_func
        self.image = image.copy()
        self.params = params
    
    def run(self):
        try:
            result = self.filter_func(self.image, self.params)
            self.finished.emit(result)
        except Exception as e:
            print(f"Error in FilterWorker: {e}")
            self.finished.emit(None)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DIP Guessing Game")
        self.resize(1200, 800)
        self.game = GameEngine()
        self.current_preview = None
        self.last_image_path = None  # Store the last loaded image path
        
        # State for dynamic sliders
        self.slider_controls = {} 
        self.current_params = {}
        
        # New feature toggles
        self.advanced_mode = False
        self.category_hints = False
        
        # Threading for smooth UI
        self.filter_worker = None
        self.pending_preview_update = False
        
        # Mission mode
        self.mission_mode = False
        self.current_mission = None

        self.setup_ui()
        self.apply_styles()

    def apply_styles(self):
        self.setStyleSheet(styling.STYLESHEET)

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # --- Header ---
        header = QHBoxLayout()
        title = QLabel("🎮 DIP Guessing Game")
        title.setObjectName("HeaderTitle")
        header.addWidget(title)
        
        header.addStretch()
        
        self.btn_load = QPushButton("📂 Load Image")
        self.btn_load.clicked.connect(self.load_image)
        header.addWidget(self.btn_load)
        
        self.combo_difficulty = QComboBox()
        self.combo_difficulty.addItems(["2 Steps", "3 Steps", "4 Steps", "5 Steps", "6 Steps"])
        self.combo_difficulty.setCurrentIndex(1) # Default 3
        header.addWidget(self.combo_difficulty)
        
        self.btn_help = QPushButton("❓ Help")
        self.btn_help.setObjectName("SecondaryButton")
        self.btn_help.clicked.connect(self.show_help)
        header.addWidget(self.btn_help)
        
        self.btn_mission = QPushButton("🎯 Mission Mode")
        self.btn_mission.setObjectName("SecondaryButton")
        self.btn_mission.clicked.connect(self.start_mission_mode)
        header.addWidget(self.btn_mission)

        self.btn_reset = QPushButton("🔄 New Game")
        self.btn_reset.clicked.connect(self.reset_game)
        self.btn_reset.setEnabled(False)
        header.addWidget(self.btn_reset)
        
        main_layout.addLayout(header)

        # --- Settings Bar ---
        settings_layout = QHBoxLayout()
        
        self.chk_advanced = QCheckBox("🔬 Advanced Mode")
        self.chk_advanced.setStyleSheet("color: #00d4ff; font-weight: bold; font-size: 14px;")
        self.chk_advanced.stateChanged.connect(self.toggle_advanced_mode)
        settings_layout.addWidget(self.chk_advanced)
        
        self.chk_hints = QCheckBox("💡 Category Hints")
        self.chk_hints.setStyleSheet("color: #00d4ff; font-weight: bold; font-size: 14px;")
        self.chk_hints.stateChanged.connect(self.toggle_category_hints)
        settings_layout.addWidget(self.chk_hints)
        
        settings_layout.addStretch()
        main_layout.addLayout(settings_layout)
        
        # --- Game Status Bar ---
        self.lbl_status = QLabel("Please load an image to start.")
        self.lbl_status.setStyleSheet("font-size: 18px; color: #00d4ff; font-weight: bold; padding: 10px; background-color: #0f3460; border-radius: 8px; border: 2px solid #00d4ff;")
        main_layout.addWidget(self.lbl_status)

        # --- Image Area ---
        image_layout = QHBoxLayout()
        
        self.frame_orig = self.create_image_card("Current Input", "Start")
        self.lbl_orig = self.frame_orig.findChild(QLabel, "ImgLabel")
        
        self.frame_prev = self.create_image_card("Your Preview", "Current Attempt")
        self.lbl_prev = self.frame_prev.findChild(QLabel, "ImgLabel")
        
        self.frame_target = self.create_image_card("Final Target", "Goal (Never Changes)")
        self.lbl_target = self.frame_target.findChild(QLabel, "ImgLabel")

        image_layout.addWidget(self.frame_orig)
        image_layout.addWidget(self.frame_prev)
        image_layout.addWidget(self.frame_target)
        
        main_layout.addLayout(image_layout, stretch=1)

        # --- Controls Area ---
        controls_frame = QFrame()
        controls_frame.setObjectName("ControlPanel")
        controls_layout = QHBoxLayout(controls_frame)

        # Left: Filter Selection
        selection_layout = QVBoxLayout()
        cat_label = QLabel("Select Category:")
        cat_label.setStyleSheet("color: #00d4ff; font-weight: bold; font-size: 14px;")
        selection_layout.addWidget(cat_label)
        self.combo_cat = QComboBox()
        self.combo_cat.currentTextChanged.connect(self.populate_filters)
        selection_layout.addWidget(self.combo_cat)
        
        filter_label = QLabel("Select Filter:")
        filter_label.setStyleSheet("color: #00d4ff; font-weight: bold; font-size: 14px;")
        selection_layout.addWidget(filter_label)
        self.combo_filter = QComboBox()
        self.combo_filter.currentTextChanged.connect(self.build_parameters_ui)
        selection_layout.addWidget(self.combo_filter)
        selection_layout.addStretch()
        
        controls_layout.addLayout(selection_layout, stretch=1)

        # Middle: Parameters (Dynamic)
        params_scroll = QScrollArea()
        params_scroll.setWidgetResizable(True)
        params_scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        
        self.params_container = QWidget()
        self.params_layout = QVBoxLayout(self.params_container)
        
        # Formula display
        self.lbl_formula = QLabel()
        self.lbl_formula.setObjectName("FormulaLabel")
        self.lbl_formula.setWordWrap(True)
        self.lbl_formula.setVisible(False)
        self.params_layout.addWidget(self.lbl_formula)
        
        params_scroll.setWidget(self.params_container)
        controls_layout.addWidget(params_scroll, stretch=2)

        # Right: Actions
        action_layout = QVBoxLayout()
        self.btn_apply = QPushButton("✅ Apply Filter")
        self.btn_apply.setFixedHeight(40)
        self.btn_apply.clicked.connect(self.check_guess)
        self.btn_apply.setEnabled(False)
        
        self.btn_hint = QPushButton("💡 Hint / Skip")
        self.btn_hint.setObjectName("SecondaryButton")
        self.btn_hint.clicked.connect(self.show_hint)
        self.btn_hint.setEnabled(False)
        
        self.btn_solution = QPushButton("📋 View Solution")
        self.btn_solution.setObjectName("SecondaryButton")
        self.btn_solution.clicked.connect(self.show_solution)
        self.btn_solution.setEnabled(False)
        
        self.btn_report = QPushButton("📊 Full Report")
        self.btn_report.setObjectName("SecondaryButton")
        self.btn_report.clicked.connect(self.show_full_report)
        self.btn_report.setEnabled(False)

        action_layout.addWidget(self.btn_apply)
        action_layout.addWidget(self.btn_hint)
        action_layout.addWidget(self.btn_solution)
        action_layout.addWidget(self.btn_report)
        action_layout.addStretch()
        
        controls_layout.addLayout(action_layout, stretch=1)
        main_layout.addWidget(controls_frame)

        # Initialize Data
        self.populate_categories()

    def create_image_card(self, title_text, subtitle_text):
        frame = QFrame()
        frame.setObjectName("Card")
        layout = QVBoxLayout(frame)
        
        title = QLabel(title_text)
        title.setObjectName("CardTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        lbl_img = QLabel()
        lbl_img.setObjectName("ImgLabel")
        lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_img.setText(subtitle_text)
        lbl_img.setScaledContents(True)
        lbl_img.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        
        layout.addWidget(title)
        layout.addWidget(lbl_img, stretch=1)
        return frame

    # --- Logic ---

    def populate_categories(self):
        categories = sorted(list(set(m["category"] for m in FILTER_METADATA.values())))
        self.combo_cat.addItems(categories)
        self.populate_filters(categories[0])

    def populate_filters(self, category):
        self.combo_filter.blockSignals(True)
        self.combo_filter.clear()
        filters = [k for k, v in FILTER_METADATA.items() if v["category"] == category]
        if filters:
            self.combo_filter.addItems(filters)
            self.combo_filter.blockSignals(False)
            self.build_parameters_ui(filters[0])
        else:
            self.combo_filter.blockSignals(False)

    def build_parameters_ui(self, filter_name):
        # Clear previous params - properly delete widgets and layouts
        # Keep the formula label
        formula_widget = self.lbl_formula
        self.params_layout.removeWidget(formula_widget)
        
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self.clear_layout(item.layout())
        
        # Re-add formula label at top
        self.params_layout.addWidget(formula_widget)
        
        self.slider_controls = {}
        self.current_params = {}
        
        if not filter_name or filter_name not in FILTER_METADATA: 
            return

        meta = FILTER_METADATA[filter_name]
        
        # Update formula display
        if self.advanced_mode and "formula" in meta:
            self.lbl_formula.setText(f"📐 Formula: {meta['formula']}")
            self.lbl_formula.setVisible(True)
        else:
            self.lbl_formula.setVisible(False)
        
        if not meta["params"]:
            lbl = QLabel("No parameters for this filter.")
            lbl.setStyleSheet("color: #00d4ff; font-size: 14px; font-style: italic;")
            self.params_layout.addWidget(lbl)
            self.current_params = {}
        else:
            for p_name, p_info in meta["params"].items():
                row = QHBoxLayout()
                lbl = QLabel(f"{p_name}:")
                lbl.setStyleSheet("color: #00d4ff; font-weight: bold; font-size: 13px;")
                val_lbl = QLabel(str(p_info["default"]))
                val_lbl.setStyleSheet("color: #00ffff; font-weight: bold; font-size: 13px; background-color: #1a1a2e; padding: 5px; border-radius: 4px; border: 1px solid #00d4ff;")
                val_lbl.setFixedWidth(50)
                
                slider = QSlider(Qt.Orientation.Horizontal)
                
                # Handling Float vs Int sliders logic
                is_float = isinstance(p_info["step"], float)
                factor = 100 if is_float else 1
                
                slider.setMinimum(int(p_info["min"] * factor))
                slider.setMaximum(int(p_info["max"] * factor))
                slider.setValue(int(p_info["default"] * factor))
                slider.setSingleStep(int(p_info["step"] * factor))
                
                # Block signals temporarily to avoid triggering updates during setup
                slider.blockSignals(True)
                
                # Connect signal
                # Use default arguments to capture loop variable
                slider.valueChanged.connect(lambda v, l=val_lbl, n=p_name, f=factor: 
                                            self.on_param_change(v, l, n, f))

                self.slider_controls[p_name] = slider
                self.current_params[p_name] = p_info["default"]
                
                # Only show sliders in advanced mode
                lbl.setVisible(self.advanced_mode)
                slider.setVisible(self.advanced_mode)
                val_lbl.setVisible(self.advanced_mode)
                
                row.addWidget(lbl)
                row.addWidget(slider)
                row.addWidget(val_lbl)
                self.params_layout.addLayout(row)
                
                # Unblock signals after setup
                slider.blockSignals(False)
        
        # Only update preview if we have an image loaded
        if self.game.original_image is not None:
            self.update_preview()
    
    def clear_layout(self, layout):
        """Helper method to properly clear a layout"""
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self.clear_layout(item.layout())

    def on_param_change(self, value, label_widget, param_name, factor):
        try:
            real_value = value / factor
            if factor > 1:
                # Float parameter
                label_widget.setText(f"{real_value:.1f}")
                self.current_params[param_name] = real_value
            else:
                # Integer parameter
                int_value = int(real_value)
                label_widget.setText(str(int_value))
                self.current_params[param_name] = int_value
            
            # Only update preview if we have an image loaded
            if self.game.original_image is not None:
                self.update_preview()
        except Exception as e:
            print(f"Error in on_param_change: {e}")

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            try:
                diff = int(self.combo_difficulty.currentText().split()[0])
                self.game.start_game(path, diff)
                self.last_image_path = path  # Store the path for reset
                self.update_game_state_ui()
                self.btn_reset.setEnabled(True)
                self.btn_apply.setEnabled(True)
                self.btn_hint.setEnabled(True)
                self.btn_solution.setEnabled(True)
                self.btn_report.setEnabled(True)
            except Exception as e:
                QMessageBox.critical(self, "Error Loading Image", f"Could not load image:\n{str(e)}")

    def reset_game(self):
        """Reset the game with a new random pipeline using the same image"""
        if self.last_image_path and self.game.original_image is not None:
            try:
                diff = int(self.combo_difficulty.currentText().split()[0])
                
                # Check if the original file still exists
                import os
                if os.path.exists(self.last_image_path):
                    # Start new game with same image from original path
                    self.game.start_game(self.last_image_path, diff)
                else:
                    # If original file doesn't exist, save current image temporarily
                    temp_path = "temp_game_image.png"
                    current_image = self.game.original_image.copy()
                    cv2.imwrite(temp_path, cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR))
                    self.game.start_game(temp_path, diff)
                    self.last_image_path = temp_path
                
                self.update_game_state_ui()
                QMessageBox.information(self, "New Game", "New puzzle generated with the same image!")
            except Exception as e:
                print(f"Error in reset_game: {e}")
                import traceback
                traceback.print_exc()
                QMessageBox.warning(self, "Reset Failed", "Please load a new image to start a new game.")
        else:
            QMessageBox.information(self, "No Image", "Please load an image first!")

    def update_game_state_ui(self):
        status_text = f"Progress: Step {self.game.current_step + 1} / {self.game.difficulty}"
        
        # Add category hint if enabled
        if self.category_hints and self.game.current_step < len(self.game.pipeline):
            expected_filter = self.game.pipeline[self.game.current_step]
            category = FILTER_METADATA[expected_filter]["category"]
            status_text += f" | 💡 Hint: Category is '{category}'"
        
        self.lbl_status.setText(status_text)
        
        # Update displays
        # Original shows the starting point (changes as you progress)
        input_img = self.game.get_current_input()
        # Target ALWAYS shows the final result (never changes)
        target_img = self.game.get_final_target()
        
        self.set_pixmap(self.lbl_orig, input_img)
        self.set_pixmap(self.lbl_target, target_img)
        self.update_preview()

    def update_preview(self):
        if self.game.original_image is None: 
            return
        
        filter_name = self.combo_filter.currentText()
        if not filter_name or filter_name not in FILTER_METADATA: 
            return

        try:
            # Apply filter on current working image
            input_img = self.game.get_current_input()
            if input_img is None:
                return
                
            func = FILTER_METADATA[filter_name]["func"]
            
            # Make a copy to avoid modifying the original
            input_copy = input_img.copy()
            
            # If worker is busy, mark for update later
            if self.filter_worker and self.filter_worker.isRunning():
                self.pending_preview_update = True
                return
            
            # Start background processing
            self.filter_worker = FilterWorker(func, input_copy, self.current_params if self.current_params else {})
            self.filter_worker.finished.connect(self.on_preview_ready)
            self.filter_worker.start()
            
        except Exception as e:
            print(f"Error in update_preview: {e}")
            import traceback
            traceback.print_exc()
    
    def on_preview_ready(self, result):
        """Called when background filter processing completes"""
        try:
            if result is not None:
                self.current_preview = result
                self.set_pixmap(self.lbl_prev, self.current_preview)
                
                # Update hot/cold indicator if in advanced mode
                if self.advanced_mode:
                    self.update_parameter_feedback()
            
            # If another update was requested while processing, do it now
            if self.pending_preview_update:
                self.pending_preview_update = False
                QTimer.singleShot(100, self.update_preview)
                
        except Exception as e:
            print(f"Error in on_preview_ready: {e}")
    
    def update_parameter_feedback(self):
        """Update visual feedback for parameter accuracy (hot/cold indicator)"""
        try:
            filter_name = self.combo_filter.currentText()
            if not filter_name or filter_name not in FILTER_METADATA:
                return
            
            # Get parameter distance (0.0 = perfect, 1.0 = far)
            distance = self.game.get_parameter_distance(filter_name, self.current_params)
            
            # Determine color based on distance
            if distance < 0.1:
                color = "#00ff00"  # Green - Hot (very close)
                feedback = "🔥 HOT! Very close!"
            elif distance < 0.3:
                color = "#ffff00"  # Yellow - Warm (getting close)
                feedback = "☀️ Warm! Getting closer..."
            else:
                color = "#00d4ff"  # Blue - Cold (far off)
                feedback = "❄️ Cold. Keep trying..."
            
            # Update preview frame border
            self.frame_prev.setStyleSheet(f"""
                QFrame#Card {{
                    background-color: #0f3460;
                    border-radius: 12px;
                    border: 3px solid {color};
                    padding: 10px;
                }}
            """)
            
            # Update status if not showing category hint
            if not self.category_hints:
                current_status = self.lbl_status.text()
                if "Progress:" in current_status:
                    base_status = current_status.split("|")[0].strip()
                    self.lbl_status.setText(f"{base_status} | {feedback}")
                    
        except Exception as e:
            print(f"Error in update_parameter_feedback: {e}")

    def set_pixmap(self, label, cv_img):
        try:
            # Ensure image is in correct format
            if cv_img is None:
                return
            
            # Make a copy to ensure data persists
            cv_img = cv_img.copy()
            
            # Handle grayscale images
            if len(cv_img.shape) == 2:
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
            
            h, w, ch = cv_img.shape
            
            # Ensure image is uint8
            if cv_img.dtype != np.uint8:
                cv_img = cv_img.astype(np.uint8)
            
            bytes_per_line = ch * w
            
            # Make sure data is contiguous
            cv_img = np.ascontiguousarray(cv_img)
            
            # Convert to QImage - copy the data to avoid memory issues
            q_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
            pixmap = QPixmap.fromImage(q_img)
            
            # Scale to fit label while keeping aspect ratio
            if label.width() > 0 and label.height() > 0:
                scaled_pixmap = pixmap.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                label.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"Error in set_pixmap: {e}")
            import traceback
            traceback.print_exc()

    def resizeEvent(self, event):
        # Don't update on every resize to avoid crashes
        # The images will scale automatically with the labels
        super().resizeEvent(event)

    def check_guess(self):
        if self.current_preview is None:
            QMessageBox.warning(self, "No Preview", "Please wait for the preview to load.")
            return
            
        filter_name = self.combo_filter.currentText()
        try:
            is_name, is_img, is_creative, similarity = self.game.check_guess(filter_name, self.current_preview)
            
            if is_creative:
                # Creative solution bonus!
                msg = QMessageBox(self)
                msg.setIcon(QMessageBox.Icon.Information)
                msg.setWindowTitle("Creative Solution!")
                msg.setText(f"🌟 Creative Solution Bonus!\n\nYou found a different way to achieve the target!\n\nYour filter: {filter_name}\nExpected: {self.game.pipeline[self.game.current_step]}\nSimilarity: {similarity:.2%}\n\nThis counts as correct!")
                msg.exec()
                
                self.game.current_step += 1
                if self.game.current_step >= self.game.difficulty:
                    self.show_victory()
                else:
                    self.update_game_state_ui()
                    
            elif is_name and is_img:
                QMessageBox.information(self, "Correct!", f"✅ Step {self.game.current_step + 1} Complete!\nSimilarity: {similarity:.2%}")
                self.game.current_step += 1
                
                if self.game.current_step >= self.game.difficulty:
                    self.show_victory()
                else:
                    self.update_game_state_ui()
            elif is_name:
                distance = self.game.get_parameter_distance(filter_name, self.current_params)
                QMessageBox.warning(self, "Close!", f"Correct Filter, but parameters are off!\n\nParameter accuracy: {(1-distance)*100:.1f}%\nAdjust sliders and try again!")
            else:
                QMessageBox.critical(self, "Wrong!", f"Incorrect Filter.\n\nSimilarity: {similarity:.2%}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def show_hint(self):
        expected = self.game.pipeline[self.game.current_step]
        expected_params = self.game.pipeline_params[self.game.current_step]
        
        hint_text = f"The answer is: {expected}\n"
        if expected_params:
            hint_text += "Parameters:\n"
            for k, v in expected_params.items():
                hint_text += f"  {k}: {v}\n"
        hint_text += "\n(Skipping step)"
        
        QMessageBox.information(self, "Hint", hint_text)
        self.game.current_step += 1
        if self.game.current_step >= self.game.difficulty:
             self.show_victory()
        else:
             self.update_game_state_ui()

    def toggle_advanced_mode(self, state):
        self.advanced_mode = (state == Qt.CheckState.Checked.value)
        # Rebuild UI to show/hide parameters
        current_filter = self.combo_filter.currentText()
        if current_filter:
            self.build_parameters_ui(current_filter)

    def toggle_category_hints(self, state):
        self.category_hints = (state == Qt.CheckState.Checked.value)
        # Update status bar
        if self.game.original_image is not None:
            self.update_game_state_ui()

    def show_help(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Help - How to Play")
        dialog.resize(600, 500)
        
        layout = QVBoxLayout(dialog)
        
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setHtml("""
        <h2 style='color: #00d4ff;'>🎮 How to Play</h2>
        <p><b>Objective:</b> Recreate the target image by guessing which filters were applied to the original image.</p>
        
        <h3 style='color: #00d4ff;'>Steps:</h3>
        <ol>
            <li><b>Load an Image:</b> Click "📂 Load Image" to select an image file</li>
            <li><b>Choose Difficulty:</b> Select how many filter steps you want (2-6)</li>
            <li><b>Analyze:</b> Compare the Original and Target images</li>
            <li><b>Select Filter:</b> Choose a category and filter you think was applied</li>
            <li><b>Adjust Parameters:</b> In Advanced Mode, fine-tune the filter parameters</li>
            <li><b>Apply:</b> Click "✅ Apply Filter" to check your guess</li>
            <li><b>Progress:</b> If correct, move to the next step. If wrong, try again!</li>
        </ol>
        
        <h3 style='color: #00d4ff;'>Filter Categories:</h3>
        <ul>
            <li><b>Point Ops:</b> Negative, Log Transform, Gamma Correction, Contrast Stretch, Threshold</li>
            <li><b>Smoothing:</b> Averaging Blur, Gaussian Blur, Median Filter</li>
            <li><b>Edge Detection:</b> Sobel, Prewitt, Laplacian</li>
            <li><b>Frequency Domain:</b> DFT Low Pass, DFT High Pass</li>
        </ul>
        
        <h3 style='color: #00d4ff;'>Features:</h3>
        <ul>
            <li><b>🔬 Advanced Mode:</b> Shows mathematical formulas and enables parameter adjustment</li>
            <li><b>💡 Category Hints:</b> Shows which category the next filter belongs to</li>
            <li><b>💡 Hint/Skip:</b> Reveals the answer and skips the current step</li>
            <li><b>📋 View Solution:</b> Shows the complete pipeline with all parameters</li>
        </ul>
        
        <p style='color: #00ffff;'><b>Tip:</b> Enable Category Hints to narrow down your choices!</p>
        """)
        
        layout.addWidget(text_edit)
        
        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        btn_box.accepted.connect(dialog.accept)
        layout.addWidget(btn_box)
        
        dialog.exec()

    def show_solution(self):
        if not self.game.pipeline:
            QMessageBox.information(self, "No Game", "Please start a game first!")
            return
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Pipeline Solution")
        dialog.resize(500, 400)
        
        layout = QVBoxLayout(dialog)
        
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        
        solution_html = "<h2 style='color: #00d4ff;'>📋 Complete Pipeline Solution</h2>"
        
        for i, (filter_name, params) in enumerate(zip(self.game.pipeline, self.game.pipeline_params)):
            meta = FILTER_METADATA[filter_name]
            solution_html += f"<h3 style='color: #00ffff;'>Step {i+1}: {filter_name}</h3>"
            solution_html += f"<p><b>Category:</b> {meta['category']}</p>"
            
            if 'formula' in meta:
                solution_html += f"<p><b>Formula:</b> <code>{meta['formula']}</code></p>"
            
            if params:
                solution_html += "<p><b>Parameters:</b></p><ul>"
                for k, v in params.items():
                    solution_html += f"<li>{k}: {v}</li>"
                solution_html += "</ul>"
            else:
                solution_html += "<p><i>No parameters</i></p>"
            
            solution_html += "<hr>"
        
        text_edit.setHtml(solution_html)
        layout.addWidget(text_edit)
        
        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        btn_box.accepted.connect(dialog.accept)
        layout.addWidget(btn_box)
        
        dialog.exec()

    def show_full_report(self):
        """Show complete visual report with all intermediate images"""
        if not self.game.pipeline or not self.game.intermediates:
            QMessageBox.information(self, "No Game", "Please start a game first!")
            return
        
        dialog = QDialog(self)
        dialog.setWindowTitle("📊 Full Pipeline Report")
        dialog.resize(1000, 700)
        
        main_layout = QVBoxLayout(dialog)
        
        # Title
        title = QLabel("<h2 style='color: #00d4ff;'>📊 Complete Pipeline Report</h2>")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)
        
        # Scrollable area for images
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: #16213e; }")
        
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Add original image
        self._add_report_step(scroll_layout, 0, "Original Image", 
                             self.game.original_image, None, None)
        
        # Add each step with intermediate images
        for i, (filter_name, params, intermediate) in enumerate(zip(
            self.game.pipeline, 
            self.game.pipeline_params, 
            self.game.intermediates
        )):
            meta = FILTER_METADATA[filter_name]
            self._add_report_step(scroll_layout, i+1, filter_name, 
                                 intermediate, meta, params)
        
        scroll.setWidget(scroll_widget)
        main_layout.addWidget(scroll)
        
        # Buttons
        btn_box = QDialogButtonBox()
        btn_save = QPushButton("💾 Save Report as Images")
        btn_save.clicked.connect(lambda: self.save_report_images())
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dialog.accept)
        
        btn_box.addButton(btn_save, QDialogButtonBox.ButtonRole.ActionRole)
        btn_box.addButton(btn_close, QDialogButtonBox.ButtonRole.RejectRole)
        main_layout.addWidget(btn_box)
        
        dialog.exec()

    def _add_report_step(self, layout, step_num, title, image, meta, params):
        """Add a single step to the report with image and details"""
        step_frame = QFrame()
        step_frame.setObjectName("Card")
        step_frame.setStyleSheet("""
            QFrame#Card {
                background-color: #0f3460;
                border-radius: 12px;
                border: 2px solid #00d4ff;
                padding: 15px;
                margin: 10px;
            }
        """)
        
        step_layout = QHBoxLayout(step_frame)
        
        # Left side: Image
        img_label = QLabel()
        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_label.setStyleSheet("""
            background-color: #1a1a2e;
            border: 1px solid #00d4ff;
            border-radius: 8px;
            padding: 5px;
        """)
        
        if image is not None:
            # Convert and display image
            img_copy = image.copy()
            if len(img_copy.shape) == 2:
                img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2RGB)
            
            h, w, ch = img_copy.shape
            if img_copy.dtype != np.uint8:
                img_copy = img_copy.astype(np.uint8)
            
            bytes_per_line = ch * w
            img_copy = np.ascontiguousarray(img_copy)
            q_img = QImage(img_copy.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
            pixmap = QPixmap.fromImage(q_img)
            
            # Scale to reasonable size
            scaled_pixmap = pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio, 
                                         Qt.TransformationMode.SmoothTransformation)
            img_label.setPixmap(scaled_pixmap)
        
        img_label.setFixedSize(320, 320)
        step_layout.addWidget(img_label)
        
        # Right side: Details
        details_layout = QVBoxLayout()
        
        if step_num == 0:
            # Original image
            step_title = QLabel(f"<h3 style='color: #00ffff;'>📷 {title}</h3>")
            details_layout.addWidget(step_title)
            
            info = QLabel("<p style='color: #e0e0e0;'>This is the starting image before any filters are applied.</p>")
            info.setWordWrap(True)
            details_layout.addWidget(info)
        else:
            # Filter step
            step_title = QLabel(f"<h3 style='color: #00ffff;'>Step {step_num}: {title}</h3>")
            details_layout.addWidget(step_title)
            
            if meta:
                category_label = QLabel(f"<p><b style='color: #00d4ff;'>Category:</b> <span style='color: #e0e0e0;'>{meta['category']}</span></p>")
                details_layout.addWidget(category_label)
                
                if 'formula' in meta:
                    formula_label = QLabel(f"<p><b style='color: #00d4ff;'>Formula:</b></p><p style='color: #00ffff; font-family: Courier New; background-color: #1a1a2e; padding: 8px; border-radius: 4px;'>{meta['formula']}</p>")
                    formula_label.setWordWrap(True)
                    details_layout.addWidget(formula_label)
                
                if params:
                    params_text = "<p><b style='color: #00d4ff;'>Parameters:</b></p><ul style='color: #e0e0e0;'>"
                    for k, v in params.items():
                        params_text += f"<li>{k}: <span style='color: #00ffff;'>{v}</span></li>"
                    params_text += "</ul>"
                    params_label = QLabel(params_text)
                    params_label.setWordWrap(True)
                    details_layout.addWidget(params_label)
                else:
                    no_params = QLabel("<p style='color: #888; font-style: italic;'>No parameters</p>")
                    details_layout.addWidget(no_params)
        
        details_layout.addStretch()
        step_layout.addLayout(details_layout, stretch=1)
        
        layout.addWidget(step_frame)

    def save_report_images(self):
        """Save all intermediate images to a folder"""
        if not self.game.pipeline or not self.game.intermediates:
            QMessageBox.warning(self, "No Game", "No game data to save!")
            return
        
        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Save Report")
        if folder:
            try:
                import os
                
                # Save original
                orig_path = os.path.join(folder, "00_original.png")
                cv2.imwrite(orig_path, cv2.cvtColor(self.game.original_image, cv2.COLOR_RGB2BGR))
                
                # Save each intermediate
                for i, (filter_name, intermediate) in enumerate(zip(self.game.pipeline, self.game.intermediates)):
                    filename = f"{i+1:02d}_{filter_name.replace(' ', '_')}.png"
                    filepath = os.path.join(folder, filename)
                    
                    img_to_save = intermediate.copy()
                    if len(img_to_save.shape) == 2:
                        img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_GRAY2RGB)
                    
                    cv2.imwrite(filepath, cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR))
                
                # Save text report
                report_path = os.path.join(folder, "pipeline_report.txt")
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write("=" * 60 + "\n")
                    f.write("DIP GUESSING GAME - PIPELINE REPORT\n")
                    f.write("=" * 60 + "\n\n")
                    
                    f.write("ORIGINAL IMAGE\n")
                    f.write(f"File: 00_original.png\n\n")
                    
                    for i, (filter_name, params) in enumerate(zip(self.game.pipeline, self.game.pipeline_params)):
                        meta = FILTER_METADATA[filter_name]
                        f.write(f"STEP {i+1}: {filter_name}\n")
                        f.write(f"File: {i+1:02d}_{filter_name.replace(' ', '_')}.png\n")
                        f.write(f"Category: {meta['category']}\n")
                        if 'formula' in meta:
                            f.write(f"Formula: {meta['formula']}\n")
                        if params:
                            f.write("Parameters:\n")
                            for k, v in params.items():
                                f.write(f"  - {k}: {v}\n")
                        else:
                            f.write("Parameters: None\n")
                        f.write("\n")
                
                QMessageBox.information(self, "Success", 
                    f"Report saved successfully!\n\n"
                    f"Saved {len(self.game.intermediates) + 1} images and pipeline_report.txt to:\n{folder}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save report:\n{str(e)}")
    
    def start_mission_mode(self):
        """Start mission mode with educational scenarios"""
        dialog = QDialog(self)
        dialog.setWindowTitle("🎯 Mission Mode - Choose Your Challenge")
        dialog.resize(700, 600)
        
        layout = QVBoxLayout(dialog)
        
        title = QLabel("<h2 style='color: #00d4ff;'>🎯 Mission Mode</h2><p style='color: #e0e0e0;'>Learn WHY filters are used in real-world scenarios!</p>")
        title.setWordWrap(True)
        layout.addWidget(title)
        
        # Scrollable mission list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Group missions by difficulty
        for diff in [2, 3, 4]:
            diff_label = QLabel(f"<h3 style='color: #00ffff;'>Difficulty {diff}</h3>")
            scroll_layout.addWidget(diff_label)
            
            missions = get_mission_by_difficulty(diff)
            for mission_id, mission in missions.items():
                mission_btn = QPushButton(f"📋 {mission['title']}")
                mission_btn.setStyleSheet("""
                    QPushButton {
                        text-align: left;
                        padding: 15px;
                        font-size: 13px;
                        background-color: #0f3460;
                        border: 2px solid #00d4ff;
                        border-radius: 8px;
                    }
                    QPushButton:hover {
                        background-color: #16213e;
                        border: 2px solid #00ffff;
                    }
                """)
                mission_btn.clicked.connect(lambda checked, mid=mission_id: self.show_mission_details(mid, dialog))
                scroll_layout.addWidget(mission_btn)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btn_box.rejected.connect(dialog.reject)
        layout.addWidget(btn_box)
        
        dialog.exec()
    
    def show_mission_details(self, mission_id, parent_dialog):
        """Show detailed mission information"""
        mission = MISSIONS[mission_id]
        
        dialog = QDialog(self)
        dialog.setWindowTitle(f"🎯 {mission['title']}")
        dialog.resize(600, 500)
        
        layout = QVBoxLayout(dialog)
        
        # Mission details
        details = QTextEdit()
        details.setReadOnly(True)
        details.setHtml(f"""
        <h2 style='color: #00d4ff;'>{mission['title']}</h2>
        
        <h3 style='color: #00ffff;'>📋 Scenario:</h3>
        <p style='color: #e0e0e0;'>{mission['description']}</p>
        
        <h3 style='color: #00ffff;'>🎯 Your Goal:</h3>
        <p style='color: #e0e0e0;'>{mission['goal']}</p>
        
        <h3 style='color: #00ffff;'>💡 Hint:</h3>
        <p style='color: #ffff00;'>{mission['hint']}</p>
        
        <h3 style='color: #00ffff;'>📚 Why This Matters:</h3>
        <p style='color: #e0e0e0;'>{mission['context']}</p>
        
        <h3 style='color: #00ffff;'>🔧 Suggested Filters:</h3>
        <ul style='color: #00ffff;'>
        {''.join(f'<li>{f}</li>' for f in mission['suggested_filters'])}
        </ul>
        
        <h3 style='color: #00ffff;'>⚙️ Difficulty:</h3>
        <p style='color: #e0e0e0;'>{mission['difficulty']} steps</p>
        """)
        layout.addWidget(details)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        btn_start = QPushButton("🚀 Start This Mission")
        btn_start.clicked.connect(lambda: self.load_mission(mission_id, dialog, parent_dialog))
        btn_layout.addWidget(btn_start)
        
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setObjectName("SecondaryButton")
        btn_cancel.clicked.connect(dialog.reject)
        btn_layout.addWidget(btn_cancel)
        
        layout.addLayout(btn_layout)
        
        dialog.exec()
    
    def load_mission(self, mission_id, details_dialog, list_dialog):
        """Load a mission and start the game"""
        mission = MISSIONS[mission_id]
        
        # Close dialogs
        details_dialog.accept()
        list_dialog.accept()
        
        # Ask user to load an appropriate image
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("Load Image")
        msg.setText(f"Mission: {mission['title']}\n\nPlease load an image suitable for this scenario.\n\nFor best results, use an image that matches the mission context.")
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()
        
        # Load image
        path, _ = QFileDialog.getOpenFileName(self, "Open Image for Mission", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            try:
                # Start game with mission parameters
                self.mission_mode = True
                self.current_mission = mission
                
                diff = mission['difficulty']
                self.game.start_game(path, diff)
                self.last_image_path = path
                
                # Update UI
                self.update_game_state_ui()
                self.btn_reset.setEnabled(True)
                self.btn_apply.setEnabled(True)
                self.btn_hint.setEnabled(True)
                self.btn_solution.setEnabled(True)
                self.btn_report.setEnabled(True)
                
                # Show mission reminder
                self.lbl_status.setText(f"🎯 MISSION: {mission['title']} | Step 1 / {diff}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error Loading Image", f"Could not load image:\n{str(e)}")

    def show_victory(self):
        # Create victory message with pipeline summary
        victory_text = "🏆 Congratulations! You solved the pipeline!\n\n"
        victory_text += "Complete Solution:\n"
        for i, filter_name in enumerate(self.game.pipeline):
            victory_text += f"Step {i+1}: {filter_name}\n"
        
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("Victory!")
        msg.setText(victory_text)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        
        # Animate the message box (simple approach)
        msg.exec()
        
        self.btn_apply.setEnabled(False)
        self.btn_hint.setEnabled(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())