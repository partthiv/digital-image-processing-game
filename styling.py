STYLESHEET = """
QMainWindow {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                stop:0 #1a1a2e, stop:1 #16213e);
}

QLabel {
    font-family: 'Segoe UI', sans-serif;
    color: #e0e0e0;
}

/* Header Styling */
QLabel#HeaderTitle {
    font-size: 28px;
    font-weight: bold;
    color: #00d4ff;
    padding: 10px;
}

/* Card Styling for Image Containers */
QFrame#Card {
    background-color: #0f3460;
    border-radius: 12px;
    border: 2px solid #00d4ff;
    padding: 10px;
}

QLabel#CardTitle {
    font-size: 16px;
    font-weight: bold;
    color: #00d4ff;
    padding-bottom: 8px;
}

QLabel#ImgLabel {
    background-color: #1a1a2e;
    border: 1px solid #00d4ff;
    border-radius: 8px;
    color: #888;
    padding: 20px;
}

/* Controls Section */
QFrame#ControlPanel {
    background-color: #0f3460;
    border-radius: 12px;
    border: 2px solid #00d4ff;
    padding: 20px;
}

QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #00d4ff, stop:1 #0099cc);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 14px;
    font-weight: bold;
}

QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #00ffff, stop:1 #00d4ff);
}

QPushButton:pressed {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #0099cc, stop:1 #007799);
}

QPushButton:disabled {
    background-color: #555;
    color: #888;
}

QPushButton#SecondaryButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #e94560, stop:1 #c72c41);
}

QPushButton#SecondaryButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #ff5577, stop:1 #e94560);
}

QComboBox {
    border: 2px solid #00d4ff;
    border-radius: 6px;
    padding: 8px;
    background-color: #1a1a2e;
    color: #e0e0e0;
    font-size: 13px;
    min-height: 25px;
}

QComboBox:hover {
    border: 2px solid #00ffff;
    background-color: #16213e;
}

QComboBox::drop-down {
    border: none;
    width: 30px;
}

QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 8px solid #00d4ff;
    margin-right: 8px;
}

QComboBox QAbstractItemView {
    background-color: #1a1a2e;
    color: #e0e0e0;
    selection-background-color: #00d4ff;
    selection-color: #1a1a2e;
    border: 2px solid #00d4ff;
    border-radius: 6px;
    padding: 5px;
    outline: none;
}

QComboBox QAbstractItemView::item {
    min-height: 30px;
    padding: 5px;
    border-radius: 4px;
}

QComboBox QAbstractItemView::item:hover {
    background-color: #0f3460;
    color: #00ffff;
}

QComboBox QAbstractItemView::item:selected {
    background-color: #00d4ff;
    color: #1a1a2e;
}

QSlider::groove:horizontal {
    border: 2px solid #00d4ff;
    background: #1a1a2e;
    height: 10px;
    border-radius: 6px;
}

QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 #00d4ff, stop:1 #0099cc);
    border-radius: 6px;
}

QSlider::handle:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #00ffff, stop:1 #00d4ff);
    border: 2px solid #1a1a2e;
    width: 20px;
    height: 20px;
    margin: -7px 0;
    border-radius: 10px;
}

QSlider::handle:horizontal:hover {
    background: #00ffff;
    border: 2px solid #00d4ff;
}

QCheckBox {
    color: #00d4ff;
    font-weight: bold;
    spacing: 8px;
}

QCheckBox::indicator {
    width: 20px;
    height: 20px;
    border: 2px solid #00d4ff;
    border-radius: 4px;
    background-color: #1a1a2e;
}

QCheckBox::indicator:hover {
    border: 2px solid #00ffff;
    background-color: #16213e;
}

QCheckBox::indicator:checked {
    background-color: #00d4ff;
    border: 2px solid #00d4ff;
}

QCheckBox::indicator:checked:hover {
    background-color: #00ffff;
}

QLabel#FormulaLabel {
    background-color: #1a1a2e;
    border: 2px solid #00d4ff;
    border-radius: 8px;
    padding: 10px;
    color: #00ffff;
    font-size: 13px;
    font-family: 'Courier New', monospace;
}

QTextEdit {
    background-color: #1a1a2e;
    color: #e0e0e0;
    border: 2px solid #00d4ff;
    border-radius: 8px;
    padding: 10px;
    font-size: 13px;
}

QDialog {
    background-color: #16213e;
}

QDialogButtonBox QPushButton {
    min-width: 80px;
}
"""