# digital-image-processing-game
Interactive educational game for learning Digital Image Processing through gameplay

# 🎮 DIP Guessing Game

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyQt6](https://img.shields.io/badge/PyQt6-6.x-green.svg)](https://pypi.org/project/PyQt6/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red.svg)](https://opencv.org)
[![License](https://img.shields.io/badge/License-Educational-yellow.svg)](LICENSE)

**Interactive Educational Game for Learning Digital Image Processing**

Transform how students learn image processing through hands-on gameplay! Players identify which filters were applied to transform images, learning by doing rather than just reading.

## 📋 Table of Contents
1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [All Features](#all-features)
4. [Complete Workflows](#complete-workflows)
5. [Filter Library](#filter-library)
6. [Mission Mode](#mission-mode)
7. [Technical Details](#technical-details)
8. [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

**DIP Guessing Game** is an interactive educational application for learning Digital Image Processing (DIP). Players identify which image filters were applied to transform an original image into a target image.

### Purpose
- **Educational**: Learn image processing concepts through gameplay
- **Interactive**: Hands-on experience with real filters
- **Professional**: Real-world scenarios through Mission Mode
- **Comprehensive**: 13 filters across 4 categories

### Key Statistics
- **13 Filters** across 4 categories
- **10 Mission Scenarios** with real-world context
- **50+ Features** including advanced learning tools
- **6 Difficulty Levels** (2-6 filter steps)

---

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/dip-guessing-game.git
cd dip-guessing-game

# Install dependencies
pip install PyQt6 opencv-python numpy scikit-image

# Run the game
python main.py
```

### First Game (30 seconds)
1. Click "📂 Load Image"
2. Select any image file
3. Choose difficulty (start with "3 Steps")
4. Compare Original vs Final Target
5. Select category and filter
6. Click "✅ Apply Filter"
7. Progress through all steps to win!

## 🎯 Features

### Core Game (10 features)
- ✅ **Image Loading** - PNG, JPG, JPEG, BMP support
- ✅ **6 Difficulty Levels** - 2-6 filter steps
- ✅ **Smart Pipeline Generation** - No black screens, always solvable
- ✅ **Three-Panel Display** - Input | Preview | Target
- ✅ **Real-time Preview** - Background threading, no UI freezing
- ✅ **Validation System** - SSIM + MSE for accurate checking
- ✅ **Progress Tracking** - Visual step-by-step progress
- ✅ **Victory Celebration** - Completion rewards
- ✅ **Error Handling** - Graceful failure recovery
- ✅ **Professional UI** - Dark theme, smooth animations

### Advanced Learning (8 features)
- 🔬 **Advanced Mode** - Mathematical formulas and parameter controls
- 💡 **Category Hints** - Shows filter category when stuck
- 🎯 **Mission Mode** - 10 real-world scenarios
- 🔥 **Hot/Cold Indicators** - Real-time parameter feedback
- 🌟 **Creative Solutions** - Accepts alternative filters (SSIM-based)
- 📊 **Full Reports** - Visual pipeline with all intermediate images
- 💾 **Export System** - Save reports for teaching
- ❓ **Help System** - Complete interactive guide

### Educational Tools (10 features)
- 📋 **Solution Viewer** - Text-based pipeline display
- 💡 **Hint System** - Reveals answers when stuck
- 📊 **Visual Reports** - Complete pipeline with images
- 💾 **Teaching Materials** - Export for presentations
- 🎓 **Mission Scenarios** - Real-world applications
- 📚 **Formula Display** - Mathematical foundations
- 🎮 **Progressive Difficulty** - Learn at your pace
- 🔄 **Unlimited Practice** - Reset and replay
- 📈 **Visual Feedback** - Immediate results
- 🏆 **Achievement System** - Creative solution bonuses

---

## 📖 COMPLETE WORKFLOWS

### For Students

#### Beginner Workflow (First Time)
```
1. SETUP (2 min)
   - Launch application
   - Load sample image
   - Select difficulty: 2-3 steps
   
2. ENABLE LEARNING AIDS (30 sec)
   ✅ Check "🔬 Advanced Mode"
   ✅ Check "💡 Category Hints"
   
3. PLAY & LEARN (5-10 min)
   - Analyze visual changes
   - Use category hint
   - Select filter
   - Adjust parameters (watch preview)
   - Apply and get feedback
   
4. USE HELP WHEN STUCK
   - Click "💡 Hint/Skip" for answer
   - Click "📊 Full Report" to see all steps
   - Learn from solutions
```

#### Advanced Workflow (Experienced)
```
1. Challenge Mode
   - Difficulty: 4-6 steps
   - Disable all hints
   - Time yourself
   
2. Creative Solutions
   - Try alternative filters
   - Experiment with parameters
   - Earn Creative Solution Bonus
   
3. Mission Mode
   - Click "🎯 Mission Mode"
   - Choose real-world scenario
   - Apply knowledge contextually
```

### For Teachers

#### Lecture Preparation (10 min)
```
1. GENERATE EXAMPLES
   - Load sample image
   - Select difficulty 3-4
   - Review generated pipeline
   - Regenerate if needed
   
2. SAVE MATERIALS
   - Click "📊 Full Report"
   - Review all steps visually
   - Click "💾 Save Report as Images"
   - Choose folder for materials
   
3. CREATE PRESENTATION
   - Import saved images to slides
   - Use pipeline_report.txt for details
   - Add explanations
   - Ready for class!
```

#### Live Demonstration (20 min)
```
1. SETUP (2 min)
   - Project on screen
   - Load interesting image
   - Show interface
   
2. INTERACTIVE CHALLENGE (10 min)
   - Show Original and Target
   - Students suggest filters
   - Try suggestions live
   - Discuss results
   
3. REVEAL SOLUTION (5 min)
   - Click "📊 Full Report"
   - Show each step
   - Explain formulas
   - Answer questions
   
4. STUDENT PRACTICE (3 min)
   - Students try on computers
   - Enable hints for beginners
   - Teacher helps as needed
```

#### Create Homework (15 min)
```
1. Generate 3 pipelines
2. Save each report
3. Create assignment document
4. Provide solution files
```

---

## 🎨 FILTER LIBRARY (13 Filters)

### Point Operations (5)
1. **Negative** - Inverts colors
2. **Log Transform** - Enhances dark regions
3. **Gamma Correction** - Adjusts brightness
4. **Contrast Stretch** - Enhances contrast
5. **Threshold** - Binary segmentation

### Smoothing (3)
6. **Averaging Blur** - Simple mean filter
7. **Gaussian Blur** - Weighted smoothing
8. **Median Filter** - Noise reduction

### Edge Detection (3)
9. **Sobel Edge** - Gradient-based
10. **Prewitt Edge** - Alternative gradient
11. **Laplacian** - Second derivative

### Frequency Domain (2)
12. **DFT Low Pass** - Removes high frequencies
13. **DFT High Pass** - Removes low frequencies

---

## 🎯 MISSION MODE (10 Scenarios)

Real-world applications with educational context:

1. **Medical Imaging** - Bone fracture detection
2. **Security** - CCTV footage enhancement
3. **Satellite** - Terrain analysis
4. **Document** - Text extraction for OCR
5. **Photo Restoration** - Old photo enhancement
6. **Forensics** - Fingerprint analysis
7. **Astronomy** - Nebula detail enhancement
8. **Manufacturing** - Defect detection
9. **Facial Recognition** - Preprocessing
10. **Biology** - Cell boundary detection

Each mission includes:
- Real-world scenario
- Clear goal
- Helpful hints
- Educational context
- Suggested filters

---

## 🔧 TECHNICAL DETAILS

### Architecture
- **Frontend**: PyQt6 (GUI)
- **Backend**: OpenCV (image processing)
- **Threading**: QThread (smooth UI)
- **Validation**: SSIM + MSE

### Dependencies
```
PyQt6==6.10.0
opencv-python==4.12.0
numpy==2.1.1
scikit-image==0.25.2
```

### File Structure
```
project/
├── main.py              # Main application
├── game_logic.py        # Game engine & filters
├── styling.py           # UI styling
├── missions.py          # Mission scenarios
└── Documentation/
    ├── README.md
    ├── COMPLETE_FEATURE_LIST.md
    ├── CRITICAL_FIXES_APPLIED.md
    ├── CRASH_FIXES.md
    └── INSTALLATION_GUIDE.md
```

---

## 🐛 TROUBLESHOOTING

### PowerShell Terminal Crash
**Error**: Exit code -2147023895
**Solution**: Switch to Command Prompt in VS Code/Kiro

### Application Won't Start
1. Check Python version (3.9+)
2. Reinstall dependencies
3. Update graphics drivers

### Images Not Loading
- Supported: PNG, JPG, JPEG, BMP
- Check file permissions
- Try different image

### UI Freezing
- Fixed with background threading
- Update to latest version

---

## 📚 Additional Documentation

- **COMPLETE_FEATURE_LIST.md** - All 50+ features detailed
- **CRITICAL_FIXES_APPLIED.md** - 6 major improvements
- **CRASH_FIXES.md** - Troubleshooting guide
- **INSTALLATION_GUIDE.md** - Setup instructions

---

## 🎓 Educational Value

### For Students
- Learn by doing
- Immediate feedback
- Visual understanding
- Real-world context

### For Teachers
- Ready-made examples
- Live demonstrations
- Homework generation
- Assessment tools

### For Self-Learners
- Progressive difficulty
- Comprehensive help
- Mission scenarios
- Creative freedom

---

## 🏆 Success Metrics

- **Learning**: Understand all 13 filters
- **Mastery**: Complete 6-step pipelines
- **Creativity**: Earn Creative Solution Bonuses
- **Application**: Complete all 10 missions

---

---

## 📁 Project Structure

```
dip-guessing-game/
├── main.py              # Main application (1,112 lines)
├── game_logic.py        # Game engine & filters (500+ lines)
├── styling.py           # UI styling (200+ lines)
├── missions.py          # Mission scenarios (150+ lines)
├── .kiro/specs/         # Complete specification
│   ├── requirements.md  # 15 requirements, 75 acceptance criteria
│   └── design.md        # Architecture, 29 correctness properties
├── Documentation/       # Comprehensive docs (12,000+ lines)
│   ├── README.md        # This file
│   ├── USER_GUIDE.md    # Complete user manual
│   ├── CODE_EXPLANATION.md # Every line explained
│   └── ... (8 more files)
└── requirements.txt     # Python dependencies
```

## 🔧 Technical Details

- **Language**: Python 3.9+
- **GUI**: PyQt6 (cross-platform)
- **Image Processing**: OpenCV
- **Validation**: SSIM (scikit-image)
- **Architecture**: Threaded, event-driven
- **Lines of Code**: ~2,000 (application) + ~12,000 (docs)

## 🎓 Educational Use

### For Students
- Learn by doing, not just reading
- Immediate visual feedback
- Progressive difficulty
- Real-world applications

### For Teachers
- Ready-made examples
- Live demonstration tool
- Export teaching materials
- Assessment capabilities

### For Self-Learners
- Comprehensive help system
- Mission-based learning
- Creative experimentation
- Complete documentation

## 🤝 Contributing

This is an educational project. Contributions welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📜 License

Educational Use - See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyQt6** - GUI framework
- **OpenCV** - Image processing
- **NumPy** - Numerical operations
- **scikit-image** - SSIM calculation

---

**⭐ Star this repository if you find it useful for learning Digital Image Processing!**
