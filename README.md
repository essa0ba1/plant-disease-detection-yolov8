# Plant Disease Detection using YOLOv8

A deep learning-based plant disease detection system that uses YOLOv8 to identify and classify diseases in various plant leaves. The application provides interactive web interfaces built with both Streamlit and Gradio for easy image upload and disease analysis.
## Dataset source 
- [Plant Disease Detection (Roboflow, YOLOv8)]([https://universe.roboflow.com/smart-farmer/plant-disease-detection-3](https://universe.roboflow.com/graduation-project-2023/plants-diseases-detection-and-classification))


## 🌱 Features

- **Multi-Crop Support**: Detects diseases in multiple plant types including:
  - **Apple**: Scab Leaf, Leaf (healthy), Rust Leaf
  - **Bell Pepper**: Leaf Spot, Leaf (healthy)
  - **Blueberry**: Leaf
  - **Cherry**: Leaf
  - **Corn**: Gray Leaf Spot, Leaf Blight, Rust Leaf
  - **Grape**: Leaf Black Rot, Leaf (healthy)
  - **Peach**: Leaf
  - **Potato**: Leaf Early Blight, Leaf Late Blight, Leaf (healthy)
  - **Raspberry**: Leaf
  - **Soyabean**: Leaf
  - **Squash**: Powdery Mildew Leaf
  - **Strawberry**: Leaf
  - **Tomato**: Early Blight Leaf, Septoria Leaf Spot, Leaf Bacterial Spot, Leaf Late Blight, Leaf Mosaic Virus, Leaf Yellow Virus, Leaf (healthy), Mold Leaf, Two Spotted Spider Mites Leaf

- **Real-time Detection**: Upload images and get instant disease detection results
- **Visual Annotations**: Bounding boxes with color-coded disease classifications
- **Statistical Analysis**: Interactive bar charts showing disease frequency distribution
- **Export Results**: Download annotated images with detection results

## 🚀 Technologies Used

- **YOLOv8** (Ultralytics) - Object detection model
- **Streamlit** - Web application framework (Streamlit version)
- **Gradio** - Web application framework (Gradio version)
- **Plotly** - Interactive data visualization
- **PIL (Pillow)** - Image processing
- **NumPy** - Numerical operations
- **OpenCV** - Computer vision operations

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager

## 🔧 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd plant-disease-detection-yolov8--main
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   The required packages include:
   - `ultralytics==8.1.10`
   - `streamlit==1.29.0` (for Streamlit version)
   - `gradio>=4.0.0` (for Gradio version)
   - `plotly==5.18.0`
   - `Pillow==10.1.0`
   - `numpy==1.24.3`
   

3. **Ensure model file is present**:
   - The project includes a pre-trained model file `best.pt`
   - Make sure this file is in the project root directory

## 💻 Usage

### Streamlit Version

1. **Launch the Streamlit application**:
   ```bash
   streamlit run main.py
   ```

2. **Access the web interface**:
   - The application will open in your default web browser
   - Typically available at `http://localhost:8501`

3. **Upload and analyze images**:
   - Click on the file uploader to select a plant leaf image
   - Adjust the confidence threshold slider (default: 0.25)
   - Click the "start analyzing .." button to process the image
   - View the annotated image with bounding boxes, labels, and confidence scores
   - Check the frequency chart for detected diseases
   - Download the annotated image if needed

### Gradio Version

1. **Launch the Gradio application**:
   ```bash
   python app.py
   ```

2. **Access the web interface**:
   - The application will open in your default web browser
   - Typically available at `http://localhost:7860`

3. **Upload and analyze images**:
   - View the expandable section showing all 29 detectable diseases
   - Upload a plant leaf image using the image uploader
   - Adjust the confidence threshold slider (default: 0.25)
   - Click "Start Analyzing" or let it auto-update when you change the threshold
   - View the annotated image with bounding boxes, labels, and confidence scores
   - Check the interactive frequency chart for detected diseases

## 🧪 Testing the Model

**Best Way to Test: Using Google Images**

The best way to test the model is by using images from Google Images. Here's why and how:

### Why Google Images?
- **Diverse Dataset**: Google Images provides a wide variety of plant disease images from different sources, lighting conditions, and angles
- **Real-world Scenarios**: Images from Google are more representative of real-world conditions than synthetic or curated datasets
- **Easy Access**: Quickly find and test various plant diseases without needing a physical collection
- **Validation**: Helps verify the model's performance across different image qualities and styles

### How to Test:
1. **Search for specific plant diseases** on Google Images:
   - Example searches: "tomato leaf blight", "apple scab leaf", "corn rust leaf", etc.
2. **Download test images**:
   - Right-click on images and save them to your computer
   - Try different image qualities and resolutions
3. **Upload to the application**:
   - Use the file uploader in either Streamlit or Gradio interface
   - Test with various confidence thresholds (0.1 to 0.9)
4. **Evaluate results**:
   - Check if the model correctly identifies the disease
   - Verify bounding boxes are accurate
   - Compare confidence scores with visual assessment

### Tips for Better Testing:
- Test with **multiple images** of the same disease to check consistency
- Try **different image qualities** (high resolution, low resolution, compressed)
- Test **edge cases** (multiple diseases in one image, healthy leaves, unclear images)
- Adjust **confidence threshold** to see how it affects detection sensitivity
- Compare results with **known ground truth** when available

## 📊 Model Details

- **Architecture**: YOLOv8 (Medium variant - yolov8m)
- **Training**: The model was trained on a custom dataset with 50 epochs
- **Confidence Threshold**: 0.25 (configurable via UI slider)
- **Classes**: 29 different plant disease classes (including healthy leaves)

## 📁 Project Structure

```
plant-disease-detection-yolov8--main/
│
├── main.py                 # Main Streamlit application
├── app.py          # Gradio application (alternative UI)
├── best.pt                 # Pre-trained YOLOv8 model
├── requirements.txt        # Python dependencies
├── yolov8_plant.ipynb      # Jupyter notebook for training
├── output.png             # Output image (generated after analysis)
├── images.jpeg            # Sample images
└── README.md              # This file
```

## 🎨 Supported Disease Classes

The model can detect the following disease classes:

1. Apple Scab Leaf
2. Apple leaf
3. Apple rust leaf
4. Bell_pepper leaf spot
5. Bell_pepper leaf
6. Blueberry leaf
7. Cherry leaf
8. Corn Gray leaf spot
9. Corn leaf blight
10. Corn rust leaf
11. Peach leaf
12. Potato leaf early blight
13. Potato leaf late blight
14. Potato leaf
15. Raspberry leaf
16. Soyabean leaf
17. Squash Powdery mildew leaf
18. Strawberry leaf
19. Tomato Early blight leaf
20. Tomato Septoria leaf spot
21. Tomato leaf bacterial spot
22. Tomato leaf late blight
23. Tomato leaf mosaic virus
24. Tomato leaf yellow virus
25. Tomato leaf
26. Tomato mold leaf
27. Tomato two spotted spider mites leaf
28. Grape leaf black rot
29. Grape leaf

## 🔬 Model Training

The model training process is documented in `yolov8_plant.ipynb`. Key training parameters:
- **Epochs**: 50
- **Learning Rate**: 0.0005
- **Data Augmentation**: Enabled
- **Base Model**: YOLOv8 Medium (yolov8m.pt)

## 📝 Notes

- The model uses a confidence threshold of 0.25 by default (adjustable via UI slider)
- Each disease class has a unique color for visualization
- The application tracks and displays the frequency of detected diseases
- Results are saved as `output.png` in the project directory (Streamlit version)
- The Gradio version provides a more modern UI with expandable disease information
- **Recommended Testing**: Use Google Images to find diverse test cases for better model validation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available for educational and research purposes.

## 🙏 Acknowledgments

- Ultralytics for the YOLOv8 framework
- Streamlit for the web framework (Streamlit version)
- Gradio for the web framework (Gradio version)
- The dataset contributors for plant disease images

## 📧 Contact

For questions or issues, please open an issue in the repository.

---

**Note**: Make sure you have sufficient computational resources (GPU recommended) for optimal performance, especially during model training.

