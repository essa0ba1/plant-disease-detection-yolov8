from ultralytics import YOLO  
import streamlit as st 
import plotly.graph_objs as go
from PIL import Image , ImageFilter as F
import numpy as np 
from io import BytesIO
colors = {
    'Apple Scab Leaf': ['rgb(165, 42, 42)', (165, 42, 42)],  # Brown
    'Apple leaf': ['rgb(128, 0, 128)', (128, 0, 128)],  # Purple
    'Apple rust leaf': ['rgb(0, 255, 0)', (0, 255, 0)],  # Green
    'Bell_pepper leaf spot': ['rgb(255, 215, 0)', (255, 215, 0)],  # Gold
    'Bell_pepper leaf': ['rgb(139, 69, 19)', (139, 69, 19)],  # Brown
    'Blueberry leaf': ['rgb(128, 128, 128)', (128, 128, 128)],  # Gray
    'Cherry leaf': ['rgb(0, 128, 0)', (0, 128, 0)],  # Dark green
    'Corn Gray leaf spot': ['rgb(255, 0, 0)', (255, 0, 0)],  # Red
    'Corn leaf blight': ['rgb(255, 165, 0)', (255, 165, 0)],  # Orange
    'Corn rust leaf': ['rgb(139, 0, 0)', (139, 0, 0)],  # Dark red
    'Peach leaf': ['rgb(255, 20, 147)', (255, 20, 147)],  # Pink
    'Potato leaf early blight': ['rgb(255, 105, 180)', (255, 105, 180)],  # Hot pink
    'Potato leaf late blight': ['rgb(0, 0, 139)', (0, 0, 139)],  # Dark blue
    'Potato leaf': ['rgb(218, 112, 214)', (218, 112, 214)],  # Orchid
    'Raspberry leaf': ['rgb(255, 0, 255)', (255, 0, 255)],  # Magenta
    'Soyabean leaf': ['rgb(255, 69, 0)', (255, 69, 0)],  # Red-orange
    'Squash Powdery mildew leaf': ['rgb(0, 255, 255)', (0, 255, 255)],  # Cyan
    'Strawberry leaf': ['rgb(255, 255, 0)', (255, 255, 0)],  # Yellow
    'Tomato Early blight leaf': ['rgb(154, 205, 50)', (154, 205, 50)],  # Yellow-green
    'Tomato Septoria leaf spot': ['rgb(0, 0, 255)', (0, 0, 255)],  # Blue
    'Tomato leaf bacterial spot': ['rgb(255, 99, 71)', (255, 99, 71)],  # Tomato
    'Tomato leaf late blight': ['rgb(46, 139, 87)', (46, 139, 87)],  # Sea green
    'Tomato leaf mosaic virus': ['rgb(255, 192, 203)', (255, 192, 203)],  # Pink
    'Tomato leaf yellow virus': ['rgb(173, 255, 47)', (173, 255, 47)],  # Green-yellow
    'Tomato leaf': ['rgb(0, 128, 128)', (0, 128, 128)],  # Teal
    'Tomato mold leaf': ['rgb(128, 0, 0)', (128, 0, 0)],  # Maroon
    'Tomato two spotted spider mites leaf': ['rgb(70, 130, 180)', (70, 130, 180)],  # Steel blue
    'grape leaf black rot': ['rgb(0, 255, 127)', (0, 255, 127)],  # Spring green
    'grape leaf': ['rgb(34, 139, 34)', (34, 139, 34)]  # Forest green
}

frequencies = {
    'Apple Scab Leaf': 0,
    'Apple leaf': 0,
    'Apple rust leaf': 0,
    'Bell_pepper leaf spot': 0,
    'Bell_pepper leaf': 0,
    'Blueberry leaf': 0,
    'Cherry leaf': 0,
    'Corn Gray leaf spot': 0,
    'Corn leaf blight': 0,
    'Corn rust leaf': 0,
    'Peach leaf': 0,
    'Potato leaf early blight': 0,
    'Potato leaf late blight': 0,
    'Potato leaf': 0,
    'Raspberry leaf': 0,
    'Soyabean leaf': 0,
    'Squash Powdery mildew leaf': 0,
    'Strawberry leaf': 0,
    'Tomato Early blight leaf': 0,
    'Tomato Septoria leaf spot': 0,
    'Tomato leaf bacterial spot': 0,
    'Tomato leaf late blight': 0,
    'Tomato leaf mosaic virus': 0,
    'Tomato leaf yellow virus': 0,
    'Tomato leaf': 0,
    'Tomato mold leaf': 0,
    'Tomato two spotted spider mites leaf': 0,
    'grape leaf black rot': 0,
    'grape leaf': 0
}

model = YOLO("best.pt").to("cpu")

st.title('Plants (Apple,tomato,corn,..) disease detector using yolov8 ')

# Confidence threshold slider
confidence_threshold = st.slider(
    "**Confidence Threshold**",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05,
    help="Adjust the confidence threshold for detections. Lower values show more detections (including less confident ones), higher values show only highly confident detections."
)

image  = st.file_uploader("**put your image for examintion  :** ")


if image is not None :
    
    image = Image.open(image).convert("RGB")
    image_filtered = image.filter(F.SHARPEN)
    image_np = np.array(image)
    
    button = st.button("start analyzing .." , type="primary")

    if button :
        # Run prediction with the selected confidence threshold
        result = model.predict(source=image_np, conf=confidence_threshold, save=False)
        names = result[0].names 
        data = result[0].boxes.data.numpy() 
        # Reset frequencies for new analysis
        for key in frequencies:
            frequencies[key] = 0
        
        # Extract class names and confidence from detections
        for bbox in data:
            class_id = int(bbox[5])
            confidence = float(bbox[4])
            class_name = names[class_id]
            
            # Update frequencies
            if class_name in frequencies:
                frequencies[class_name] += 1
            else:
                frequencies[class_name] = 1
        
        # Use Ultralytics built-in plotting with labels and confidence
        annotated_image = result[0].plot()  # This includes labels and confidence scores
        annotated_image_pil = Image.fromarray(annotated_image)
        #annotated_image_pil.save("output.png")
        
        # Display confidence threshold info
        st.info(f"**Confidence Threshold:** {confidence_threshold:.2f} | **Detections Found:** {len(data)}")
        


        x = list(frequencies.values())
        y = list(frequencies.keys())
        # Get colors for each class, use default if not found
        colors_list = [colors[key][0] if key in colors else 'rgb(128, 128, 128)' for key in y]
        # Create a bar plot
        fig = go.Figure(data=[go.Bar(x=y, y=x, marker_color=colors_list)])

        # Display image in the first column
        
        st.image("output.png", caption='Annotated Image with Labels and Confidence Scores',)
        st.download_button(
                label="Download image",
                data=BytesIO(annotated_image_pil.tobytes()),
                file_name="result_image.jpg",
                key="download_button",
                help="Click to download the annotated image.",
            )

        # Display frequencies in the second column

        st.plotly_chart(fig, use_container_width=True)
