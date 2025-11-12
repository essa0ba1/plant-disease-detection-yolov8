from ultralytics import YOLO  
import gradio as gr
import plotly.graph_objs as go
from PIL import Image, ImageFilter as F
import numpy as np 
import io
import base64

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

# Load model
model = YOLO("best.pt").to("cpu")

# Get all class names from the model
def get_all_detectable_classes():
    """Get all class names that the model can detect"""
    # Get class names from model (this will be available after first prediction or from model metadata)
    try:
        # Try to get from model metadata
        if hasattr(model, 'names') and model.names:
            return model.names
        else:
            # Fallback to the frequencies dictionary keys
            return {i: name for i, name in enumerate(frequencies.keys())}
    except:
        # Return from frequencies as fallback
        return {i: name for i, name in enumerate(frequencies.keys())}

# Organize classes by plant type for display
def create_disease_info_html():
    """Create HTML content showing all detectable diseases organized by plant type"""
    classes_dict = get_all_detectable_classes()
    all_classes = list(classes_dict.values())
    
    # Organize by plant type
    plant_diseases = {
        "🍎 Apple": ["Apple Scab Leaf", "Apple leaf", "Apple rust leaf"],
        "🫑 Bell Pepper": ["Bell_pepper leaf spot", "Bell_pepper leaf"],
        "🫐 Blueberry": ["Blueberry leaf"],
        "🍒 Cherry": ["Cherry leaf"],
        "🌽 Corn": ["Corn Gray leaf spot", "Corn leaf blight", "Corn rust leaf"],
        "🍇 Grape": ["grape leaf black rot", "grape leaf"],
        "🍑 Peach": ["Peach leaf"],
        "🥔 Potato": ["Potato leaf early blight", "Potato leaf late blight", "Potato leaf"],
        "🫐 Raspberry": ["Raspberry leaf"],
        "🫘 Soyabean": ["Soyabean leaf"],
        "🎃 Squash": ["Squash Powdery mildew leaf"],
        "🍓 Strawberry": ["Strawberry leaf"],
        "🍅 Tomato": [
            "Tomato Early blight leaf", "Tomato Septoria leaf spot", 
            "Tomato leaf bacterial spot", "Tomato leaf late blight",
            "Tomato leaf mosaic virus", "Tomato leaf yellow virus",
            "Tomato leaf", "Tomato mold leaf", "Tomato two spotted spider mites leaf"
        ]
    }
    
    html = "<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 10px 0;'>"
    html += "<h3 style='color: #1f77b4; margin-bottom: 15px;'>📋 All Detectable Diseases (29 Classes)</h3>"
    
    for plant, diseases in plant_diseases.items():
        html += f"<div style='margin-bottom: 15px;'>"
        html += f"<strong style='font-size: 16px; color: #2c3e50;'>{plant}</strong><br>"
        html += "<div style='margin-left: 20px; margin-top: 5px;'>"
        for disease in diseases:
            # Check if it's a healthy leaf (just "X leaf" without disease name) or disease
            is_healthy = (
                disease.lower().endswith("leaf") and 
                not any(keyword in disease.lower() for keyword in ["scab", "rust", "spot", "blight", "mosaic", "virus", "bacterial", "mold", "mites", "mildew", "rot"])
            )
            
            if is_healthy:
                html += f"<span style='color: #27ae60;'>✓ {disease} (Healthy)</span><br>"
            else:
                html += f"<span style='color: #e74c3c;'>⚠ {disease} (Disease)</span><br>"
        html += "</div></div>"
    
    html += f"<p style='margin-top: 15px; color: #7f8c8d;'><strong>Total: {len(all_classes)} detectable classes</strong></p>"
    html += "</div>"
    
    return html

def analyze_image(input_image, confidence_threshold):
    """
    Analyze plant image for disease detection
    
    Args:
        input_image: PIL Image or numpy array
        confidence_threshold: float between 0.0 and 1.0
    
    Returns:
        annotated_image: PIL Image with bounding boxes
        plot_figure: Plotly figure object for the frequency chart
        info_text: String with detection statistics
    """
    if input_image is None:
        # Return empty figure if no image
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Disease Detection Frequency",
            xaxis_title="Disease Class",
            yaxis_title="Count",
            height=400
        )
        return None, empty_fig, "Please upload an image"
    
    try:
        # Convert to PIL Image if needed
        if isinstance(input_image, np.ndarray):
            image = Image.fromarray(input_image)
        else:
            image = input_image
        
        image = image.convert("RGB")
        image_filtered = image.filter(F.SHARPEN)
        image_np = np.array(image)
        
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
        
        # Create frequency chart - filter out zero frequencies
        filtered_freq = {k: v for k, v in frequencies.items() if v > 0}
        
        if len(filtered_freq) > 0:
            x = list(filtered_freq.values())
            y = list(filtered_freq.keys())
            # Get colors for each class, use default if not found
            colors_list = [colors[key][0] if key in colors else 'rgb(128, 128, 128)' for key in y]
            
            # Create a bar plot
            fig = go.Figure(data=[go.Bar(x=y, y=x, marker_color=colors_list)])
            fig.update_layout(
                title="Disease Detection Frequency",
                xaxis_title="Disease Class",
                yaxis_title="Count",
                height=400,
                xaxis_tickangle=-45,
                showlegend=False
            )
        else:
            # No detections - show empty plot with message
            fig = go.Figure()
            fig.update_layout(
                title="Disease Detection Frequency - No Detections Found",
                xaxis_title="Disease Class",
                yaxis_title="Count",
                height=400,
                annotations=[dict(
                    x=0.5, y=0.5,
                    xref="paper", yref="paper",
                    text="No diseases detected at this confidence threshold",
                    showarrow=False,
                    font=dict(size=16, color="gray")
                )]
            )
        
        # Create info text
        info_text = f"**Confidence Threshold:** {confidence_threshold:.2f} | **Detections Found:** {len(data)}"
        
        return annotated_image_pil, fig, info_text
    
    except Exception as e:
        # Return empty figure on error
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Disease Detection Frequency",
            xaxis_title="Disease Class",
            yaxis_title="Count",
            height=400
        )
        return None, empty_fig, f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Plant Disease Detection using YOLOv8", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌱 Plants Disease Detector using YOLOv8")
    gr.Markdown("Upload an image of a plant leaf to detect diseases. Adjust the confidence threshold to control detection sensitivity.")
    
    # Add expandable section for detectable diseases (open by default so users can see what's detectable)
    with gr.Accordion("📋 All Detectable Diseases (29 classes) - Click to expand/collapse", open=True):
        disease_info_html = create_disease_info_html()
        gr.HTML(disease_info_html)
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                label="Upload Plant Image",
                type="pil",
                height=400
            )
            
            confidence_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.25,
                step=0.05,
                label="Confidence Threshold",
                info="Lower values show more detections (including less confident ones), higher values show only highly confident detections."
            )
            
            analyze_btn = gr.Button("🔍 Start Analyzing", variant="primary", size="lg")
        
        with gr.Column():
            image_output = gr.Image(
                label="Annotated Image with Labels and Confidence Scores",
                type="pil",
                height=400
            )
            
            info_output = gr.Markdown()
            
            plot_output = gr.Plot(
                label="Disease Detection Frequency Chart"
            )
    
    # Set up the function call
    analyze_btn.click(
        fn=analyze_image,
        inputs=[image_input, confidence_slider],
        outputs=[image_output, plot_output, info_output]
    )
    
    # Also trigger on confidence change (optional)
    confidence_slider.change(
        fn=analyze_image,
        inputs=[image_input, confidence_slider],
        outputs=[image_output, plot_output, info_output]
    )
    
    with gr.Row():
        gr.Markdown("""
        ### 📊 Quick Reference
        
        **Supported Plants:** Apple, Bell Pepper, Blueberry, Cherry, Corn, Grape, Peach, Potato, Raspberry, Soyabean, Squash, Strawberry, Tomato
        
        **Total Detectable Classes:** 29 (including healthy leaves and various diseases)
        
        💡 **Tip:** Use the expandable section above to see the complete list of all detectable diseases organized by plant type.
        """)

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)

