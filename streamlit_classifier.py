import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
# from PIL import Image
from PIL import Image, ImageOps
import requests
import json
from datetime import datetime
import io
import tempfile
import os

# Configure Streamlit page
st.set_page_config(
    page_title="Mango Tree Classifier",
    page_icon="🥭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E7D32;
        margin-bottom: 30px;
    }
    .upload-section {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .result-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-bar {
        background: #e0e0e0;
        border-radius: 10px;
        height: 20px;
        margin: 5px 0;
        overflow: hidden;
    }
    .prediction-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    .mango-prediction {
        background: #4CAF50;
    }
    .not-mango-prediction {
        background: #f44336;
    }
    .other-prediction {
        background: #2196F3;
    }
    .stProgress .stAlert {
        background: transparent;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'classification_results' not in st.session_state:
    st.session_state.classification_results = []

# Configuration - Update these paths for your SavedModel
SAVEDMODEL_PATH = st.secrets.get("SAVEDMODEL_PATH", "./models/mango_classifier_savedmodel")
# Alternative: You can also use a model from TensorFlow Hub or cloud storage
# SAVEDMODEL_PATH = "gs://your-bucket/models/mango_classifier"
# SAVEDMODEL_PATH = "https://tfhub.dev/your-model/1"

@st.cache_resource
def load_tensorflow_savedmodel(model_path):
    """
    Load TensorFlow SavedModel from local path or URL
    JavaScript equivalent: loadModel() function
    """
    try:
        st.info("🔄 Loading TensorFlow SavedModel...")
        
        # Load the SavedModel
        model = keras.layers.TFSMLayer(SAVEDMODEL_PATH, call_endpoint='serving_default')
        
        # For Teachable Machine models, class labels are typically ordered
        # You can customize this based on your specific model
        class_labels = ['mango_tree', 'not_mango_tree'] 
        
        st.success("✅ SavedModel loaded successfully!")
        
        return {
            'model': model,
            'labels': class_labels,
            'loaded': True,
            'path': model_path
        }
        
    except Exception as e:
        st.error(f"❌ Error loading SavedModel: {str(e)}")
        st.error("💡 Tips:")
        st.error("  - Make sure the model path is correct")
        st.error("  - Ensure the SavedModel directory contains saved_model.pb")
        st.error("  - Check that all required files are present")
        return None

def preprocess_image(image):
    """
    Preprocess image for prediction
    JavaScript equivalent: Image preprocessing in classifyImage()
    """

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # # Resize image to model input size (typically 224x224 for Teachable Machine)
    # image = image.resize((224, 224))
    
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)


    image_array = np.asarray(image)


    # Convert to numpy array and normalize
    image_array = np.array(image_array.astype(np.float32)) / 255.0

    data[0] = image_array
    
    # # Add batch dimension
    # image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def classify_image_with_savedmodel(image, model_info):
    """
    Real image classification using TensorFlow SavedModel
    JavaScript equivalent: classifyImage() function
    """
    try:
        if not model_info or 'model' not in model_info:
            raise ValueError("Model not properly loaded")
        
        model = model_info['model']
        labels = model_info['labels']
        
        # Preprocess the image for the model
        processed_image = preprocess_image(image)
        
        st.write("image preprocessed for classification")
        # Make prediction using the SavedModel
        with st.spinner("🔍 Classifying image..."):
            predictions = model(processed_image)#, verbose=0)
        st.write("prediction daone")
        # Handle different output formats
        if len(predictions.shape) == 2:  # Shape: (1, num_classes)
            prediction_probs = predictions[0]
        else:  # Shape: (num_classes,)
            prediction_probs = predictions
        
        # Convert predictions to the expected format
        prediction_results = []
        for i, prob in enumerate(prediction_probs):
            if i < len(labels):
                prediction_results.append({
                    'className': labels[i],
                    'probability': float(prob)
                })
        
        # Ensure probabilities sum to 1 (normalize if needed)
        total_prob = sum(pred['probability'] for pred in prediction_results)
        if total_prob > 0:
            for pred in prediction_results:
                pred['probability'] = pred['probability'] / total_prob
        
        # Sort by probability (highest first) to match original behavior
        prediction_results.sort(key=lambda x: x['probability'], reverse=True)
        
        return prediction_results
        
    except Exception as e:
        st.error(f"❌ Error during classification: {str(e)}")
        return None

def display_predictions(predictions, image_name):
    """
    Display prediction results
    JavaScript equivalent: Predictions display in result card
    """
    st.markdown(f"### 📊 Results for {image_name}")
    
    # Sort predictions by priority (mango_tree first, then by probability)
    def get_prediction_priority(pred):
        if pred['className'].lower() == 'mango_tree':
            return (1, -pred['probability'])
        elif pred['className'].lower() == 'not_mango_tree':
            return (2, -pred['probability'])
        else:
            return (3, -pred['probability'])
    
    sorted_predictions = sorted(predictions, key=get_prediction_priority)
    
    for i, pred in enumerate(sorted_predictions):
        class_name = pred['className']
        probability = pred['probability']
        percentage = probability * 100
        
        # Determine color based on class
        if class_name.lower() == 'mango_tree':
            color = '#4CAF50'  # Green
        elif class_name.lower() == 'not_mango_tree':
            color = '#f44336'  # Red  
        else:
            color = '#2196F3'  # Blue
        
        # Display prediction with progress bar
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**{class_name}**")
            st.progress(probability)
        
        with col2:
            st.markdown(f"**{percentage:.1f}%**")
        
        st.markdown("---")

def main():
    """
    Main Streamlit application
    JavaScript equivalent: TeachableMachineImageClassifier component
    """
    # Header
    st.markdown('<h1 class="main-header">🥭 Teachable Machine Mango Tree Classifier</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 18px;">Upload images to classify mango trees using AI</p>', 
                unsafe_allow_html=True)
    
    # Load model on first run
    if not st.session_state.model_loaded:
        with st.spinner("Loading AI model..."):
            model_info = load_tensorflow_savedmodel(SAVEDMODEL_PATH)
            if model_info:
                st.session_state.model = model_info
                st.session_state.model_loaded = True
                st.success("✅ SavedModel loaded successfully!")
                
                # Display model info
                with st.expander("📊 Model Information"):
                    st.write(f"**Model Path:** `{SAVEDMODEL_PATH}`")
                    st.write(f"**Classes:** {', '.join(model_info['labels'])}")
                    st.write(f"**Model Type:** TensorFlow SavedModel")
                    
                    # Display model architecture summary if available
                    try:
                        model_summary = []
                        model_info['model'].summary(print_fn=lambda x: model_summary.append(x))
                        st.code('\n'.join(model_summary[:10]) + '\n...' if len(model_summary) > 10 else '\n'.join(model_summary))
                    except:
                        st.write("Model architecture details not available")
            else:
                st.error("❌ Failed to load SavedModel. Please check the model path.")
                st.stop()
    
    # Upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload Images")
    
    uploaded_files = st.file_uploader(
        "Choose image files",
        type=['jpg', 'jpeg', 'png', 'bmp', 'gif'],
        accept_multiple_files=True,
        help="Select one or more image files for classification"
    )
    
    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) uploaded**")
        
        # Process button
        if st.button("🔍 Classify Images", type="primary"):
            process_images(uploaded_files)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Results section
    if st.session_state.classification_results:
        st.markdown("## 📋 Classification Results")
        
        # Clear results button
        if st.button("🗑️ Clear All Results"):
            st.session_state.classification_results = []
            st.experimental_rerun()
        
        # Display results
        for result in reversed(st.session_state.classification_results):  # Show newest first
            display_result_card(result)
    
    else:
        # Empty state
        st.markdown("### 💡 Instructions")
        st.info("""
        1. 📁 **Prepare your SavedModel**: Make sure you have a TensorFlow SavedModel directory
        2. 📂 **Update model path**: Set `SAVEDMODEL_PATH` in your code or Streamlit secrets
        3. 🔍 **Upload images**: Use the file uploader above to select your images
        4. 📊 **View results**: Click 'Classify Images' to analyze your photos
        5. 🥭 **Get predictions**: The AI will classify each image with confidence scores
        
        **SavedModel Requirements:**
        - Model should accept images of shape (224, 224, 3)
        - Input should be normalized to [0, 1] range
        - Output should be class probabilities
        """)

def process_images(uploaded_files):
    """
    Process multiple uploaded images
    JavaScript equivalent: handleFileUpload() function
    """
    if not st.session_state.model:
        st.error("Model not loaded!")
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_files = len(uploaded_files)
    
    for i, uploaded_file in enumerate(uploaded_files):
        # Update progress
        progress = (i + 1) / total_files
        progress_bar.progress(progress)
        status_text.text(f"Processing {uploaded_file.name}... ({i+1}/{total_files})")
        
        try:
            # Load and preprocess image
            image = Image.open(uploaded_file)
            
            # Classify image
            predictions = classify_image_with_savedmodel(image, st.session_state.model)
            
            if predictions:
                # Create result object
                result = {
                    'id': len(st.session_state.classification_results) + 1,
                    'file_name': uploaded_file.name,
                    'image': image,
                    'predictions': predictions,
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'file_size': uploaded_file.size
                }
                
                st.session_state.classification_results.append(result)
        
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    st.success(f"✅ Successfully processed {total_files} image(s)!")

def display_result_card(result):
    """
    Display individual result card
    JavaScript equivalent: Result card in results grid
    """
    with st.container():
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        
        # Create columns for image and info
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Display image
            st.image(result['image'], caption=result['file_name'], use_column_width=True)
        
        with col2:
            # File info
            st.markdown(f"**📁 {result['file_name']}**")
            st.markdown(f"🕒 Processed at {result['timestamp']}")
            st.markdown(f"📏 Size: {result['file_size']} bytes")
            
            # Predictions
            st.markdown("**🎯 Predictions:**")
            
            # Sort predictions (mango_tree first, then by probability)
            sorted_predictions = sorted(
                result['predictions'], 
                key=lambda x: (
                    1 if x['className'].lower() == 'mango_tree' else 
                    2 if x['className'].lower() == 'not_mango_tree' else 3,
                    -x['probability']
                )
            )
            
            for pred in sorted_predictions:
                class_name = pred['className']
                probability = pred['probability']
                percentage = probability * 100
                
                # Display with colored progress bar
                st.markdown(f"**{class_name}:** {percentage:.1f}%")
                st.progress(probability)
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")

# Run the app
if __name__ == "__main__":
    main()
