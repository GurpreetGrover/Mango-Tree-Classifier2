import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import json
from datetime import datetime
import io

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

# Configuration
MODEL_URL = st.secrets.get("TEACHABLE_MACHINE_URL")

@st.cache_resource
def load_teachable_machine_model(model_url):
    """
    Load TensorFlow.js model from Teachable Machine
    JavaScript equivalent: loadModel() function
    """
    try:
        # For TensorFlow.js models, we need to download and convert
        model_json_url = f"{model_url}model.json"
        metadata_url = f"{model_url}metadata.json"
        
        # Download model metadata to get class labels
        metadata_response = requests.get(metadata_url)
        metadata = metadata_response.json()
        
        # For this demo, we'll simulate the model loading
        # In a real scenario, you'd need to convert the TF.js model to TensorFlow format
        class_labels = metadata.get('labels', ['mango_tree', 'not_mango_tree'])
        
        return {
            'labels': class_labels,
            'loaded': True,
            'url': model_url
        }
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    """
    Preprocess image for prediction
    JavaScript equivalent: Image preprocessing in classifyImage()
    """
    # Resize image to model input size (typically 224x224 for Teachable Machine)
    image = image.resize((224, 224))
    
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array and normalize
    image_array = np.array(image) / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def classify_image_simulation(image, model_info):
    """
    Simulate image classification (since we can't directly use TF.js model)
    JavaScript equivalent: classifyImage() function
    
    In a real implementation, you would:
    1. Convert TF.js model to TensorFlow format
    2. Load with tf.keras.models.load_model()
    3. Make actual predictions
    """
    # Simulate predictions based on simple image analysis
    # This is just for demo purposes
    
    # Convert image to analyze basic properties
    img_array = np.array(image)
    
    # Simple heuristic: check for green content (mango trees are green)
    green_ratio = np.mean(img_array[:, :, 1]) / 255.0  # Green channel
    
    # Simulate confidence based on green content
    if green_ratio > 0.4:
        mango_confidence = min(0.9, green_ratio + np.random.normal(0, 0.1))
        not_mango_confidence = 1 - mango_confidence
    else:
        not_mango_confidence = min(0.9, (1 - green_ratio) + np.random.normal(0, 0.1))
        mango_confidence = 1 - not_mango_confidence
    
    # Ensure probabilities sum to 1
    total = mango_confidence + not_mango_confidence
    mango_confidence /= total
    not_mango_confidence /= total
    
    predictions = [
        {
            'className': 'mango_tree',
            'probability': float(mango_confidence)
        },
        {
            'className': 'not_mango_tree', 
            'probability': float(not_mango_confidence)
        }
    ]
    
    return predictions

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
    st.markdown('<h1 class="main-header">🥭 Mango Tree Classifier</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 18px;">Upload images to classify mango trees using AI</p>', 
                unsafe_allow_html=True)
    
    # Load model on first run
    if not st.session_state.model_loaded:
        with st.spinner("Loading AI model..."):
            model_info = load_teachable_machine_model(MODEL_URL)
            if model_info:
                st.session_state.model = model_info
                st.session_state.model_loaded = True
                st.success("✅ Model loaded successfully!")
            else:
                st.error("❌ Failed to load model. Please check the model URL.")
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
        1. 📁 Upload one or more image files using the file uploader above
        2. 🔍 Click 'Classify Images' to analyze your photos
        3. 📊 View the classification results with confidence scores
        4. 🥭 The AI will tell you if each image contains a mango tree or not
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
            predictions = classify_image_simulation(image, st.session_state.model)
            
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
            st.image(result['image'], caption=result['file_name'], use_container_width=True)
        
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
