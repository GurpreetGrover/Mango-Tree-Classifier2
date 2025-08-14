import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image, ImageOps
from PIL.ExifTags import TAGS, GPSTAGS, IFD
import requests
import json
from datetime import datetime
import io
import tempfile
import os
import math
from itertools import combinations

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
    .duplicate-card {
        border: 2px solid #ff9800;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background: #fff3e0;
        box-shadow: 0 2px 4px rgba(255,152,0,0.2);
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
    .gps-info {
        background: #e8f5e8;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        font-size: 0.9em;
    }
    .no-gps {
        background: #ffebee;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        font-size: 0.9em;
        color: #d32f2f;
    }
    .duplicate-warning {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if 'classification_results' not in st.session_state:
    st.session_state.classification_results = []
if 'duplicate_pairs' not in st.session_state:
    st.session_state.duplicate_pairs = []
if 'threshold_distance' not in st.session_state:
    st.session_state.threshold_distance = 1.0  # 1 meter default

# Configuration - Update these paths for your SavedModel
SAVEDMODEL_PATH = st.secrets.get("SAVEDMODEL_PATH", "./models/mango_classifier_savedmodel")
# Alternative: You can also use a model from TensorFlow Hub or cloud storage
# SAVEDMODEL_PATH = "gs://your-bucket/models/mango_classifier"
# SAVEDMODEL_PATH = "https://tfhub.dev/your-model/1"

def extract_gps_from_exif(image):
    """
    Extract GPS coordinates from image EXIF data
    Returns: dict with latitude, longitude or None if no GPS data
    """
    try:
        exif_data = image.getexif()
        if not exif_data:
            return None
        
        gps_ifd = exif_data.get_ifd(IFD.GPSInfo)
        if gps_ifd:

            st.write("  ✅ Found GPS IFD")
            # st.write(f"  GPS IFD contents: {dict(gps_ifd)}")
            return parse_gps_ifd(gps_ifd)

        else:
            st.write("  ❌ No GPS IFD found")
    except (ImportError, AttributeError) as e:
        print(f"  ❌ get_ifd() not available: {e}")

def parse_gps_ifd(gps_ifd):
    """Parse GPS data from IFD object"""
    try:
        gps_data = {}
        for gps_tag in gps_ifd:
            gps_tag_name = GPSTAGS.get(gps_tag, gps_tag)
            gps_data[gps_tag_name] = gps_ifd[gps_tag]
            
        return extract_coordinates(gps_data)
    except Exception as e:
        print(f"Error parsing GPS IFD: {e}")
        return None

def extract_coordinates(gps_data):
    """Extract lat/lon from parsed GPS data"""
    try:
        lat = gps_data.get('GPSLatitude')
        lat_ref = gps_data.get('GPSLatitudeRef')
        lon = gps_data.get('GPSLongitude')
        lon_ref = gps_data.get('GPSLongitudeRef')
        
        # print(f"GPS Components found:")
        # print(f"  Latitude: {lat} ({type(lat)})")
        # print(f"  Latitude Ref: {lat_ref}")
        # print(f"  Longitude: {lon} ({type(lon)})")
        # print(f"  Longitude Ref: {lon_ref}")
        
        if lat and lon and lat_ref and lon_ref:
            latitude = convert_dms_to_dd(lat, lat_ref)
            longitude = convert_dms_to_dd(lon, lon_ref)
            
            return {
                'latitude': latitude,
                'longitude': longitude,
                'lat_ref': lat_ref,
                'lon_ref': lon_ref
            }
        else:
            print("❌ Missing GPS components")
            return None
            
    except Exception as e:
        print(f"Error extracting coordinates: {e}")
        return None




        # # Look for GPS info in EXIF data
        # gps_info = None
        # gps_info = exif_data.get(34853)

        # if isinstance(gps_info, dict):
        #     st.write(f"  GPSInfo value is not a dictionary, cannot parse GPS tags directly. Value: {gps_info}")
        #     return None

        # gps_data = {}
        # for tag, gps_value in gps_info.items():
        #     tag_name = GPSTAGS.get(tag, tag)
        #     gps_data[gps_tag_name] = gps_value
            # if tag_name == "GPSInfo":
                # gps_info = value
                # for gps_tag in value:
                #     gps_tag_name = GPSTAGS.get(gps_tag, gps_tag)
                #     gps_data[gps_tag_name] = value[gps_tag]
                # break

        # Extract latitude and longitude
        # lat = gps_data.get('GPSLatitude')
        # lat_ref = gps_data.get('GPSLatitudeRef')
        # lon = gps_data.get('GPSLongitude')
        # lon_ref = gps_data.get('GPSLongitudeRef')
        
        # if lat and lon and lat_ref and lon_ref:
        #     latitude = convert_dms_to_dd(lat, lat_ref)
        #     longitude = convert_dms_to_dd(lon, lon_ref)
            
        #     return {
        #         'latitude': latitude,
        #         'longitude': longitude,
        #         'raw_gps_data': gps_data
        #     }
        
        # return None
        
    # except Exception as e:
    #     st.error(f"Error extracting GPS data: {str(e)}")
    #     return None

def convert_dms_to_dd(dms, ref):
    """
    Convert DMS (Degrees, Minutes, Seconds) to Decimal Degrees
    """
    try:
        if len(dms) >= 3:
            degrees = float(dms[0])
            minutes = float(dms[1]) 
            seconds = float(dms[2])
        elif len(dms) == 2:
            degrees = float(dms[0])
            minutes = float(dms[1])
            seconds = 0.0
        else:
            return None
        
        dd = degrees + minutes/60 + seconds/3600
        
        if ref in ['S', 'W']:
            dd = -dd
        
        return dd
    except:
        return None

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two GPS coordinates using Haversine formula
    Returns distance in meters
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of Earth in meters
    earth_radius = 6371000
    
    # Calculate the distance
    distance = earth_radius * c
    
    return distance

def find_duplicate_pairs(results, threshold_meters):
    """
    Find pairs of images that are within threshold distance of each other
    """
    duplicate_pairs = []
    
    # Get results with GPS data
    gps_results = [r for r in results if r.get('gps_location')]
    
    if len(gps_results) < 2:
        return duplicate_pairs
    
    # Check all combinations of images
    for result1, result2 in combinations(gps_results, 2):
        gps1 = result1['gps_location']
        gps2 = result2['gps_location']
        
        distance = haversine_distance(
            gps1['latitude'], gps1['longitude'],
            gps2['latitude'], gps2['longitude']
        )
        
        if distance <= threshold_meters:
            duplicate_pairs.append({
                'result1': result1,
                'result2': result2,
                'distance': distance,
                'id': f"{result1['id']}_{result2['id']}"
            })
    
    return duplicate_pairs

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
    """Preprocess image for prediction"""
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # # Resize image to model input size (typically 224x224 for Teachable Machine)
    # image = image.resize((224, 224))
    
    # Convert to RGB if not already
    # if image.mode != 'RGB':
    #     image = image.convert('RGB')

    # image = Image.open("/content/papaya-plant-500x500.jpg").convert("RGB")

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)


    image_array = np.asarray(image)


    # Convert to numpy array and normalize
    image_array = np.array(image_array.astype(np.float32)) / 255.0

    data[0] = image_array
    
    # # Add batch dimension
    # image_array = np.expand_dims(image_array, axis=0)
    
    return data

def classify_image_with_savedmodel(image, model_info):
    """Real image classification using TensorFlow SavedModel    """
    try:
        if not model_info or 'model' not in model_info:
            raise ValueError("Model not properly loaded")
        
        model = model_info['model']
        labels = model_info['labels']
        
        # Preprocess the image for the model
        processed_image = preprocess_image(image)

        # Make prediction using the SavedModel
        with st.spinner("🔍 Classifying image..."):
            predictions = model(processed_image)#, verbose=0)

        # Processing output of model classification
        prediction_tensor = predictions['sequential_3']

        # Convert predictions to the expected format
        prediction_results = []

        for index, prob in enumerate(prediction_tensor[0]):
            prediction_results.append({
                'className': labels[index],
                'probability': float(f'{np.float32(prob):.2f}')
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

def main():
    """Main Streamlit application"""
    # Header
    st.markdown('<h1 class="main-header">🥭 Mango Tree Classifier</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 18px;">Upload images to classify mango trees using AI</p>', 
                unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("⚙️ Settings")
        threshold_distance = st.slider(
            "Duplicate Detection Threshold (meters)",
            min_value=0.1,
            max_value=10.0,
            value=st.session_state.threshold_distance,
            step=0.1,
            help="Images closer than this distance will be flagged as potential duplicates"
        )
        st.session_state.threshold_distance = threshold_distance
        
        st.markdown("---")
        st.markdown("**📊 Current Session Stats:**")
        total_images = len(st.session_state.classification_results)
        images_with_gps = len([r for r in st.session_state.classification_results if r.get('gps_location')])
        total_duplicates = len(st.session_state.duplicate_pairs)
        
        st.metric("Total Images", total_images)
        st.metric("Images with GPS", images_with_gps)
        st.metric("Duplicate Pairs", total_duplicates)
   

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
                    
                    # # Display model architecture summary if available
                    # try:
                    #     model_summary = []
                    #     model_info['model'].summary(print_fn=lambda x: model_summary.append(x))
                    #     st.code('\n'.join(model_summary[:10]) + '\n...' if len(model_summary) > 10 else '\n'.join(model_summary))
                    # except:
                    #     st.write("Model architecture details not available")
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
        help="Select one or more image files for classification",
        key=f"uploader_{st.session_state.uploader_key}"
    )
    
    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) uploaded**")
        # Process button
        process_images(uploaded_files)
   
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Duplicate pairs section
    if st.session_state.duplicate_pairs:
        st.markdown("## ⚠️ Potential Duplicate Pairs")
        st.markdown(f"Found **{len(st.session_state.duplicate_pairs)}** potential duplicate pairs within {st.session_state.threshold_distance}m:")
        
        for pair in st.session_state.duplicate_pairs:
            display_duplicate_pair(pair)

    # Results section
    if st.session_state.classification_results:
        st.markdown("## 📋 Classification Results")
        
        # Clear results button
        if st.button("🗑️ Clear All Results"):
            st.session_state.classification_results = []
            st.session_state.duplicate_pairs = []
            st.session_state.uploader_key += 1
            st.rerun()
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
        6. 📍 **Duplicate detection**: Images with GPS data will be checked for proximity
        7. ⚙️ **Adjust threshold**: Use the sidebar to change duplicate detection sensitivity
        
        **GPS Requirements:**
        - Images must contain EXIF GPS metadata
        - Supported formats: JPG, JPEG (PNG typically doesn't contain GPS data)
        - GPS coordinates will be displayed when available
        """
        # **SavedModel Requirements:**
        # - Model should accept images of shape (224, 224, 3)
        # - Input should be normalized to [0, 1] range
        # - Output should be class probabilities
        )

def process_images(uploaded_files):
    """Process multiple uploaded images"""
    if not st.session_state.model:
        st.error("Model not loaded!")
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_files = len(uploaded_files)
    new_results = []
    
    for i, uploaded_file in enumerate(uploaded_files):
        # Update progress
        progress = (i + 1) / total_files
        progress_bar.progress(progress)
        status_text.text(f"Processing {uploaded_file.name}... ({i+1}/{total_files})")
        
        try:
            # Load and preprocess image
            image = Image.open(uploaded_file).convert("RGB")
            
            # Extract GPS data
            gps_location = extract_gps_from_exif(image)

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
                    'file_size': uploaded_file.size,
                    'gps_location': gps_location
                }
                
                new_results.append(result)
                # st.session_state.classification_results.append(result)
        
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
    # Add new results to session state
    st.session_state.classification_results.extend(new_results)

    # Clear duplicate pairs before checking (important!)
    st.session_state.duplicate_pairs = []

    if st.session_state.classification_results:
        st.session_state.duplicate_pairs = find_duplicate_pairs(
            st.session_state.classification_results, 
            st.session_state.threshold_distance
        )

    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    gps_count = len([r for r in st.session_state.classification_results if r['gps_location']])
    duplicate_count = len(st.session_state.duplicate_pairs)

    st.success(f"✅ Successfully processed {total_files} image(s)!")

    if gps_count > 0:
        st.info(f"📍 Found GPS data in {gps_count} image(s)")
    if duplicate_count > 0:
        st.warning(f"⚠️ Found {duplicate_count} potential duplicate pair(s)")

def display_duplicate_pair(pair):
    """Display duplicate pair information"""
    result1 = pair['result1']
    result2 = pair['result2']
    distance = pair['distance']
    
    st.markdown('<div class="duplicate-card">', unsafe_allow_html=True)
    
    st.markdown(f"### 🔄 Duplicate Pair (Distance: {distance:.2f}m)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**📁 {result1['file_name']}**")
        st.image(result1['image'], caption=f"Image 1: {result1['file_name']}", use_container_width=True)
        if result1['gps_location']:
            gps = result1['gps_location']
            st.markdown(f"📍 GPS: {gps['latitude']:.6f}, {gps['longitude']:.6f}")
    
    with col2:
        st.markdown(f"**📁 {result2['file_name']}**")
        st.image(result2['image'], caption=f"Image 2: {result2['file_name']}", use_container_width=True)
        if result2['gps_location']:
            gps = result2['gps_location']
            st.markdown(f"📍 GPS: {gps['latitude']:.6f}, {gps['longitude']:.6f}")
    
    # Action buttons
    st.markdown("**Actions:**")
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        if st.button(f"Keep Both", key=f"keep_both_{pair['id']}"):
            st.info("Both images marked as kept")
    
    with action_col2:
        if st.button(f"Remove Image 1", key=f"remove_1_{pair['id']}"):
            remove_image_from_results(result1['id'])
            st.session_state.duplicate_pairs = [p for p in st.session_state.duplicate_pairs if p['id'] != pair['id']]
            st.success(f"Removed {result1['file_name']}")
            st.rerun()
    
    with action_col3:
        if st.button(f"Remove Image 2", key=f"remove_2_{pair['id']}"):
            remove_image_from_results(result2['id'])
            st.session_state.duplicate_pairs = [p for p in st.session_state.duplicate_pairs if p['id'] != pair['id']]
            st.success(f"Removed {result2['file_name']}")
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")

def remove_image_from_results(image_id):
    """Remove an image from classification results"""
    st.session_state.classification_results = [
        r for r in st.session_state.classification_results if r['id'] != image_id
    ]

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
            
            # GPS info
            if result['gps_location']:
                gps = result['gps_location']
                st.markdown(f'<div class="gps-info">📍 GPS: {gps["latitude"]:.6f}, {gps["longitude"]:.6f}</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown('<div class="no-gps">📍 No GPS data available</div>', 
                           unsafe_allow_html=True)

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
