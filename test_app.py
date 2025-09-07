import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st

st.title("Test App")
st.write("If you see this, Streamlit is working!")

# Test TensorFlow
try:
    import tensorflow as tf
    st.success(f"TensorFlow version: {tf.__version__}")
    st.info(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
except Exception as e:
    st.error(f"TensorFlow error: {e}")

# Test other imports
try:
    import torch
    st.success(f"PyTorch version: {torch.__version__}")
except Exception as e:
    st.error(f"PyTorch error: {e}")
