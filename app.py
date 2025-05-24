import streamlit as st
import numpy as np
import cv2
from detect import detect_objects, extract_license_plates, class_names

st.title("🪖 Motorcycle Helmet & Plate Detector")