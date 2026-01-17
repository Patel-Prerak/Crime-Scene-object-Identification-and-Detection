import streamlit as st
import cv2
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import os
import io
from PIL import Image
import numpy as np
import base64
import streamlit.components.v1 as components
from vr_utils import create_aframe_scene
from depth_utils import estimate_depth

# Page configuration
st.set_page_config(
    page_title="Crime Scene Evidence Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Light Theme with Crime Scene Tape
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Root Variables */
    :root {
        --primary-light: #ffffff;
        --secondary-light: #faf5ff;
        --accent-purple: #a855f7;
        --accent-pink: #ec4899;
        --accent-blue: #3b82f6;
        --accent-yellow: #fbbf24;
        --accent-red: #dc2626;
        --text-primary: #1f2937;
        --text-secondary: #6b7280;
        --border-color: #e5e7eb;
    }
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main Background - Pure White - Force on all elements */
    .main, .stApp, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background: #ffffff !important;
    }
    
    .block-container {
        background: #ffffff !important;
        padding-top: 1rem !important;
    }
    
    body {
        background: #ffffff !important;
    }
    
    /* Professional Header - Compact */
    .header-container {
        background: linear-gradient(135deg, #a855f7 0%, #ec4899 100%);
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
        box-shadow: 
            0 8px 24px rgba(168, 85, 247, 0.2),
            0 0 0 1px rgba(255, 255, 255, 0.1) inset;
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            45deg,
            transparent 30%,
            rgba(255, 255, 255, 0.1) 50%,
            transparent 70%
        );
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .header-title {
        font-family: 'Orbitron', sans-serif;
        color: #ffffff;
        font-size: 2rem;
        font-weight: 900;
        text-align: center;
        margin: 0;
        text-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        letter-spacing: 2px;
        position: relative;
        z-index: 1;
        text-transform: uppercase;
    }
    
    .header-subtitle {
        color: rgba(255, 255, 255, 0.95);
        font-size: 0.95rem;
        text-align: center;
        margin-top: 0.4rem;
        font-weight: 500;
        letter-spacing: 1.5px;
        position: relative;
        z-index: 1;
    }
    
    .header-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        padding: 0.3rem 1rem;
        border-radius: 50px;
        margin: 0.15rem;
        font-size: 0.75rem;
        color: white;
        font-weight: 600;
        letter-spacing: 0.5px;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Premium Card Design - Light Purple/Pink - Reduced Height */
    .info-card {
        background: linear-gradient(135deg, #faf5ff 0%, #fce7f3 100%);
        border: 2px solid #e9d5ff;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
        box-shadow: 0 2px 12px rgba(168, 85, 247, 0.08);
    }
    
    .info-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 3px;
        height: 100%;
        background: linear-gradient(180deg, #a855f7 0%, #ec4899 100%);
    }
    
    .info-card:hover {
        transform: translateY(-3px);
        border-color: #d8b4fe;
        box-shadow: 0 6px 20px rgba(168, 85, 247, 0.15);
    }
    
    .feature-card {
        background: linear-gradient(135deg, #faf5ff 0%, #fce7f3 100%);
        border: 1px solid #e9d5ff;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        transition: all 0.3s ease;
        box-shadow: 0 1px 6px rgba(168, 85, 247, 0.06);
    }
    
    .feature-card:hover {
        transform: translateX(3px);
        border-color: #d8b4fe;
        box-shadow: 0 3px 12px rgba(168, 85, 247, 0.12);
    }
    
    .feature-card h4 {
        color: #a855f7;
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        font-size: 1rem;
        margin-top: 0;
        margin-bottom: 0.5rem;
        letter-spacing: 1px;
    }
    
    .feature-card ul {
        list-style: none;
        padding-left: 0;
        margin: 0;
    }
    
    .feature-card li {
        color: #6b7280;
        line-height: 1.6;
        font-size: 0.9rem;
    }
    
    .feature-card li::before {
        content: '▹ ';
        color: #a855f7;
        font-weight: bold;
        margin-right: 0.5rem;
    }
    
    .feature-card p {
        margin: 0;
        line-height: 1.6;
    }
    
    /* Professional Stat Boxes - Light Purple - Compact */
    .stat-box {
        background: linear-gradient(135deg, #faf5ff 0%, #fce7f3 100%);
        border: 2px solid #d8b4fe;
        border-radius: 12px;
        padding: 0.8rem 0.8rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(168, 85, 247, 0.12);
    }
    
    .stat-box::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(168, 85, 247, 0.08) 0%, transparent 70%);
        animation: rotate 10s linear infinite;
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .stat-box:hover {
        transform: translateY(-5px) scale(1.03);
        border-color: #c084fc;
        box-shadow: 0 8px 24px rgba(168, 85, 247, 0.2);
    }
    
    .stat-icon {
        font-size: 1.5rem;
        margin-bottom: 0.2rem;
        filter: drop-shadow(0 2px 4px rgba(168, 85, 247, 0.3));
        position: relative;
        z-index: 1;
    }
    
    .stat-number {
        font-family: 'Orbitron', sans-serif;
        font-size: 2rem;
        font-weight: 900;
        color: #a855f7;
        margin: 0;
        position: relative;
        z-index: 1;
    }
    
    .stat-label {
        font-size: 0.75rem;
        color: #6b7280;
        margin-top: 0.2rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        position: relative;
        z-index: 1;
    }
    
    /* Premium Button Design - Purple Theme */
    .stButton > button {
        background: linear-gradient(135deg, #a855f7 0%, #ec4899 100%);
        color: white;
        border: none;
        padding: 1.2rem 3rem;
        border-radius: 12px;
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        transition: all 0.3s ease;
        box-shadow: 
            0 8px 24px rgba(168, 85, 247, 0.3),
            0 0 0 1px rgba(255, 255, 255, 0.1) inset;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 
            0 12px 32px rgba(168, 85, 247, 0.4),
            0 0 0 1px rgba(255, 255, 255, 0.2) inset;
    }
    
    /* Evidence Badges */
    .evidence-badge {
        display: inline-block;
        padding: 0.7rem 1.4rem;
        border-radius: 25px;
        font-size: 0.95rem;
        font-weight: 700;
        margin: 0.3rem;
        transition: all 0.3s ease;
        letter-spacing: 1px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .evidence-badge::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        transition: left 0.5s;
    }
    
    .evidence-badge:hover::before {
        left: 100%;
    }
    
    .evidence-badge:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
    }
    
    .badge-weapon {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        color: white;
        border: 2px solid #b91c1c;
        animation: criticalPulse 2s ease-in-out infinite;
    }
    
    @keyframes criticalPulse {
        0%, 100% { 
            box-shadow: 0 2px 8px rgba(220, 38, 38, 0.3);
        }
        50% { 
            box-shadow: 0 4px 16px rgba(220, 38, 38, 0.5);
        }
    }
    
    .badge-person {
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
        color: white;
        border: 2px solid #1e40af;
    }
    
    .badge-digital {
        background: linear-gradient(135deg, #7c3aed 0%, #8b5cf6 100%);
        color: white;
        border: 2px solid #6d28d9;
    }
    
    .badge-general {
        background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
        color: white;
        border: 2px solid #059669;
    }
    
    /* Results Container - Light Purple - Reduced Height */
    .results-container {
        background: linear-gradient(135deg, #faf5ff 0%, #fce7f3 100%);
        border: 2px solid #e9d5ff;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-top: 1.5rem;
        box-shadow: 0 4px 16px rgba(168, 85, 247, 0.1);
    }
    
    .result-header {
        font-family: 'Orbitron', sans-serif;
        color: #a855f7;
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 1rem;
        padding-bottom: 0.8rem;
        border-bottom: none !important;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    
    /* Image Container */
    .image-container {
        border-radius: 12px;
        overflow: hidden;
        border: 2px solid #e9d5ff;
        box-shadow: 0 4px 16px rgba(168, 85, 247, 0.1);
        transition: all 0.3s ease;
    }
    
    .image-container:hover {
        transform: scale(1.02);
        border-color: #d8b4fe;
        box-shadow: 0 6px 24px rgba(168, 85, 247, 0.2);
    }
    
    /* Enhanced Metrics - Equal Size Cards */
    .stMetric {
        background: linear-gradient(135deg, #faf5ff 0%, #fce7f3 100%);
        border: 1px solid #e9d5ff;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(168, 85, 247, 0.08);
        transition: all 0.3s ease;
        height: 100%;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .stMetric:hover {
        transform: translateY(-3px);
        border-color: #d8b4fe;
        box-shadow: 0 6px 20px rgba(168, 85, 247, 0.15);
    }
    
    .stMetric label {
        color: #6b7280 !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        font-size: 0.85rem !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #a855f7 !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
    }
    
    .stMetric [data-testid="stMetricDelta"] {
        font-size: 0.8rem !important;
    }
    
    /* Make "Critical" delta text red */
    .stMetric [data-testid="stMetricDelta"]:has(svg[fill="red"]),
    .stMetric [data-testid="stMetricDelta"] svg[fill="red"] {
        color: #dc2626 !important;
    }
    
    /* Target delta containing "Critical" */
    .stMetric div[data-testid="stMetricDelta"] {
        color: inherit;
    }
    
    .stMetric div[data-testid="stMetricDelta"]:contains("Critical") {
        color: #dc2626 !important;
        font-weight: 700 !important;
    }
    
    /* Negative delta (red) */
    .stMetric [data-testid="stMetricDelta"] svg[data-testid="stMetricDeltaIcon-Down"],
    .stMetric [data-testid="stMetricDelta"]:has(svg[data-testid="stMetricDeltaIcon-Down"]) {
        color: #dc2626 !important;
    }
    
    /* Add spacing after metrics row */
    .stMetric {
        margin-bottom: 1.5rem;
    }
    
    /* Force equal height for metric containers */
    [data-testid="metric-container"] {
        height: 100%;
    }
    
    [data-testid="stVerticalBlock"] > div:has(.stMetric) {
        height: 100%;
    }
    
    /* Make columns equal height */
    .row-widget.stHorizontalBlock {
        display: flex;
        align-items: stretch;
    }
    
    .row-widget.stHorizontalBlock > div {
        display: flex;
        flex-direction: column;
    }
    
    /* Download Buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
        color: white;
        border: none;
        padding: 0.9rem 2rem;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
    }
    
    /* File Uploader - White Text and Border */
    .stFileUploader {
        background: linear-gradient(135deg, #faf5ff 0%, #fce7f3 100%);
        border: 2px dashed #d8b4fe;
        border-radius: 12px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: #c084fc;
        background: linear-gradient(135deg, #f5f3ff 0%, #fae8ff 100%);
        box-shadow: 0 4px 16px rgba(168, 85, 247, 0.15);
    }
    
    /* File Uploader Button - White Text and Border - Multiple Selectors */
    .stFileUploader label button,
    .stFileUploader button,
    .stFileUploader section button,
    [data-testid="stFileUploader"] button,
    [data-testid="stFileUploadDropzone"] button {
        color: black !important;
        background: linear-gradient(135deg, #a855f7 0%, #ec4899 100%) !important;
        border: 2px solid white !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stFileUploader label button:hover,
    .stFileUploader button:hover,
    .stFileUploader section button:hover,
    [data-testid="stFileUploader"] button:hover,
    [data-testid="stFileUploadDropzone"] button:hover {
        background: linear-gradient(135deg, #9333ea 0%, #db2777 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(168, 85, 247, 0.3) !important;
    }
    
    /* Additional specificity for button text */
    .stFileUploader button span,
    [data-testid="stFileUploader"] button span {
        color: white !important
    }

    /* 1. Uploaded File List - Make Text Black (Visible on White Background) */
    [data-testid="stUploadedFile"],
    [data-testid="stUploadedFile"] div,
    [data-testid="stUploadedFile"] span,
    [data-testid="stUploadedFile"] small,
    [data-testid="stUploadedFile"] svg {
        color: black !important;
    }

    /* 2. Dropzone Instructions - Make Text White (Visible on Dark Background) */
    [data-testid="stFileUploadDropzone"] div,
    [data-testid="stFileUploadDropzone"] span,
    [data-testid="stFileUploadDropzone"] small,
    [data-testid="stFileUploadDropzone"] p,
    [data-testid="stFileUploadDropzone"] svg {
        color: #e5e7eb !important; /* Light Grey/White */
        fill: #e5e7eb !important;
    }

    /* Exception for the button text itself, keep it white */
    [data-testid="stFileUploader"] button,
    [data-testid="stFileUploader"] button span {
        color: white !important;
    }
    
    /* Sidebar - White with Purple Accents */
    [data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 2px solid #e9d5ff;
    }

    /* Force ANY button in the header or sidebar nav to be visible and purple */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarNav"] > button,
    [data-testid="collapsedControl"],
    header button,
    [data-testid="stHeader"] button {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        color: white !important;
        fill: white !important;
        background-color: #a855f7 !important;
        border: 1px solid #e9d5ff !important;
        border-radius: 50% !important;
        box-shadow: 0 4px 12px rgba(168, 85, 247, 0.3) !important;
        transition: all 0.3s ease !important;
        width: 2.5rem !important;
        height: 2.5rem !important;
        z-index: 999999 !important;
        margin: 4px !important; /* Ensure spacing if multiple exist */
    }

    /* Target SVGs inside these buttons */
    header button svg,
    [data-testid="stHeader"] button svg,
    [data-testid="stSidebarCollapseButton"] svg {
        fill: white !important;
        color: white !important;
        stroke: white !important;
    }
    
    /* Hover effects */
    header button:hover,
    [data-testid="stHeader"] button:hover,
    [data-testid="stSidebarCollapseButton"]:hover {
        background-color: #9333ea !important;
        transform: scale(1.1);
        box-shadow: 0 8px 20px rgba(168, 85, 247, 0.5) !important;
    }
    
    [data-testid="stSidebar"] h3 {
        color: #a855f7;
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        letter-spacing: 2px;
    }
    
    /* DataFrames */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
    }
    
    /* Progress Bar - Purple */
    .stProgress > div > div {
        background: linear-gradient(90deg, #a855f7 0%, #ec4899 100%);
    }
    
    /* Spinner - Purple */
    .stSpinner > div {
        border-top-color: #a855f7 !important;
    }
    
    /* Alert Boxes */
    .stAlert {
        background: white;
        border-left: 4px solid #a855f7;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Text Colors */
    h1, h2, h3, h4, h5, h6 {
        color: #1f2937 !important;
    }
    
    p, li, span {
        color: #4b5563;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f3f4f6;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #2563eb 0%, #3b82f6 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #1e40af 0%, #2563eb 100%);
    }
    
    /* Success Message */
    .success-message {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border: 2px solid #10b981;
        color: #065f46;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.2);
        animation: slideIn 0.5s ease;
    }
    .stFileUploaderFileName {
    color: black !important;
    }
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* VR Mode Expander Styling - Aggressive Override */
    [data-testid="stExpander"] details > summary {
        background-color: #e0f2fe !important; /* Light Blue */
        color: #0c4a6e !important; /* Dark Blue Text */
        border: 1px solid #bae6fd !important;
        border-radius: 8px !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 600 !important;
        transition: background-color 0.3s ease !important;
    }
    
    [data-testid="stExpander"] details > summary:hover {
        background-color: #bae6fd !important; /* Slightly Darker Blue on Hover */
        color: #0c4a6e !important;
    }
    
    /* Force all text/icons inside the header to be Dark Blue */
    [data-testid="stExpander"] details > summary *,
    [data-testid="stExpander"] details > summary p,
    [data-testid="stExpander"] details > summary span,
    [data-testid="stExpander"] details > summary svg {
        color: #0c4a6e !important;
        fill: #0c4a6e !important;
        stroke: #0c4a6e !important;
    }
    
    .streamlit-expanderContent {
        border: 1px solid #e0f2fe !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
    }
    </style>
""", unsafe_allow_html=True)


class EnsembleEvidenceDetector:
    def __init__(self, standard_weights='yolov8l.pt', custom_weights='best.pt'):
        """Initialize BOTH models to cover all evidence types."""
        self.model_standard = YOLO(standard_weights)
        
        if os.path.exists(custom_weights):
            self.model_custom = YOLO(custom_weights)
        else:
            self.model_custom = None
        
        self.VISUAL_CUTOFF = 0.30
        
        # Standard Model Targets (COCO IDs)
        self.std_classes = {
            0: "Person", 24: "Backpack", 26: "Handbag",
            28: "Suitcase", 39: "Bottle", 40: "Wine Glass",
            41: "Cup", 43: "Knife", 67: "Cell Phone",
            76: "Scissors", 73: "Laptop"
        }
        
        # Custom Model Targets
        self.cust_classes = {
            0: "Gun",
            1: "Blood Stain"
        }
        
        # Color Palette
        self.colors = {
            "biohazard": (0, 0, 139),
            "weapon_gun": (0, 0, 255),
            "weapon_knife": (0, 69, 255),
            "person": (255, 255, 0),
            "digital": (255, 0, 0),
            "general": (0, 255, 255),
            "text": (255, 255, 255),
            "bg_label": (50, 50, 50)
        }
    
    def analyze_image(self, img):
        """Analyze a single image and return results."""
        img_h, img_w = img.shape[:2]
        master_log = []
        
        # PASS 1: STANDARD MODEL
        res_std = self.model_standard.predict(img, conf=0.001, iou=0.5, verbose=False)[0]
        for box in res_std.boxes:
            cls_id = int(box.cls[0])
            if cls_id in self.std_classes:
                master_log.append({
                    "Source": "Standard_Model",
                    "Label": self.std_classes[cls_id],
                    "Conf": float(box.conf[0]),
                    "Box": box.xyxy[0].tolist()
                })
        
        # PASS 2: CUSTOM MODEL (Guns/Blood)
        if self.model_custom:
            res_cust = self.model_custom.predict(img, conf=0.001, iou=0.5, verbose=False)[0]
            for box in res_cust.boxes:
                cls_id = int(box.cls[0])
                if cls_id in self.cust_classes:
                    master_log.append({
                        "Source": "Custom_Model",
                        "Label": self.cust_classes[cls_id],
                        "Conf": float(box.conf[0]),
                        "Box": box.xyxy[0].tolist()
                    })
        
        # PASS 3: PROCESSING & VISUALIZATION
        annotated_img = img.copy()
        csv_data = []
        
        for item in master_log:
            label = item['Label']
            conf = item['Conf']
            x1, y1, x2, y2 = map(int, item['Box'])
            
            # Clamp coordinates
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w, x2)
            y2 = min(img_h, y2)
            
            # Add to CSV Data
            csv_data.append({
                "Timestamp": datetime.now().isoformat(),
                "Model_Source": item['Source'],
                "Evidence_Type": label,
                "Confidence_Score": conf,
                "Confidence_Text": f"{conf:.2%}",
                "Visualized": "YES" if conf > self.VISUAL_CUTOFF else "NO",
                "Coords": [x1, y1, x2, y2]
            })
            
            # Draw Visuals (High Confidence Only)
            if conf > self.VISUAL_CUTOFF:
                # Color Selection
                if "Blood" in label:
                    c = self.colors["biohazard"]
                elif "Gun" in label:
                    c = self.colors["weapon_gun"]
                elif "Knife" in label:
                    c = self.colors["weapon_knife"]
                elif "Person" in label:
                    c = self.colors["person"]
                elif "Phone" in label or "Laptop" in label:
                    c = self.colors["digital"]
                else:
                    c = self.colors["general"]
                
                # Draw Box
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), c, 3)
                
                # Label
                lbl = f"{label} {conf:.0%}"
                (text_w, text_h), baseline = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                
                # Default: Draw ABOVE the box
                bg_x1 = x1
                bg_y1 = y1 - text_h - 10
                bg_x2 = x1 + text_w + 10
                bg_y2 = y1
                text_x = x1 + 5
                text_y = y1 - 5
                
                # Boundary checks
                if y1 - text_h - 10 < 0:
                    bg_y1 = y1
                    bg_y2 = y1 + text_h + 10
                    text_y = y1 + text_h + 5
                
                if x1 + text_w + 10 > img_w:
                    shift_amount = (x1 + text_w + 10) - img_w
                    bg_x1 -= shift_amount
                    bg_x2 -= shift_amount
                    text_x -= shift_amount
                
                # Draw Label Background
                cv2.rectangle(annotated_img, (bg_x1, bg_y1), (bg_x2, bg_y2), self.colors["bg_label"], -1)
                
                # Draw Text
                cv2.putText(annotated_img, lbl, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors["text"], 2)
        
        return annotated_img, csv_data


def main():
    # Professional Header
    st.markdown("""
        <div class="header-container">
            <h1 class="header-title">🔍 CRIME SCENE ANALYZER</h1>
            <p class="header-subtitle">AI-POWERED FORENSIC EVIDENCE DETECTION SYSTEM</p>
            <div style="text-align: center; margin-top: 1rem;">
                <span class="header-badge">⚡ REAL-TIME ANALYSIS</span>
                <span class="header-badge">🔒 SECURE PROCESSING</span>
                <span class="header-badge">🎯 DUAL AI MODELS</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📋 SYSTEM INFORMATION")
        st.markdown("""
            <div class="feature-card">
                <h4>🎯 DETECTION CAPABILITIES</h4>
                <ul style="font-size: 0.95rem;">
                    <li>Persons & Suspects</li>
                    <li>Weapons (Guns, Knives)</li>
                    <li>Blood Stains</li>
                    <li>Digital Evidence</li>
                    <li>Personal Items</li>
                    <li>Containers & Objects</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="feature-card">
                <h4>⚙️ AI CONFIGURATION</h4>
                <p style="font-size: 0.9rem; line-height: 1.8;">
                    <strong style="color: #2563eb;">Standard Model:</strong> YOLOv8 Large<br>
                    <strong style="color: #2563eb;">Custom Model:</strong> Forensic Specialist<br>
                    <strong style="color: #2563eb;">Threshold:</strong> 30% Confidence
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
            <div style="text-align: center; color: #6b7280; font-size: 0.85rem; line-height: 2;">
                <p>🔒 SECURE EVIDENCE PROCESSING</p>
                <p>⚡ REAL-TIME ANALYSIS</p>
                <p>📊 DETAILED REPORTING</p>
                <p>🎯 HIGH ACCURACY</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Main Stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="stat-box">
                <div class="stat-icon">🤖</div>
                <p class="stat-number">2</p>
                <p class="stat-label">AI Models</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="stat-box">
                <div class="stat-icon">🎯</div>
                <p class="stat-number">11+</p>
                <p class="stat-label">Evidence Types</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="stat-box">
                <div class="stat-icon">⚡</div>
                <p class="stat-number">30%</p>
                <p class="stat-label">Min Confidence</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Add spacing between sections
    st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
    
    # Upload Section
    st.markdown("""
        <div class="info-card">
            <h3 style="color: #a855f7; margin-top: 0; font-family: 'Orbitron', sans-serif; letter-spacing: 2px;">📤 UPLOAD CRIME SCENE IMAGE</h3>
            <p style="color: #6b7280; margin-bottom: 0; line-height: 1.6;">
                Upload a crime scene photograph for AI-powered evidence detection and analysis.
                Supported formats: <strong style="color: #a855f7;">JPG, JPEG, PNG</strong>
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a crime scene image for analysis",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        # Display original image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
                <div class="results-container">
                    <h3 class="result-header">📸 ORIGINAL IMAGE</h3>
                </div>
            """, unsafe_allow_html=True)
            image = Image.open(uploaded_file)
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Process button
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔍 ANALYZE EVIDENCE", use_container_width=True):
            with st.spinner("🔄 PROCESSING IMAGE WITH AI MODELS..."):
                # Convert PIL to CV2
                img_array = np.array(image)
                img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                # Initialize detector
                detector = EnsembleEvidenceDetector(
                    standard_weights='yolov8l.pt',
                    custom_weights='Custom_Model/weights/best.pt'
                )
                
                # Analyze
                annotated_img, csv_data = detector.analyze_image(img_cv2)
                
                # Convert back to RGB for display
                annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                
                # Store in session state
                st.session_state['annotated_img'] = annotated_img_rgb
                st.session_state['csv_data'] = csv_data
                
                # --- VR GENERATION ---
                # Prepare detections for VR
                vr_detections = []
                for item in csv_data:
                    if item['Visualized'] == 'YES':
                        vr_detections.append({
                            'Label': item['Evidence_Type'],
                            'Conf': item['Confidence_Score'],
                            'Box': item['Coords']
                        })
                
                # Convert original image to Base64
                if not os.path.exists("generated_vr"):
                    os.makedirs("generated_vr")
                
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_b64 = base64.b64encode(buffered.getvalue()).decode()
                img_src = f"data:image/png;base64,{img_b64}"
                
                # --- DEPTH ESTIMATION ---
                st.info("Generating 3D Detail Map (This adds depth)...")
                depth_map_pil, depth_array = estimate_depth(image)
                
                # Convert Depth to Base64
                buffered_depth = io.BytesIO()
                depth_map_pil.save(buffered_depth, format="PNG")
                depth_b64 = base64.b64encode(buffered_depth.getvalue()).decode()
                depth_src = f"data:image/png;base64,{depth_b64}"
                # ------------------------
                
                # Generate VR HTML
                img_w, img_h = image.size
                vr_html_path = "generated_vr/index.html"
                
                # Pass depth data
                create_aframe_scene(vr_detections, img_src, depth_src, depth_array, img_w, img_h, vr_html_path)
                
                # Read back the HTML 
                with open(vr_html_path, 'r', encoding='utf-8') as f:
                    st.session_state['vr_html'] = f.read()
                # ---------------------
                
                # Success message
                st.markdown("""
                    <div class="success-message">
                        ✅ ANALYSIS COMPLETE! Evidence detected and classified.
                    </div>
                """, unsafe_allow_html=True)
        
        # Display results if available
        if 'annotated_img' in st.session_state and 'csv_data' in st.session_state:
            with col2:
                st.markdown("""
                    <div class="results-container">
                        <h3 class="result-header">🎯 ANALYSIS RESULTS</h3>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(st.session_state['annotated_img'], use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # --- VR DISPLAY SECTION ---
            if 'vr_html' in st.session_state:
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("🕶️ ENTER VR MODE (IMMERSIVE EVIDENCE VIEW)", expanded=True):
                     st.markdown("""
                        <div style="background-color: #f3f4f6; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #a855f7;">
                            <strong style="color: #000000;">Instructions:</strong> 
                            <ul style="margin-top: 0.5rem; margin-bottom: 0;">
                                <li>Use <strong>WASD</strong> keys to move (on PC)</li>
                                <li><strong>Click & Drag</strong> to look around</li>
                                <li>On Mobile: Move phone to look around</li>
                                <li>Click 'VR' button in bottom right for Fullscreen VR</li>
                            </ul>
                        </div>
                     """, unsafe_allow_html=True)
                     components.html(st.session_state['vr_html'], height=500, scrolling=False)
            # --------------------------
            
            # Evidence summary
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
                <div class="results-container">
                    <h3 class="result-header">📊 EVIDENCE SUMMARY</h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Add spacing between header and metrics
            st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
            
            if st.session_state['csv_data']:
                df = pd.DataFrame(st.session_state['csv_data'])
                df = df.sort_values(by="Confidence_Score", ascending=False)
                
                # Filter high confidence detections
                high_conf_df = df[df['Confidence_Score'] > 0.30]
                
                if not high_conf_df.empty:
                    # Statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        total_evidence = len(high_conf_df)
                        st.metric("TOTAL EVIDENCE", total_evidence, delta="High Confidence")
                    
                    with col2:
                        weapons = len(high_conf_df[high_conf_df['Evidence_Type'].str.contains('Gun|Knife', case=False, na=False)])
                        st.metric("WEAPONS DETECTED", weapons, delta="⚠️ Critical" if weapons > 0 else "✅ None", delta_color="inverse" if weapons > 0 else "normal")
                    
                    with col3:
                        persons = len(high_conf_df[high_conf_df['Evidence_Type'] == 'Person'])
                        st.metric("PERSONS IDENTIFIED", persons)
                    
                    with col4:
                        avg_conf = high_conf_df['Confidence_Score'].mean()
                        st.metric("AVG CONFIDENCE", f"{avg_conf:.1%}")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Evidence badges
                    st.markdown("""
                        <h4 style="color: #2563eb; font-family: 'Orbitron', sans-serif; letter-spacing: 2px;">
                            🏷️ DETECTED EVIDENCE TYPES
                        </h4>
                    """, unsafe_allow_html=True)
                    
                    evidence_html = ""
                    for _, row in high_conf_df.iterrows():
                        label = row['Evidence_Type']
                        conf = row['Confidence_Text']
                        
                        if 'Gun' in label or 'Knife' in label:
                            badge_class = "badge-weapon"
                        elif 'Person' in label:
                            badge_class = "badge-person"
                        elif 'Phone' in label or 'Laptop' in label:
                            badge_class = "badge-digital"
                        else:
                            badge_class = "badge-general"
                        
                        evidence_html += f'<span class="evidence-badge {badge_class}">{label} ({conf})</span>'
                    
                    st.markdown(evidence_html, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Detailed table
                    st.markdown("""
                        <h4 style="color: #2563eb; font-family: 'Orbitron', sans-serif; letter-spacing: 2px;">
                            📋 DETAILED EVIDENCE REPORT
                        </h4>
                    """, unsafe_allow_html=True)
                    
                    display_df = high_conf_df[['Evidence_Type', 'Confidence_Text', 'Model_Source', 'Visualized']].copy()
                    display_df.columns = ['Evidence Type', 'Confidence', 'Detection Model', 'Visualized']
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                    
                    # Download options
                    st.markdown("<br>", unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Download CSV
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 DOWNLOAD FULL REPORT (CSV)",
                            data=csv,
                            file_name=f"evidence_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        # Download annotated image using BytesIO
                        img_pil = Image.fromarray(st.session_state['annotated_img'])
                        buf = io.BytesIO()
                        img_pil.save(buf, format='PNG')
                        buf.seek(0)
                        st.download_button(
                            label="🖼️ DOWNLOAD ANNOTATED IMAGE",
                            data=buf,
                            file_name=f"annotated_scene_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            mime="image/png",
                            use_container_width=True
                        )
                else:
                    st.info("ℹ️ No high-confidence evidence detected in this image.")
            else:
                st.info("ℹ️ No evidence detected in this image.")
    else:
        # Instructions when no file is uploaded
        st.markdown("""
            <div class="info-card" style="margin-top: 2rem;">
                <h3 style="color: #2563eb; font-family: 'Orbitron', sans-serif; letter-spacing: 2px;">📖 HOW TO USE</h3>
                <ol style="color: #4b5563; line-height: 2; font-size: 1.05rem;">
                    <li><strong style="color: #2563eb;">Upload Image:</strong> Click the upload button above and select a crime scene photograph</li>
                    <li><strong style="color: #2563eb;">Analyze:</strong> Click the "ANALYZE EVIDENCE" button to process the image</li>
                    <li><strong style="color: #2563eb;">Review Results:</strong> View detected evidence with confidence scores and visual annotations</li>
                    <li><strong style="color: #2563eb;">Download Reports:</strong> Export detailed CSV reports and annotated images</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="info-card">
                <h3 style="color: #2563eb; font-family: 'Orbitron', sans-serif; letter-spacing: 2px;">⚠️ IMPORTANT NOTES</h3>
                <ul style="color: #4b5563; line-height: 2; font-size: 1.05rem;">
                    <li>This system uses advanced AI models for evidence detection</li>
                    <li>Results should be verified by forensic professionals</li>
                    <li>Minimum confidence threshold is set to 30% for visualization</li>
                    <li>All detections are logged regardless of confidence level</li>
                    <li>Weapon detections trigger critical alerts with pulsing indicators</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
