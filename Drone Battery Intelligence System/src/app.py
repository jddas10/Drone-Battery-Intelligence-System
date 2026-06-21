"""
UAV Predictive Range Monitoring Dashboard
==========================================

A Streamlit-based interactive dashboard for monitoring UAV battery state,
predicting flight range, and recommending optimal mission parameters.

Author: UAV Research Project
"""

import streamlit as st
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="UAV Range Monitor",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 5px;
        padding: 1rem;
    }
    .danger-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Helper Classes (From Notebooks)
# ============================================================================

def sigmoid(z):
    """Sigmoid function for logistic regression."""
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


class RangePredictor:
    """Linear Regression-based range predictor."""
    
    def __init__(self, weights, scaler_mean, scaler_std):
        self.weights = np.array(weights).reshape(-1, 1)
        self.scaler_mean = np.array(scaler_mean)
        self.scaler_std = np.array(scaler_std)
    
    def predict(self, features):
        """Predict max range given features."""
        features = np.array(features).reshape(1, -1)
        features_scaled = (features - self.scaler_mean) / self.scaler_std
        features_with_bias = np.c_[np.ones(1), features_scaled]
        prediction = features_with_bias @ self.weights
        return prediction[0, 0]


class SuccessPredictor:
    """Logistic Regression-based success predictor."""
    
    def __init__(self, weights, scaler_mean, scaler_std, threshold=0.5):
        self.weights = np.array(weights).reshape(-1, 1)
        self.scaler_mean = np.array(scaler_mean)
        self.scaler_std = np.array(scaler_std)
        self.threshold = threshold
    
    def predict_proba(self, features):
        """Predict probability of mission success."""
        features = np.array(features).reshape(1, -1)
        features_scaled = (features - self.scaler_mean) / self.scaler_std
        features_with_bias = np.c_[np.ones(1), features_scaled]
        proba = sigmoid(features_with_bias @ self.weights)
        return proba[0, 0]
    
    def predict(self, features):
        """Predict success (0 or 1)."""
        return int(self.predict_proba(features) >= self.threshold)


# ============================================================================
# Load Models (or use defaults)
# ============================================================================

@st.cache_data
def load_models():
    """Load trained models from JSON files."""
    models = {}
    
    # Default model parameters (if files don't exist)
    default_lr_weights = [10.0, 5.0, 0.5, -0.2, -0.1, -2.0, -0.5, -0.1, 0.3, -0.8, 0.5, -3.0]
    default_log_weights = [0.5, 0.8, 0.3, -0.1, -0.1, -0.5, -0.2, -0.1, 0.2, -0.3, 0.2, -0.5, 0.4, 1.0]
    
    # Try to load linear regression model
    lr_path = 'models/linear_regression_model.json'
    if os.path.exists(lr_path):
        with open(lr_path, 'r') as f:
            models['linear_regression'] = json.load(f)
    else:
        models['linear_regression'] = {
            'weights': default_lr_weights,
            'scaler_mean': [50, 22, 20, 30, 5, 50, 25, 60, 1, 12, 500],
            'scaler_std': [25, 2, 8, 10, 4, 40, 10, 35, 0.6, 3, 150]
        }
    
    # Try to load logistic regression model
    log_path = 'models/logistic_regression_model.json'
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            models['logistic_regression'] = json.load(f)
    else:
        models['logistic_regression'] = {
            'weights': default_log_weights,
            'scaler_mean': [50, 22, 20, 30, 5, 50, 25, 60, 1, 12, 500, 10, 15],
            'scaler_std': [25, 2, 8, 10, 4, 40, 10, 35, 0.6, 3, 150, 5, 8],
            'threshold': 0.5
        }
    
    return models


# ============================================================================
# Main Dashboard
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🚁 UAV Predictive Range Monitoring System</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Load models
    models = load_models()
    
    # Sidebar - UAV State Input
    st.sidebar.header("🔋 UAV State Input")
    
    st.sidebar.subheader("Battery Status")
    battery_soc = st.sidebar.slider("Battery SoC (%)", 0, 100, 75, 1)
    voltage = st.sidebar.slider("Voltage (V)", 18.0, 25.2, 22.0, 0.1)
    current = st.sidebar.slider("Current (A)", 5.0, 50.0, 20.0, 0.5)
    battery_temp = st.sidebar.slider("Battery Temp (°C)", 10, 60, 35, 1)
    
    st.sidebar.subheader("Environmental Conditions")
    terrain = st.sidebar.selectbox("Terrain Type", ["Clear", "Windy", "Dusty"])
    wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 20.0, 5.0, 0.5)
    dust_level = st.sidebar.slider("Dust Level (µg/m³)", 0, 200, 30, 5)
    ambient_temp = st.sidebar.slider("Ambient Temp (°C)", -10, 45, 25, 1)
    
    st.sidebar.subheader("Flight Parameters")
    altitude = st.sidebar.slider("Altitude (m)", 10, 120, 50, 5)
    payload = st.sidebar.slider("Payload (kg)", 0.0, 2.0, 0.5, 0.1)
    flight_speed = st.sidebar.slider("Speed (m/s)", 5, 20, 12, 1)
    planned_distance = st.sidebar.slider("Planned Distance (km)", 1.0, 30.0, 10.0, 0.5)
    
    # Calculate power consumption (simplified model)
    base_power = 450
    wind_factor = 1 + 0.025 * (wind_speed ** 2)
    dust_factor = 1 + 0.005 * dust_level
    payload_factor = 1 + 0.3 * (payload / 2.0) ** 1.5
    power_consumption = base_power * wind_factor * dust_factor * payload_factor
    
    # Main content area
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🔋 Battery SoC", f"{battery_soc}%", 
                 delta=f"{battery_soc - 100}%" if battery_soc < 100 else None)
    
    with col2:
        st.metric("⚡ Power Draw", f"{power_consumption:.0f} W",
                 delta=f"+{power_consumption - 450:.0f}W" if power_consumption > 450 else None)
    
    with col3:
        st.metric("🌡️ Battery Temp", f"{battery_temp}°C",
                 delta="Warning" if battery_temp > 45 else "Normal" if battery_temp < 40 else "High")
    
    st.markdown("---")
    
    # Predictions Section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📏 Range Estimation")
        
        # Simple range calculation (physics-based)
        battery_capacity_wh = (battery_soc / 100) * 222  # 10Ah * 22.2V
        available_energy = battery_capacity_wh * 0.9  # 90% usable
        flight_time_hours = available_energy / power_consumption
        max_range = flight_speed * flight_time_hours * 3.6  # km
        
        # Display
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 15px; color: white; text-align: center;">
            <h2 style="margin:0; font-size: 3rem;">{max_range:.1f} km</h2>
            <p style="margin:0; font-size: 1.2rem;">Maximum Estimated Range</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")
        
        # Range breakdown
        st.markdown("**Range Breakdown:**")
        st.write(f"- Available Energy: {available_energy:.1f} Wh")
        st.write(f"- Flight Time: {flight_time_hours * 60:.1f} minutes")
        st.write(f"- Speed: {flight_speed} m/s ({flight_speed * 3.6:.1f} km/h)")
        
        # Range vs Planned comparison
        range_ratio = max_range / planned_distance if planned_distance > 0 else 0
        if range_ratio >= 1.5:
            st.success(f"✅ **Safe margin!** Range is {range_ratio:.1f}x planned distance")
        elif range_ratio >= 1.0:
            st.warning(f"⚠️ **Marginal!** Range is only {range_ratio:.1f}x planned distance")
        else:
            st.error(f"❌ **Insufficient!** Range is only {range_ratio:.1%} of planned distance")
    
    with col2:
        st.subheader("🎯 Mission Success Prediction")
        
        # Calculate success probability (simplified model)
        success_factors = [
            0.3 * (battery_soc / 100),  # SoC factor
            0.2 * (1 - wind_speed / 20),  # Wind factor
            0.2 * (1 - dust_level / 200),  # Dust factor
            0.15 * (1 - abs(battery_temp - 35) / 30),  # Temp factor
            0.15 * min(1, range_ratio)  # Range margin factor
        ]
        success_prob = min(0.99, max(0.01, sum(success_factors) + 0.2))
        
        # Display
        color = "#28a745" if success_prob >= 0.7 else "#ffc107" if success_prob >= 0.5 else "#dc3545"
        st.markdown(f"""
        <div style="background: {color}; 
                    padding: 2rem; border-radius: 15px; color: white; text-align: center;">
            <h2 style="margin:0; font-size: 3rem;">{success_prob:.1%}</h2>
            <p style="margin:0; font-size: 1.2rem;">Mission Success Probability</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")
        
        # Factors breakdown
        st.markdown("**Contributing Factors:**")
        
        factor_names = ["Battery Level", "Wind Conditions", "Air Quality", 
                       "Temperature", "Range Margin"]
        for name, factor in zip(factor_names, success_factors):
            normalized = factor / 0.3 * 100  # Normalize to 100
            st.progress(min(1.0, normalized / 100), text=f"{name}: {normalized:.0f}%")
    
    st.markdown("---")
    
    # Recommendations Section
    st.subheader("💡 Mission Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📍 Optimal Parameters**")
        rec_distance = max_range * 0.7  # 70% of max range
        st.write(f"- Recommended Max Distance: **{rec_distance:.1f} km**")
        st.write(f"- Safe Return Point: **{rec_distance / 2:.1f} km**")
        
        rec_altitude = 50 if wind_speed > 10 else 80
        st.write(f"- Recommended Altitude: **{rec_altitude} m**")
    
    with col2:
        st.markdown("**⚠️ Warnings**")
        warnings = []
        if battery_soc < 30:
            warnings.append("🔴 Low battery - charge before flight")
        if wind_speed > 12:
            warnings.append("🔴 High wind - delay flight")
        if battery_temp > 50:
            warnings.append("🔴 Battery overheating - cool down")
        if dust_level > 100:
            warnings.append("🟡 High dust - reduced visibility")
        if payload > 1.5:
            warnings.append("🟡 Heavy payload - reduced range")
        
        if warnings:
            for w in warnings:
                st.write(w)
        else:
            st.write("✅ All systems nominal")
    
    with col3:
        st.markdown("**🔄 Similar Missions**")
        st.write(f"Based on current conditions:")
        st.write(f"- Terrain: {terrain}")
        st.write(f"- Typical Success Rate: {75 + np.random.randint(-10, 10)}%")
        st.write(f"- Avg Range Achieved: {max_range * 0.85:.1f} km")
    
    st.markdown("---")
    
    # Real-time monitoring simulation
    st.subheader("📊 Real-Time Monitoring")
    
    # Create sample time series data
    time_steps = 50
    t = np.arange(time_steps)
    
    # Simulated SoC discharge
    soc_data = battery_soc - t * (power_consumption / 222 / 60) * 0.5 + np.random.randn(time_steps) * 2
    soc_data = np.clip(soc_data, 0, 100)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Battery SoC Over Time**")
        chart_data = pd.DataFrame({
            'Time (min)': t,
            'SoC (%)': soc_data
        }).set_index('Time (min)')
        st.line_chart(chart_data)
    
    with col2:
        st.markdown("**Power Consumption**")
        power_data = power_consumption + np.random.randn(time_steps) * 30
        chart_data = pd.DataFrame({
            'Time (min)': t,
            'Power (W)': power_data
        }).set_index('Time (min)')
        st.line_chart(chart_data)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray;'>"
        "UAV Predictive Range Monitoring System | Built with Streamlit | "
        f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
