import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="CarPrice AI - Intelligent Car Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS for professional website design
st.markdown("""
<style>
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container styling */
    .main {
        padding-top: 0rem;
    }
    
    /* Header and Navigation */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 2rem;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 15px 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .site-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: white;
        text-align: center;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .site-subtitle {
        font-size: 1.1rem;
        color: #e8f4f8;
        text-align: center;
        margin: 0.5rem 0 1.5rem 0;
        font-weight: 300;
    }
    
    .nav-container {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 1rem;
    }
    
    .nav-button {
        background: rgba(255, 255, 255, 0.2);
        border: 2px solid rgba(255, 255, 255, 0.3);
        color: white;
        padding: 0.7rem 1.5rem;
        border-radius: 25px;
        text-decoration: none;
        font-weight: 500;
        transition: all 0.3s ease;
        cursor: pointer;
        backdrop-filter: blur(10px);
    }
    
    .nav-button:hover {
        background: rgba(255, 255, 255, 0.3);
        border-color: rgba(255, 255, 255, 0.5);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .nav-button.active {
        background: white;
        color: #667eea;
        border-color: white;
        font-weight: 600;
    }
    
    /* Page content styling */
    .page-header {
        font-size: 2.2rem;
        color: #2c3e50;
        margin-bottom: 1.5rem;
        text-align: center;
        font-weight: 600;
    }
    
    .section-header {
        font-size: 1.8rem;
        color: #34495e;
        margin: 2rem 0 1rem 0;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
        font-weight: 500;
    }
    
    /* Card styling */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #e1e8ed;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(240, 147, 251, 0.3);
        margin: 1rem 0;
    }
    
    .prediction-result {
        font-size: 2.5rem;
        font-weight: bold;
        color: #27ae60;
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 20px;
        border: 3px solid #27ae60;
        margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(39, 174, 96, 0.2);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Feature highlights */
    .feature-highlight {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #ff6b6b;
    }
    
    .stats-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .stat-box {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-top: 4px solid #667eea;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
        margin: 0;
    }
    
    .stat-label {
        color: #7f8c8d;
        font-size: 0.9rem;
        margin: 0.5rem 0 0 0;
    }
</style>
""", unsafe_allow_html=True)

class CarPricePredictorApp:
    def __init__(self):
        self.best_model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_names = None
        self.model_scores = None
        self.df = None
        self.current_page = "Home"
        self.load_models_and_data()
        
    def render_header_and_navigation(self):
        """Render the professional header and navigation bar with functional buttons"""
        # Initialize navigation state
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'Home'
        
        st.markdown("""
        <div class="header-container">
            <h1 class="site-title">🚗 CarPrice </h1>
            <p class="site-subtitle">Intelligent Car Price Prediction using Advanced Machine Learning</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create navigation buttons
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            if st.button("🏠 Home", use_container_width=True, type="primary" if st.session_state.current_page == 'Home' else "secondary"):
                st.session_state.current_page = 'Home'

        with col2:
            if st.button("📋 Available Details", use_container_width=True, type="primary" if st.session_state.current_page == 'Available Details' else "secondary"):
                st.session_state.current_page = 'Available Details'
        
        with col3:
            if st.button("💰 Price Predictor", use_container_width=True, type="primary" if st.session_state.current_page == 'Price Predictor' else "secondary"):
                st.session_state.current_page = 'Price Predictor'
        
        with col5:
            if st.button("📊 Model Analytics", use_container_width=True, type="primary" if st.session_state.current_page == 'Model Analytics' else "secondary"):
                st.session_state.current_page = 'Model Analytics'
        
        with col4:
            if st.button("📊 Visualizations", use_container_width=True, type="primary" if st.session_state.current_page == 'Visualizations' else "secondary"):
                st.session_state.current_page = 'Visualizations'
        
        with col6:
            if st.button("📚 About", use_container_width=True, type="primary" if st.session_state.current_page == 'About' else "secondary"):
                st.session_state.current_page = 'About'
        
        st.markdown("---")
        
        return st.session_state.current_page
        
    def load_models_and_data(self):
        """Load trained models and data"""
        try:
            # Load models
            self.best_model = joblib.load('models/best_model.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            self.label_encoders = joblib.load('models/label_encoders.pkl')
            self.feature_names = joblib.load('models/feature_names.pkl')
            self.model_scores = joblib.load('models/model_scores.pkl')
            
            # Load data
            self.df = pd.read_csv('cleaned_car_data.csv')
            
            return True
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return False
    
    def create_cascading_filters(self):
        """Create cascading filter system for car selection"""
        # Check if data is loaded
        if self.df is None:
            st.error("Data not loaded. Please check if cleaned_car_data.csv exists.")
            return None
        
        st.markdown('<h3 class="section-header">🔍 Select Your Car Specifications</h3>', unsafe_allow_html=True)
        
        # Initialize session state for filters
        if 'selected_brand' not in st.session_state:
            st.session_state.selected_brand = None
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = None
        if 'filter_step' not in st.session_state:
            st.session_state.filter_step = 1
        
        # Step 1: Brand Selection
        st.markdown("#### Step 1: Choose Brand")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            brands = ['Select a brand...'] + sorted(self.df['Brand'].unique().tolist())
            selected_brand = st.selectbox(
                "Select Car Brand",
                brands,
                key="brand_select",
                help="Choose the car manufacturer"
            )
            
            if selected_brand != 'Select a brand...':
                st.session_state.selected_brand = selected_brand
                st.session_state.filter_step = 2
        
        with col2:
            if st.session_state.selected_brand:
                brand_count = len(self.df[self.df['Brand'] == st.session_state.selected_brand])
                st.metric("Available Models", brand_count)
        
        # Step 2: Model Selection (if brand is selected)
        if st.session_state.filter_step >= 2 and st.session_state.selected_brand:
            st.markdown("#### Step 2: Choose Model")
            
            # Filter models based on selected brand
            brand_models = self.df[self.df['Brand'] == st.session_state.selected_brand]['Model'].unique()
            models = ['Select a model...'] + sorted(brand_models.tolist())
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_model = st.selectbox(
                    f"Select {st.session_state.selected_brand} Model",
                    models,
                    key="model_select",
                    help="Choose the specific model"
                )
                
                if selected_model != 'Select a model...':
                    st.session_state.selected_model = selected_model
                    st.session_state.filter_step = 3
            
            with col2:
                if selected_model != 'Select a model...':
                    # Get model info
                    model_info = self.df[
                        (self.df['Brand'] == st.session_state.selected_brand) & 
                        (self.df['Model'] == selected_model)
                    ].iloc[0]
                    st.metric("Price Range", f"₹{model_info['Ex_Showroom_PriceRs.']:,.0f}")
        
        # Step 3: Specifications (if model is selected)
        if st.session_state.filter_step >= 3 and st.session_state.selected_model:
            st.markdown("#### Step 3: Specify Details")
            
            # Get available options for selected brand/model
            filtered_df = self.df[
                (self.df['Brand'] == st.session_state.selected_brand) & 
                (self.df['Model'] == st.session_state.selected_model)
            ]
            
            if len(filtered_df) > 0:
                # Use the first matching record as base, but allow customization
                base_car = filtered_df.iloc[0]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Seating Capacity
                    available_seats = sorted(self.df['Seating_Capacity'].unique())
                    seating_capacity = st.selectbox(
                        "Seating Capacity",
                        available_seats,
                        index=available_seats.index(base_car['Seating_Capacity']) if base_car['Seating_Capacity'] in available_seats else 0
                    )
                    
                    # Fuel Type
                    fuel_types = sorted(self.df['Fuel_Type'].unique())
                    selected_fuel = st.selectbox(
                        "Fuel Type",
                        fuel_types,
                        index=fuel_types.index(base_car['Fuel_Type']) if base_car['Fuel_Type'] in fuel_types else 0
                    )
                    
                    # Body Type
                    body_types = sorted(self.df['Body_Type'].unique())
                    selected_body = st.selectbox(
                        "Body Type",
                        body_types,
                        index=body_types.index(base_car['Body_Type']) if base_car['Body_Type'] in body_types else 0
                    )
                
                with col2:
                    # Transmission
                    transmission_types = sorted(self.df['Type'].unique())
                    selected_transmission = st.selectbox(
                        "Transmission",
                        transmission_types,
                        index=transmission_types.index(base_car['Type']) if base_car['Type'] in transmission_types else 0
                    )
                    
                    # Cylinders
                    cylinders = st.slider(
                        "Number of Cylinders",
                        int(self.df['Cylinders'].min()),
                        int(self.df['Cylinders'].max()),
                        int(base_car['Cylinders'])
                    )
                    
                    # Emission Norm
                    emission_norm = st.slider(
                        "Emission Norm (BS)",
                        int(self.df['Emission_NormBS'].min()),
                        int(self.df['Emission_NormBS'].max()),
                        int(base_car['Emission_NormBS'])
                    )
                
                with col3:
                    # Fuel Capacity
                    fuel_capacity = st.slider(
                        "Fuel Capacity (L)",
                        float(self.df['Fuel_Capacity'].min()),
                        float(self.df['Fuel_Capacity'].max()),
                        float(base_car['Fuel_Capacity'])
                    )
                    
                    # Mileage
                    mileage = st.slider(
                        "Mileage (km/l)",
                        float(self.df['ARAI_Certified_Mileage'].min()),
                        float(self.df['ARAI_Certified_Mileage'].max()),
                        float(base_car['ARAI_Certified_Mileage'])
                    )
                    
                    # Wheelbase
                    wheelbase = st.slider(
                        "Wheelbase (mm)",
                        int(self.df['Wheelbase'].min()),
                        int(self.df['Wheelbase'].max()),
                        int(base_car['Wheelbase'])
                    )
                
                # Boot Space (full width)
                boot_space = st.slider(
                    "Boot Space (L)",
                    int(self.df['Boot_Space'].min()),
                    int(self.df['Boot_Space'].max()),
                    int(base_car['Boot_Space'])
                )
                
                return {
                    'Brand': st.session_state.selected_brand,
                    'Model': st.session_state.selected_model,
                    'Fuel_Type': selected_fuel,
                    'Body_Type': selected_body,
                    'Type': selected_transmission,
                    'Cylinders': cylinders,
                    'Emission_NormBS': emission_norm,
                    'Fuel_Capacity': fuel_capacity,
                    'ARAI_Certified_Mileage': mileage,
                    'Seating_Capacity': seating_capacity,
                    'Wheelbase': wheelbase,
                    'Boot_Space': boot_space
                }
        
        return None
    
    def display_similar_cars(self, inputs, predicted_price):
        """Display similar cars based on user inputs"""
        if self.df is None:
            return
        
        try:
            st.markdown("#### 🔍 Similar Cars")
            
            # Find cars with similar specifications
            similar_cars = self.df[
                (self.df['Brand'] == inputs['Brand']) |
                (self.df['Fuel_Type'] == inputs['Fuel_Type']) |
                (self.df['Body_Type'] == inputs['Body_Type'])
            ].copy()
            
            # Calculate price difference
            similar_cars['Price_Diff'] = abs(similar_cars['Ex_Showroom_PriceRs.'] - predicted_price)
            
            # Sort by price similarity and get top 3
            similar_cars = similar_cars.nsmallest(3, 'Price_Diff')
            
            if len(similar_cars) > 0:
                for idx, car in similar_cars.iterrows():
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.write(f"**{car['Brand']} {car['Model']}**")
                        st.write(f"*{car['Fuel_Type']} • {car['Body_Type']} • {car['Seating_Capacity']} seats*")
                    
                    with col2:
                        st.metric("Actual Price", f"₹{car['Ex_Showroom_PriceRs.']:,.0f}")
                    
                    with col3:
                        diff = car['Ex_Showroom_PriceRs.'] - predicted_price
                        diff_pct = (diff / predicted_price) * 100
                        st.metric("Difference", f"{diff_pct:+.1f}%")
                    
                    st.markdown("---")
            else:
                st.info("No similar cars found in the dataset.")
                
        except Exception as e:
            st.warning(f"Could not load similar cars: {e}")
    
    def prepare_input_for_prediction(self, inputs):
        """Prepare user inputs for model prediction"""
        # Create DataFrame with user inputs
        input_df = pd.DataFrame([inputs])
        
        # Apply same encoding as training
        # Label encode high cardinality features
        if 'Brand' in self.label_encoders:
            try:
                input_df['Brand'] = self.label_encoders['Brand'].transform([inputs['Brand']])[0]
            except ValueError:
                # Handle unseen brand
                input_df['Brand'] = 0
        
        if 'Body_Type' in self.label_encoders:
            try:
                input_df['Body_Type'] = self.label_encoders['Body_Type'].transform([inputs['Body_Type']])[0]
            except ValueError:
                input_df['Body_Type'] = 0
        
        # One-hot encode low cardinality features
        # Fuel Type
        fuel_columns = [col for col in self.feature_names if col.startswith('Fuel_Type_')]
        for col in fuel_columns:
            fuel_type = col.replace('Fuel_Type_', '')
            input_df[col] = 1 if inputs['Fuel_Type'] == fuel_type else 0
        
        # Transmission Type
        type_columns = [col for col in self.feature_names if col.startswith('Type_')]
        for col in type_columns:
            trans_type = col.replace('Type_', '')
            input_df[col] = 1 if inputs['Type'] == trans_type else 0
        
        # Remove original categorical columns that were one-hot encoded
        input_df = input_df.drop(['Fuel_Type', 'Type'], axis=1, errors='ignore')
        
        # Ensure all feature columns are present
        for feature in self.feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        # Reorder columns to match training data
        input_df = input_df[self.feature_names]
        
        return input_df
    
    def predict_price(self, inputs):
        """Make price prediction"""
        try:
            # Prepare input
            input_df = self.prepare_input_for_prediction(inputs)
            
            # Make prediction
            if self.best_model['name'] in ['Linear_Regression', 'Ridge_Regression', 'Lasso_Regression']:
                # Scale input for linear models
                input_scaled = self.scaler.transform(input_df)
                prediction = self.best_model['model'].predict(input_scaled)[0]
            else:
                # Use original features for tree-based models
                prediction = self.best_model['model'].predict(input_df)[0]
            
            return max(0, prediction)  # Ensure non-negative price
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None
    
    def display_model_performance(self):
        """Display model performance metrics"""
        st.header("📊 Model Performance")
        
        col1, col2, col3 = st.columns(3)
        
        best_model_name = self.best_model['name']
        best_scores = self.model_scores[best_model_name]
        
        with col1:
            st.metric(
                label="R² Score",
                value=f"{best_scores['test_r2']:.4f}",
                help="Coefficient of determination - higher is better"
            )
        
        with col2:
            st.metric(
                label="RMSE",
                value=f"₹{best_scores['test_rmse']:,.0f}",
                help="Root Mean Square Error - lower is better"
            )
        
        with col3:
            st.metric(
                label="MAE",
                value=f"₹{best_scores['test_mae']:,.0f}",
                help="Mean Absolute Error - lower is better"
            )
        
        # Model comparison chart
        st.subheader("Model Comparison")
        
        models_data = []
        for model_name, scores in self.model_scores.items():
            models_data.append({
                'Model': model_name,
                'R² Score': scores['test_r2'],
                'RMSE': scores['test_rmse'],
                'MAE': scores['test_mae']
            })
        
        models_df = pd.DataFrame(models_data)
        
        fig = px.bar(models_df, x='Model', y='R² Score', 
                     title='Model Performance Comparison (R² Score)',
                     color='R² Score', color_continuous_scale='viridis')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def display_market_insights(self):
        """Display market insights and clustering results"""
        st.header("🎯 Market Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Price distribution
            fig = px.histogram(self.df, x='Ex_Showroom_PriceRs.', nbins=30,
                             title='Price Distribution in Dataset')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Brand popularity
            brand_counts = self.df['Brand'].value_counts().head(10)
            fig = px.bar(x=brand_counts.values, y=brand_counts.index, 
                        orientation='h', title='Top 10 Brands by Count')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Clustering visualization if available
        if 'Cluster' in self.df.columns:
            st.subheader("🎯 Market Segmentation")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.scatter(self.df, x='ARAI_Certified_Mileage', y='Ex_Showroom_PriceRs.',
                               color='Cluster', title='Price vs Mileage by Market Segment',
                               labels={'Ex_Showroom_PriceRs.': 'Price (Rs.)', 
                                      'ARAI_Certified_Mileage': 'Mileage (km/l)'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                cluster_counts = self.df['Cluster'].value_counts().sort_index()
                fig = px.pie(values=cluster_counts.values, names=cluster_counts.index,
                           title='Market Segment Distribution')
                st.plotly_chart(fig, use_container_width=True)
    
    def display_feature_analysis(self, inputs):
        """Display feature analysis for the selected car"""
        st.header("🔍 Feature Analysis")
        
        # Compare with similar cars
        similar_cars = self.df[
            (self.df['Brand'] == inputs['Brand']) |
            (self.df['Fuel_Type'] == inputs['Fuel_Type']) |
            (self.df['Body_Type'] == inputs['Body_Type'])
        ]
        
        if len(similar_cars) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Similar Cars Price Range")
                avg_price = similar_cars['Ex_Showroom_PriceRs.'].mean()
                min_price = similar_cars['Ex_Showroom_PriceRs.'].min()
                max_price = similar_cars['Ex_Showroom_PriceRs.'].max()
                
                st.metric("Average Price", f"₹{avg_price:,.0f}")
                st.metric("Price Range", f"₹{min_price:,.0f} - ₹{max_price:,.0f}")
            
            with col2:
                st.subheader("Feature Comparison")
                fig = px.box(similar_cars, y='Ex_Showroom_PriceRs.',
                           title=f'Price Distribution for Similar Cars')
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    def render_home_page(self):
        """Render the home/dashboard page"""
        st.markdown('<h2 class="page-header">Welcome to CarPrice AI</h2>', unsafe_allow_html=True)
        
        # Hero section
        st.markdown("""
        <div class="feature-highlight">
            <h3>🎯 Accurate Car Price Predictions</h3>
            <p>Our advanced machine learning models analyze multiple factors to provide precise car price estimates with 83.78% accuracy using Random Forest algorithm.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick stats
        if self.df is not None:
            st.markdown('<h3 class="section-header">📊 Platform Statistics</h3>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="stat-box">
                    <p class="stat-number">{len(self.df)}</p>
                    <p class="stat-label">Cars Analyzed</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stat-box">
                    <p class="stat-number">{self.df['Brand'].nunique()}</p>
                    <p class="stat-label">Brands Covered</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="stat-box">
                    <p class="stat-number">83.78%</p>
                    <p class="stat-label">Model Accuracy</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                avg_price = self.df['Ex_Showroom_PriceRs.'].mean()
                st.markdown(f"""
                <div class="stat-box">
                    <p class="stat-number">₹{avg_price/1000000:.1f}M</p>
                    <p class="stat-label">Avg Price</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Features overview
        st.markdown('<h3 class="section-header">🚀 Key Features</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-card">
                <h4>💰 Price Prediction</h4>
                <p>Get instant car price estimates based on specifications like brand, fuel type, engine details, and more.</p>
            </div>
            
            <div class="info-card">
                <h4>📊 Model Analytics</h4>
                <p>Compare different ML algorithms and understand model performance with detailed metrics and visualizations.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-card">
                <h4>🎯 Market Insights</h4>
                <p>Explore market trends, brand analysis, and customer segmentation through interactive charts.</p>
            </div>
            
            <div class="info-card">
                <h4>🔍 Feature Analysis</h4>
                <p>Understand which factors most influence car prices with SHAP explanations and feature importance.</p>
            </div>
            """, unsafe_allow_html=True)
    
    def render_price_predictor(self):
        """Render the price prediction page"""
        st.markdown('<h2 class="page-title">💰 Car Price Predictor</h2>', unsafe_allow_html=True)
        
        # Create cascading filter inputs
        inputs = self.create_cascading_filters()
        
        if inputs is None:
            st.info("👆 Please select a brand and model above to get started with price prediction.")
            return
        
        # Add some spacing
        st.markdown("---")
        
        # Prediction section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 🔮 Get Price Prediction")
            if st.button("Predict Price", type="primary", use_container_width=True):
                try:
                    # Prepare input for prediction
                    input_array = self.prepare_input_for_prediction(inputs)
                    
                    if input_array is not None:
                        # Make prediction using the correct model structure
                        if self.best_model['name'] in ['Linear_Regression', 'Ridge_Regression', 'Lasso_Regression']:
                            # Scale input for linear models
                            input_scaled = self.scaler.transform(input_array)
                            prediction = self.best_model['model'].predict(input_scaled)[0]
                        else:
                            # Use original features for tree-based models
                            prediction = self.best_model['model'].predict(input_array)[0]
                        
                        # Calculate confidence interval (using model's prediction variance)
                        confidence_interval = prediction * 0.1  # 10% confidence interval
                        
                        # Display prediction with enhanced styling
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                   padding: 20px; border-radius: 10px; margin: 10px 0; color: white; text-align: center;">
                            <h2 style="margin: 0; color: white;">₹{prediction:,.0f}</h2>
                            <p style="margin: 5px 0; opacity: 0.9;">Predicted Price</p>
                            <p style="margin: 0; font-size: 14px; opacity: 0.8;">
                                Range: ₹{prediction-confidence_interval:,.0f} - ₹{prediction+confidence_interval:,.0f}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show input summary in a nice table
                        st.markdown("#### 📋 Selected Specifications")
                        
                        # Create a more organized display
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.markdown(f"""
                            **Vehicle Details:**
                            - **Brand:** {inputs['Brand']}
                            - **Model:** {inputs['Model']}
                            - **Fuel Type:** {inputs['Fuel_Type']}
                            - **Body Type:** {inputs['Body_Type']}
                            - **Transmission:** {inputs['Type']}
                            - **Seating:** {inputs['Seating_Capacity']} seats
                            """)
                        
                        with col_b:
                            st.markdown(f"""
                            **Technical Specs:**
                            - **Cylinders:** {inputs['Cylinders']}
                            - **Emission Norm:** BS{inputs['Emission_NormBS']}
                            - **Fuel Capacity:** {inputs['Fuel_Capacity']} L
                            - **Mileage:** {inputs['ARAI_Certified_Mileage']} km/l
                            - **Wheelbase:** {inputs['Wheelbase']} mm
                            - **Boot Space:** {inputs['Boot_Space']} L
                            """)
                        
                        # Find similar cars
                        self.display_similar_cars(inputs, prediction)
                        
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
        
        with col2:
            # Show model performance and statistics
            st.markdown("#### 📊 Model Performance")
            if hasattr(self, 'model_scores') and self.model_scores:
                # Extract R² scores from model_scores dictionary
                r2_scores = [scores['test_r2'] for scores in self.model_scores.values()]
                best_score = max(r2_scores)
                st.metric("Model Accuracy", f"{best_score:.1%}", help="R² score of the best performing model")
            
            if self.df is not None:
                # Show relevant statistics
                avg_price = self.df['Ex_Showroom_PriceRs.'].mean()
                st.metric("Market Average", f"₹{avg_price:,.0f}", help="Average price across all cars")
                
                # Show brand-specific stats if brand is selected
                if 'selected_brand' in st.session_state and st.session_state.selected_brand:
                    brand_df = self.df[self.df['Brand'] == st.session_state.selected_brand]
                    brand_avg = brand_df['Ex_Showroom_PriceRs.'].mean()
                    st.metric(f"{st.session_state.selected_brand} Average", f"₹{brand_avg:,.0f}")
                    
                    total_models = len(brand_df)
                    st.metric("Available Variants", total_models)
    
    def render_model_analytics_page(self):
        """Render the model analytics page"""
        st.markdown('<h2 class="page-header">📊 Model Analytics</h2>', unsafe_allow_html=True)
        self.display_model_performance()
    
    def render_visualizations_page(self):
        """Render comprehensive visualizations page"""
        st.markdown('<h2 class="page-title">📊 Data Visualizations</h2>', unsafe_allow_html=True)
        
        if self.df is None:
            st.error("Data not loaded. Please check if cleaned_car_data.csv exists.")
            return
        
        # Visualization filters
        st.markdown("### 🎛️ Visualization Filters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_brands = st.multiselect(
                "Filter by Brands",
                options=sorted(self.df['Brand'].unique()),
                default=[],
                help="Leave empty to show all brands"
            )
        
        with col2:
            selected_fuel_types = st.multiselect(
                "Filter by Fuel Type",
                options=sorted(self.df['Fuel_Type'].unique()),
                default=[],
                help="Leave empty to show all fuel types"
            )
        
        with col3:
            price_range = st.slider(
                "Price Range (₹)",
                min_value=int(self.df['Ex_Showroom_PriceRs.'].min()),
                max_value=int(self.df['Ex_Showroom_PriceRs.'].max()),
                value=(int(self.df['Ex_Showroom_PriceRs.'].min()), int(self.df['Ex_Showroom_PriceRs.'].max())),
                format="₹%d"
            )
        
        # Apply filters
        filtered_df = self.df.copy()
        if selected_brands:
            filtered_df = filtered_df[filtered_df['Brand'].isin(selected_brands)]
        if selected_fuel_types:
            filtered_df = filtered_df[filtered_df['Fuel_Type'].isin(selected_fuel_types)]
        filtered_df = filtered_df[
            (filtered_df['Ex_Showroom_PriceRs.'] >= price_range[0]) & 
            (filtered_df['Ex_Showroom_PriceRs.'] <= price_range[1])
        ]
        
        st.markdown("---")
        
        # Dataset Overview
        st.markdown("### 📈 Dataset Overview")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Cars", len(filtered_df))
        with col2:
            st.metric("Brands", filtered_df['Brand'].nunique())
        with col3:
            st.metric("Avg Price", f"₹{filtered_df['Ex_Showroom_PriceRs.'].mean():,.0f}")
        with col4:
            st.metric("Price Range", f"₹{filtered_df['Ex_Showroom_PriceRs.'].min():,.0f} - ₹{filtered_df['Ex_Showroom_PriceRs.'].max():,.0f}")
        with col5:
            st.metric("Fuel Types", filtered_df['Fuel_Type'].nunique())
        
        st.markdown("---")
        
        # Section 1: Price Analysis
        st.markdown("### 💰 Price Analysis")
        st.markdown("**Understanding car price distribution patterns and brand-wise pricing strategies**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Price distribution histogram
            fig_hist = px.histogram(
                filtered_df, 
                x='Ex_Showroom_PriceRs.', 
                nbins=30,
                title='Price Distribution',
                labels={'Ex_Showroom_PriceRs.': 'Price (₹)', 'count': 'Number of Cars'},
                color_discrete_sequence=['#667eea']
            )
            fig_hist.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)
            
            st.markdown("""
            **📊 What this shows:** Distribution of car prices across the dataset
            
            **🔍 Key Insights:**
            - Most cars are concentrated in the lower price range
            - Shows market segments (economy, mid-range, luxury)
            - Right-skewed distribution indicates few very expensive cars
            - Helps identify price clusters and market gaps
            """)
        
        with col2:
            # Price by brand boxplot
            top_brands = filtered_df['Brand'].value_counts().head(10).index
            brand_data = filtered_df[filtered_df['Brand'].isin(top_brands)]
            
            fig_box = px.box(
                brand_data, 
                x='Brand', 
                y='Ex_Showroom_PriceRs.',
                title='Price Distribution by Top 10 Brands',
                labels={'Ex_Showroom_PriceRs.': 'Price (₹)'},
                color='Brand'
            )
            fig_box.update_layout(height=400, xaxis_tickangle=-45, showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)
            
            st.markdown("""
            **📊 What this shows:** Price range and variability for each brand
            
            **🔍 Key Insights:**
            - Box shows median price and quartiles for each brand
            - Whiskers indicate price range within each brand
            - Outliers represent unusually priced models
            - Helps compare brand positioning (luxury vs economy)
            """)
        
        st.markdown("---")
        
        # Section 2: Brand Analysis
        st.markdown("### 🏢 Brand Analysis")
        st.markdown("**Analyzing brand market presence and pricing strategies**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Brand popularity
            brand_counts = filtered_df['Brand'].value_counts().head(15)
            fig_brand = px.bar(
                x=brand_counts.values,
                y=brand_counts.index,
                orientation='h',
                title='Top 15 Brands by Number of Models',
                labels={'x': 'Number of Models', 'y': 'Brand'},
                color=brand_counts.values,
                color_continuous_scale='viridis'
            )
            fig_brand.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig_brand, use_container_width=True)
            
            st.markdown("""
            **📊 What this shows:** Market presence by model variety
            
            **🔍 Key Insights:**
            - Brands with most diverse model portfolios
            - Market coverage and customer choice options
            - Indicates brand's commitment to different segments
            - Higher count suggests broader market strategy
            """)
        
        with col2:
            # Average price by brand
            avg_price_by_brand = filtered_df.groupby('Brand')['Ex_Showroom_PriceRs.'].mean().sort_values(ascending=False).head(15)
            fig_avg_price = px.bar(
                x=avg_price_by_brand.index,
                y=avg_price_by_brand.values,
                title='Top 15 Brands by Average Price',
                labels={'x': 'Brand', 'y': 'Average Price (₹)'},
                color=avg_price_by_brand.values,
                color_continuous_scale='plasma'
            )
            fig_avg_price.update_layout(height=500, xaxis_tickangle=-45, showlegend=False)
            st.plotly_chart(fig_avg_price, use_container_width=True)
            
            st.markdown("""
            **📊 What this shows:** Brand positioning by average pricing
            
            **🔍 Key Insights:**
            - Luxury vs economy brand classification
            - Premium positioning strategies
            - Price-based market segmentation
            - Brand value perception in the market
            """)
        
        st.markdown("---")
        
        # Section 3: Fuel Type & Body Type Analysis
        st.markdown("### ⛽ Fuel Type & Body Type Analysis")
        st.markdown("**Market composition by fuel technology and vehicle categories**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Fuel type distribution
            fuel_counts = filtered_df['Fuel_Type'].value_counts()
            fig_fuel = px.pie(
                values=fuel_counts.values,
                names=fuel_counts.index,
                title='Fuel Type Distribution',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_fuel.update_layout(height=400)
            st.plotly_chart(fig_fuel, use_container_width=True)
            
            st.markdown("""
            **📊 What this shows:** Market share of different fuel technologies
            
            **🔍 Key Insights:**
            - Dominance of traditional vs alternative fuels
            - Electric/hybrid adoption trends
            - Environmental consciousness indicators
            - Future market direction predictions
            """)
        
        with col2:
            # Body type distribution
            body_counts = filtered_df['Body_Type'].value_counts()
            fig_body = px.pie(
                values=body_counts.values,
                names=body_counts.index,
                title='Body Type Distribution',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_body.update_layout(height=400)
            st.plotly_chart(fig_body, use_container_width=True)
            
            st.markdown("""
            **📊 What this shows:** Consumer preferences for vehicle categories
            
            **🔍 Key Insights:**
            - Popular vehicle form factors
            - Lifestyle and usage pattern indicators
            - Market demand for different vehicle types
            - Urban vs family-oriented preferences
            """)
        
        st.markdown("---")
        
        # Section 4: Performance Analysis
        st.markdown("### 🚗 Performance Analysis")
        st.markdown("**Relationship between vehicle performance metrics and pricing**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Mileage vs Price scatter
            fig_scatter = px.scatter(
                filtered_df,
                x='ARAI_Certified_Mileage',
                y='Ex_Showroom_PriceRs.',
                color='Fuel_Type',
                size='Cylinders',
                hover_data=['Brand', 'Model'],
                title='Mileage vs Price Analysis',
                labels={'ARAI_Certified_Mileage': 'Mileage (km/l)', 'Ex_Showroom_PriceRs.': 'Price (₹)'}
            )
            fig_scatter.update_layout(height=500)
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            st.markdown("""
            **📊 What this shows:** Fuel efficiency vs pricing relationship
            
            **🔍 Key Insights:**
            - Higher mileage doesn't always mean lower price
            - Fuel type significantly affects this relationship
            - Bubble size shows engine complexity (cylinders)
            - Performance vs economy trade-offs
            """)
        
        with col2:
            # Seating capacity distribution
            seating_counts = filtered_df['Seating_Capacity'].value_counts().sort_index()
            fig_seating = px.bar(
                x=seating_counts.index,
                y=seating_counts.values,
                title='Seating Capacity Distribution',
                labels={'x': 'Seating Capacity', 'y': 'Number of Cars'},
                color=seating_counts.values,
                color_continuous_scale='blues'
            )
            fig_seating.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig_seating, use_container_width=True)
            
            st.markdown("""
            **📊 What this shows:** Market preference for family size accommodation
            
            **🔍 Key Insights:**
            - Most popular seating configurations
            - Family vs individual transportation needs
            - Commercial vs personal use indicators
            - Market gaps in specific seating categories
            """)
        
        st.markdown("---")
        
        # Section 5: Technical Specifications
        st.markdown("### 🔧 Technical Specifications")
        st.markdown("**Engine complexity and transmission preferences impact on pricing**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cylinders vs Price
            fig_cyl = px.box(
                filtered_df,
                x='Cylinders',
                y='Ex_Showroom_PriceRs.',
                title='Price Distribution by Number of Cylinders',
                labels={'Ex_Showroom_PriceRs.': 'Price (₹)'},
                color='Cylinders'
            )
            fig_cyl.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_cyl, use_container_width=True)
            
            st.markdown("""
            **📊 What this shows:** Engine complexity vs pricing relationship
            
            **🔍 Key Insights:**
            - More cylinders generally mean higher prices
            - Engine performance and price correlation
            - Premium segment preference for larger engines
            - Power vs efficiency market segments
            """)
        
        with col2:
            # Transmission type distribution
            trans_counts = filtered_df['Type'].value_counts()
            fig_trans = px.bar(
                x=trans_counts.index,
                y=trans_counts.values,
                title='Transmission Type Distribution',
                labels={'x': 'Transmission Type', 'y': 'Number of Cars'},
                color=trans_counts.values,
                color_continuous_scale='greens'
            )
            fig_trans.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_trans, use_container_width=True)
            
            st.markdown("""
            **📊 What this shows:** Consumer preference for transmission types
            
            **🔍 Key Insights:**
            - Manual vs automatic market adoption
            - Convenience vs cost considerations
            - Urban driving pattern influences
            - Technology advancement trends
            """)
        
        st.markdown("---")
        
        # Section 6: Correlation Analysis
        st.markdown("### 📊 Correlation Analysis")
        st.markdown("**Understanding relationships between different car features and pricing**")
        
        # Select numeric columns for correlation
        numeric_cols = ['Ex_Showroom_PriceRs.', 'Cylinders', 'Fuel_Capacity', 'ARAI_Certified_Mileage', 
                       'Seating_Capacity', 'Wheelbase', 'Boot_Space', 'Emission_NormBS']
        
        correlation_matrix = filtered_df[numeric_cols].corr()
        
        fig_corr = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            title='Feature Correlation Matrix',
            color_continuous_scale='RdBu_r'
        )
        fig_corr.update_layout(height=600)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.markdown("""
        **📊 What this shows:** Statistical relationships between all numeric features
        
        **🔍 Key Insights:**
        - **Red colors** indicate positive correlations (features increase together)
        - **Blue colors** indicate negative correlations (one increases, other decreases)
        - **Values closer to 1 or -1** show stronger relationships
        - **Price correlations** reveal which features most influence car pricing
        - **Feature interdependencies** help understand car design trade-offs
        
        **💡 How to read:** Look for dark red/blue cells connecting price with other features to identify key pricing factors.
        """)
        
        st.markdown("---")
        
        # Section 7: Advanced Analysis
        st.markdown("### 🎯 Advanced Analysis")
        st.markdown("**Deep dive into complex relationships with trend analysis**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Price vs Mileage by Fuel Type
            fig_fuel_analysis = px.scatter(
                filtered_df,
                x='ARAI_Certified_Mileage',
                y='Ex_Showroom_PriceRs.',
                color='Fuel_Type',
                trendline='ols',
                title='Price vs Mileage by Fuel Type',
                labels={'ARAI_Certified_Mileage': 'Mileage (km/l)', 'Ex_Showroom_PriceRs.': 'Price (₹)'}
            )
            fig_fuel_analysis.update_layout(height=500)
            st.plotly_chart(fig_fuel_analysis, use_container_width=True)
            
            st.markdown("""
            **📊 What this shows:** Fuel efficiency trends across different fuel technologies
            
            **🔍 Key Insights:**
            - **Trend lines** show general price-mileage relationships by fuel type
            - **Electric/Hybrid** vehicles may show different patterns
            - **Diesel** cars often have better mileage but varied pricing
            - **Performance vs efficiency** trade-offs by fuel technology
            """)
        
        with col2:
            # Boot Space vs Price
            fig_boot = px.scatter(
                filtered_df,
                x='Boot_Space',
                y='Ex_Showroom_PriceRs.',
                color='Body_Type',
                title='Boot Space vs Price by Body Type',
                labels={'Boot_Space': 'Boot Space (L)', 'Ex_Showroom_PriceRs.': 'Price (₹)'}
            )
            fig_boot.update_layout(height=500)
            st.plotly_chart(fig_boot, use_container_width=True)
            
            st.markdown("""
            **📊 What this shows:** Storage capacity vs pricing across vehicle categories
            
            **🔍 Key Insights:**
            - **SUVs and Sedans** typically offer more boot space
            - **Sports cars** sacrifice storage for performance
            - **Premium vehicles** may have larger boot space at higher prices
            - **Practical vs luxury** positioning strategies
            """)
        
        st.markdown("---")
        
        # Data Export Section
        st.markdown("---")
        st.markdown("### 📥 Export Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_data = filtered_df.to_csv(index=False)
            st.download_button(
                label="📊 Download Filtered Data (CSV)",
                data=csv_data,
                file_name="filtered_car_data.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Summary statistics
            summary_stats = filtered_df.describe()
            summary_csv = summary_stats.to_csv()
            st.download_button(
                label="📈 Download Summary Statistics",
                data=summary_csv,
                file_name="car_data_summary.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col3:
            st.info(f"**Current Filter Results:** {len(filtered_df)} cars from {filtered_df['Brand'].nunique()} brands")
    
    def render_market_insights_page(self):
        """Render the market insights page"""
        st.markdown('<h2 class="page-header">🎯 Market Insights</h2>', unsafe_allow_html=True)
        self.display_market_insights()
        
        # Additional dataset overview
        if self.df is not None:
            st.markdown('<h3 class="section-header">📈 Dataset Overview</h3>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Cars", len(self.df))
            with col2:
                st.metric("Brands", self.df['Brand'].nunique())
            with col3:
                st.metric("Avg Price", f"₹{self.df['Ex_Showroom_PriceRs.'].mean():,.0f}")
            with col4:
                st.metric("Price Range", f"₹{self.df['Ex_Showroom_PriceRs.'].min():,.0f} - ₹{self.df['Ex_Showroom_PriceRs.'].max():,.0f}")
            
            # Dataset sample
            st.subheader("Sample Data")
            display_df = self.df[['Brand', 'Model', 'Ex_Showroom_PriceRs.', 'Fuel_Type', 
                                'Body_Type', 'ARAI_Certified_Mileage', 'Seating_Capacity']].head(10)
            display_df['Ex_Showroom_PriceRs.'] = display_df['Ex_Showroom_PriceRs.'].apply(lambda x: f"₹{x:,.0f}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    def render_about_page(self):
        """Render the about page"""
        st.markdown('<h2 class="page-header">📚 About CarPrice AI</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h3>🎯 Project Overview</h3>
            <p>CarPrice AI is an advanced machine learning platform that predicts car prices using comprehensive data analysis and multiple ML algorithms. Our system analyzes various factors including brand, specifications, performance metrics, and market trends to provide accurate price estimates.</p>
        </div>
        
        <div class="info-card">
            <h3>🔬 Technology Stack</h3>
            <ul>
                <li><strong>Data Processing:</strong> Pandas, NumPy</li>
                <li><strong>Machine Learning:</strong> Scikit-learn, XGBoost</li>
                <li><strong>Visualization:</strong> Matplotlib, Seaborn, Plotly</li>
                <li><strong>Web Application:</strong> Streamlit</li>
                <li><strong>Model Explainability:</strong> SHAP, LIME</li>
            </ul>
        </div>
        
        <div class="info-card">
            <h3>📊 Model Performance</h3>
            <p>Our best performing model (Random Forest) achieves:</p>
            <ul>
                <li><strong>R² Score:</strong> 0.8378 (83.78% accuracy)</li>
                <li><strong>RMSE:</strong> ₹5,132,652</li>
                <li><strong>MAE:</strong> ₹2,958,538</li>
            </ul>
        </div>
        
        <div class="info-card">
            <h3>🎯 Key Features</h3>
            <ul>
                <li>Real-time price prediction with confidence intervals</li>
                <li>Interactive model comparison and analytics</li>
                <li>Market segmentation using K-means clustering</li>
                <li>Feature importance analysis with SHAP explanations</li>
                <li>Comprehensive data visualization and insights</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Footer
        st.markdown("---")
        st.markdown("**Built with:** Python, Streamlit, Scikit-learn, XGBoost, Plotly")
        st.markdown("**Best Model:** Random Forest (R² = 0.8378)")
    
    def render_available_details_page(self):
        """Render the available details page"""
        st.markdown('<h2 class="page-title">📋 Available Details</h2>', unsafe_allow_html=True)
        
        if self.df is None:
            st.error("Data not loaded. Please check if cleaned_car_data.csv exists.")
            return
        
        st.markdown("### Select a Brand to View Available Car Details")
        
        # Brand selection
        brands = ['Select a brand...'] + sorted(self.df['Brand'].unique().tolist())
        selected_brand = st.selectbox(
            "Choose Car Brand",
            brands,
            key="available_details_brand_select",
            help="Select a brand to view all available car models and their specifications"
        )
        
        if selected_brand != 'Select a brand...':
            # Filter data for selected brand
            brand_data = self.df[self.df['Brand'] == selected_brand].copy()
            
            # Display brand summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Models", len(brand_data))
            
            with col2:
                avg_price = brand_data['Ex_Showroom_PriceRs.'].mean()
                st.metric("Average Price", f"₹{avg_price:,.0f}")
            
            with col3:
                min_price = brand_data['Ex_Showroom_PriceRs.'].min()
                max_price = brand_data['Ex_Showroom_PriceRs.'].max()
                st.metric("Price Range", f"₹{min_price:,.0f} - ₹{max_price:,.0f}")
            
            with col4:
                fuel_types = brand_data['Fuel_Type'].nunique()
                st.metric("Fuel Types", fuel_types)
            
            st.markdown("---")
            
            # Display detailed data table
            st.markdown(f"### 🚗 All {selected_brand} Models Available")
            
            # Prepare display dataframe
            display_df = brand_data.copy()
            
            # Format price column for better readability
            display_df['Price (₹)'] = display_df['Ex_Showroom_PriceRs.'].apply(lambda x: f"₹{x:,.0f}")
            display_df['Mileage (km/l)'] = display_df['ARAI_Certified_Mileage'].apply(lambda x: f"{x:.1f}")
            display_df['Fuel Capacity (L)'] = display_df['Fuel_Capacity'].apply(lambda x: f"{x:.1f}")
            display_df['Boot Space (L)'] = display_df['Boot_Space'].apply(lambda x: f"{x}")
            display_df['Wheelbase (mm)'] = display_df['Wheelbase'].apply(lambda x: f"{x}")
            
            # Select columns to display
            columns_to_show = [
                'Model', 'Price (₹)', 'Fuel_Type', 'Body_Type', 'Type', 
                'Seating_Capacity', 'Mileage (km/l)', 'Cylinders', 
                'Fuel Capacity (L)', 'Wheelbase (mm)', 'Boot Space (L)', 'Emission_NormBS'
            ]
            
            # Create the display dataframe
            final_display_df = display_df[columns_to_show].copy()
            
            # Rename columns for better display
            final_display_df.columns = [
                'Model', 'Price', 'Fuel Type', 'Body Type', 'Transmission',
                'Seats', 'Mileage', 'Cylinders', 'Fuel Tank', 'Wheelbase', 'Boot Space', 'Emission Norm'
            ]
            
            # Display the dataframe
            st.dataframe(
                final_display_df,
                use_container_width=True,
                hide_index=True,
                height=400
            )
            
            # Additional insights
            st.markdown("---")
            st.markdown(f"### 📊 {selected_brand} Brand Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Most common fuel type
                most_common_fuel = brand_data['Fuel_Type'].mode().iloc[0]
                fuel_count = brand_data[brand_data['Fuel_Type'] == most_common_fuel].shape[0]
                st.info(f"**Most Common Fuel Type:** {most_common_fuel} ({fuel_count} models)")
                
                # Most common body type
                most_common_body = brand_data['Body_Type'].mode().iloc[0]
                body_count = brand_data[brand_data['Body_Type'] == most_common_body].shape[0]
                st.info(f"**Most Common Body Type:** {most_common_body} ({body_count} models)")
            
            with col2:
                # Most expensive model
                most_expensive = brand_data.loc[brand_data['Ex_Showroom_PriceRs.'].idxmax()]
                st.success(f"**Most Expensive:** {most_expensive['Model']} - ₹{most_expensive['Ex_Showroom_PriceRs.']:,.0f}")
                
                # Most fuel efficient model
                most_efficient = brand_data.loc[brand_data['ARAI_Certified_Mileage'].idxmax()]
                st.success(f"**Most Fuel Efficient:** {most_efficient['Model']} - {most_efficient['ARAI_Certified_Mileage']:.1f} km/l")
            
            # Download option
            st.markdown("---")
            csv_data = brand_data.to_csv(index=False)
            st.download_button(
                label=f"📥 Download {selected_brand} Data as CSV",
                data=csv_data,
                file_name=f"{selected_brand.replace(' ', '_')}_car_data.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        else:
            # Show overall dataset summary when no brand is selected
            st.markdown("### 📈 Dataset Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Cars", len(self.df))
            
            with col2:
                st.metric("Total Brands", self.df['Brand'].nunique())
            
            with col3:
                avg_price = self.df['Ex_Showroom_PriceRs.'].mean()
                st.metric("Overall Average Price", f"₹{avg_price:,.0f}")
            
            with col4:
                st.metric("Fuel Types", self.df['Fuel_Type'].nunique())
            
            st.info("👆 Select a brand from the dropdown above to view detailed information about all available models from that brand.")
    
    def run(self):
        """Main app function with navigation"""
        # Render header and navigation
        current_page = self.render_header_and_navigation()
        
        # Render different pages based on button selection
        if current_page == 'Home':
            self.render_home_page()
        elif current_page == 'Price Predictor':
            self.render_price_predictor()
        elif current_page == 'Model Analytics':
            self.render_model_analytics_page()
        elif current_page == 'Visualizations':
            self.render_visualizations_page()
        elif current_page == 'Available Details':
            self.render_available_details_page()
        elif current_page == 'About':
            self.render_about_page()

def main():
    """Main function"""
    app = CarPricePredictorApp()
    app.run()

if __name__ == "__main__":
    main()
