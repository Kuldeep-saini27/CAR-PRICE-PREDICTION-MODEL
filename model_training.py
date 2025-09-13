import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

# Explainability Libraries
import shap
import lime
import lime.lime_tabular

# Model Persistence
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class CarPricePredictionModel:
    def __init__(self, data_path='cleaned_car_data.csv'):
        """Initialize the model with cleaned data"""
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.model_scores = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        
    def load_and_prepare_data(self):
        """Load and prepare data for machine learning"""
        print("🔄 Loading and preparing data for ML...")
        
        # Load cleaned data
        self.df = pd.read_csv(self.data_path)
        print(f"📊 Dataset shape: {self.df.shape}")
        
        # Prepare features and target
        target_col = 'Ex_Showroom_PriceRs.'
        self.y = self.df[target_col].values
        
        # Select features for modeling
        feature_cols = [
            'Brand', 'Fuel_Type', 'Body_Type', 'Type',  # Categorical
            'Cylinders', 'Emission_NormBS', 'Fuel_Capacity', 
            'ARAI_Certified_Mileage', 'Seating_Capacity', 'Wheelbase', 'Boot_Space'  # Numerical
        ]
        
        self.X = self.df[feature_cols].copy()
        print(f"✅ Selected {len(feature_cols)} features for modeling")
        
        return self.X, self.y
    
    def encode_categorical_features(self):
        """Encode categorical features"""
        print("🔄 Encoding categorical features...")
        
        categorical_cols = ['Brand', 'Fuel_Type', 'Body_Type', 'Type']
        
        # Use Label Encoding for high cardinality features like Brand
        # Use One-Hot Encoding for low cardinality features
        
        for col in categorical_cols:
            if col in self.X.columns:
                if self.X[col].nunique() > 10:  # High cardinality - use Label Encoding
                    le = LabelEncoder()
                    self.X[col] = le.fit_transform(self.X[col].astype(str))
                    self.label_encoders[col] = le
                    print(f"✅ Label encoded {col} ({self.X[col].nunique()} unique values)")
                else:  # Low cardinality - use One-Hot Encoding
                    dummies = pd.get_dummies(self.X[col], prefix=col)
                    self.X = pd.concat([self.X.drop(col, axis=1), dummies], axis=1)
                    print(f"✅ One-hot encoded {col} ({dummies.shape[1]} new columns)")
        
        self.feature_names = list(self.X.columns)
        print(f"📊 Final feature count: {len(self.feature_names)}")
        
        return self.X
    
    def split_and_scale_data(self, test_size=0.2, random_state=42):
        """Split data into train/test and scale features"""
        print("🔄 Splitting and scaling data...")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"✅ Train set: {self.X_train.shape[0]} samples")
        print(f"✅ Test set: {self.X_test.shape[0]} samples")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_linear_regression(self):
        """Train Linear Regression model"""
        print("🔄 Training Linear Regression...")
        
        # Regular Linear Regression
        lr = LinearRegression()
        lr.fit(self.X_train_scaled, self.y_train)
        
        # Ridge Regression
        ridge = Ridge(alpha=1.0)
        ridge.fit(self.X_train_scaled, self.y_train)
        
        # Lasso Regression
        lasso = Lasso(alpha=1000.0)
        lasso.fit(self.X_train_scaled, self.y_train)
        
        self.models['Linear_Regression'] = lr
        self.models['Ridge_Regression'] = ridge
        self.models['Lasso_Regression'] = lasso
        
        print("✅ Linear models trained successfully")
        
    def train_random_forest(self):
        """Train Random Forest model with hyperparameter tuning"""
        print("🔄 Training Random Forest...")
        
        # Basic Random Forest
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(self.X_train, self.y_train)
        
        self.models['Random_Forest'] = rf
        print("✅ Random Forest trained successfully")
        
    def train_xgboost(self):
        """Train XGBoost model"""
        print("🔄 Training XGBoost...")
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(self.X_train, self.y_train)
        
        self.models['XGBoost'] = xgb_model
        print("✅ XGBoost trained successfully")
        
    def evaluate_models(self):
        """Evaluate all trained models"""
        print("🔄 Evaluating models...")
        
        results = []
        
        for name, model in self.models.items():
            # Choose appropriate data based on model type
            if name in ['Linear_Regression', 'Ridge_Regression', 'Lasso_Regression']:
                X_train_eval = self.X_train_scaled
                X_test_eval = self.X_test_scaled
            else:
                X_train_eval = self.X_train
                X_test_eval = self.X_test
            
            # Make predictions
            y_train_pred = model.predict(X_train_eval)
            y_test_pred = model.predict(X_test_eval)
            
            # Calculate metrics
            train_r2 = r2_score(self.y_train, y_train_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
            train_mae = mean_absolute_error(self.y_train, y_train_pred)
            test_mae = mean_absolute_error(self.y_test, y_test_pred)
            
            # Store results
            self.model_scores[name] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae
            }
            
            results.append({
                'Model': name,
                'Train_R2': train_r2,
                'Test_R2': test_r2,
                'Train_RMSE': train_rmse,
                'Test_RMSE': test_rmse,
                'Train_MAE': train_mae,
                'Test_MAE': test_mae
            })
            
            print(f"✅ {name}:")
            print(f"   R² Score: {test_r2:.4f}")
            print(f"   RMSE: ₹{test_rmse:,.0f}")
            print(f"   MAE: ₹{test_mae:,.0f}")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Find best model based on test R²
        best_idx = results_df['Test_R2'].idxmax()
        self.best_model_name = results_df.loc[best_idx, 'Model']
        self.best_model = self.models[self.best_model_name]
        
        print(f"\n🏆 Best Model: {self.best_model_name} (R² = {results_df.loc[best_idx, 'Test_R2']:.4f})")
        
        return results_df
    
    def perform_clustering(self, n_clusters=5):
        """Perform K-means clustering for market segmentation"""
        print(f"🔄 Performing K-means clustering with {n_clusters} clusters...")
        
        # Select features for clustering
        cluster_features = ['Ex_Showroom_PriceRs.', 'ARAI_Certified_Mileage', 'Cylinders', 
                           'Fuel_Capacity', 'Seating_Capacity', 'Wheelbase']
        
        cluster_data = self.df[cluster_features].copy()
        
        # Scale the data for clustering
        cluster_scaler = StandardScaler()
        cluster_data_scaled = cluster_scaler.fit_transform(cluster_data)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(cluster_data_scaled)
        
        # Add cluster labels to original data
        self.df['Cluster'] = clusters
        
        # Analyze clusters
        cluster_analysis = self.df.groupby('Cluster')[cluster_features].mean()
        
        print("✅ Clustering completed!")
        print("\n📊 Cluster Analysis:")
        print(cluster_analysis)
        
        # Visualize clusters
        self.visualize_clusters(cluster_features)
        
        return clusters, cluster_analysis
    
    def visualize_clusters(self, cluster_features):
        """Visualize clustering results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Price vs Mileage colored by cluster
        axes[0, 0].scatter(self.df['ARAI_Certified_Mileage'], self.df['Ex_Showroom_PriceRs.'], 
                          c=self.df['Cluster'], cmap='viridis', alpha=0.6)
        axes[0, 0].set_xlabel('Mileage (km/l)')
        axes[0, 0].set_ylabel('Price (Rs.)')
        axes[0, 0].set_title('Price vs Mileage by Cluster')
        
        # Cylinders vs Price colored by cluster
        axes[0, 1].scatter(self.df['Cylinders'], self.df['Ex_Showroom_PriceRs.'], 
                          c=self.df['Cluster'], cmap='viridis', alpha=0.6)
        axes[0, 1].set_xlabel('Cylinders')
        axes[0, 1].set_ylabel('Price (Rs.)')
        axes[0, 1].set_title('Cylinders vs Price by Cluster')
        
        # Seating Capacity vs Price colored by cluster
        axes[1, 0].scatter(self.df['Seating_Capacity'], self.df['Ex_Showroom_PriceRs.'], 
                          c=self.df['Cluster'], cmap='viridis', alpha=0.6)
        axes[1, 0].set_xlabel('Seating Capacity')
        axes[1, 0].set_ylabel('Price (Rs.)')
        axes[1, 0].set_title('Seating Capacity vs Price by Cluster')
        
        # Cluster distribution
        cluster_counts = self.df['Cluster'].value_counts().sort_index()
        axes[1, 1].bar(cluster_counts.index, cluster_counts.values, color='skyblue')
        axes[1, 1].set_xlabel('Cluster')
        axes[1, 1].set_ylabel('Number of Cars')
        axes[1, 1].set_title('Cluster Distribution')
        
        plt.tight_layout()
        plt.savefig('visualizations/clustering_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def explain_model_predictions(self, sample_size=100):
        """Generate model explanations using SHAP"""
        print("🔄 Generating model explanations with SHAP...")
        
        if self.best_model_name in ['Linear_Regression', 'Ridge_Regression', 'Lasso_Regression']:
            X_explain = self.X_test_scaled[:sample_size]
        else:
            X_explain = self.X_test.iloc[:sample_size]
        
        try:
            if self.best_model_name == 'XGBoost':
                # For XGBoost, use TreeExplainer
                explainer = shap.TreeExplainer(self.best_model)
                shap_values = explainer.shap_values(X_explain)
            elif self.best_model_name == 'Random_Forest':
                # For Random Forest, use TreeExplainer
                explainer = shap.TreeExplainer(self.best_model)
                shap_values = explainer.shap_values(X_explain)
            else:
                # For linear models, use LinearExplainer
                explainer = shap.LinearExplainer(self.best_model, self.X_train_scaled)
                shap_values = explainer.shap_values(X_explain)
            
            # Create SHAP plots
            plt.figure(figsize=(12, 8))
            
            # Summary plot
            if self.best_model_name in ['Linear_Regression', 'Ridge_Regression', 'Lasso_Regression']:
                feature_names = self.feature_names
            else:
                feature_names = self.feature_names
            
            shap.summary_plot(shap_values, X_explain, feature_names=feature_names, show=False)
            plt.title(f'SHAP Summary Plot - {self.best_model_name}')
            plt.tight_layout()
            plt.savefig('visualizations/shap_summary.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Feature importance plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_explain, feature_names=feature_names, 
                            plot_type="bar", show=False)
            plt.title(f'SHAP Feature Importance - {self.best_model_name}')
            plt.tight_layout()
            plt.savefig('visualizations/shap_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("✅ SHAP explanations generated successfully!")
            
        except Exception as e:
            print(f"⚠️ SHAP explanation failed: {e}")
            print("Generating basic feature importance instead...")
            
            if hasattr(self.best_model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': self.best_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                plt.figure(figsize=(10, 6))
                sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
                plt.title(f'Feature Importance - {self.best_model_name}')
                plt.tight_layout()
                plt.savefig('visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
                plt.show()
    
    def explain_with_lime(self, sample_size=5):
        """Generate model explanations using LIME"""
        print("🔄 Generating model explanations with LIME...")
        
        try:
            # Prepare data for LIME
            if self.best_model_name in ['Linear_Regression', 'Ridge_Regression', 'Lasso_Regression']:
                X_explain = self.X_test_scaled[:sample_size]
                X_train_lime = self.X_train_scaled
            else:
                X_explain = self.X_test.iloc[:sample_size]
                X_train_lime = self.X_train
            
            # Create LIME explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train_lime.values,
                feature_names=self.feature_names,
                class_names=['Price'],
                mode='regression',
                discretize_continuous=True
            )
            
            # Generate explanations for sample instances
            explanations = []
            for i in range(min(sample_size, len(X_explain))):
                instance = X_explain.iloc[i].values if hasattr(X_explain, 'iloc') else X_explain[i]
                
                # Create prediction function
                if self.best_model_name in ['Linear_Regression', 'Ridge_Regression', 'Lasso_Regression']:
                    predict_fn = lambda x: self.best_model.predict(x)
                else:
                    predict_fn = lambda x: self.best_model.predict(x)
                
                # Generate explanation
                exp = explainer.explain_instance(
                    instance, 
                    predict_fn, 
                    num_features=len(self.feature_names)
                )
                explanations.append(exp)
                
                # Save individual explanation
                exp.save_to_file(f'visualizations/lime_explanation_{i}.html')
            
            # Create summary plot of LIME explanations
            plt.figure(figsize=(12, 8))
            
            # Collect feature importances from all explanations
            lime_importances = {}
            for exp in explanations:
                for feature, importance in exp.as_list():
                    if feature not in lime_importances:
                        lime_importances[feature] = []
                    lime_importances[feature].append(abs(importance))
            
            # Calculate average absolute importance
            avg_importances = {feature: np.mean(importances) 
                             for feature, importances in lime_importances.items()}
            
            # Sort and plot
            sorted_features = sorted(avg_importances.items(), key=lambda x: x[1], reverse=True)[:10]
            features, importances = zip(*sorted_features)
            
            plt.barh(range(len(features)), importances)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Average Absolute LIME Importance')
            plt.title(f'LIME Feature Importance Summary - {self.best_model_name}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('visualizations/lime_summary.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"✅ LIME explanations generated for {len(explanations)} instances!")
            print("📁 Individual explanations saved as HTML files in visualizations/")
            
        except Exception as e:
            print(f"⚠️ LIME explanation failed: {e}")
            print("LIME requires additional setup. Install with: pip install lime")
    
    def save_models(self):
        """Save trained models and preprocessors"""
        print("🔄 Saving models...")
        
        # Create models directory
        if not os.path.exists('models'):
            os.makedirs('models')
        
        # Save all models
        for name, model in self.models.items():
            joblib.dump(model, f'models/{name.lower()}_model.pkl')
        
        # Save preprocessors
        joblib.dump(self.scaler, 'models/scaler.pkl')
        joblib.dump(self.label_encoders, 'models/label_encoders.pkl')
        
        # Save feature names
        joblib.dump(self.feature_names, 'models/feature_names.pkl')
        
        # Save model scores
        joblib.dump(self.model_scores, 'models/model_scores.pkl')
        
        # Save best model info
        best_model_info = {
            'name': self.best_model_name,
            'model': self.best_model,
            'score': self.model_scores[self.best_model_name]['test_r2']
        }
        joblib.dump(best_model_info, 'models/best_model.pkl')
        
        print("✅ All models saved successfully!")
    
    def create_model_comparison_plot(self, results_df):
        """Create model comparison visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # R² Score comparison
        axes[0, 0].bar(results_df['Model'], results_df['Test_R2'], color='skyblue')
        axes[0, 0].set_title('Model Comparison - R² Score')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # RMSE comparison
        axes[0, 1].bar(results_df['Model'], results_df['Test_RMSE'], color='lightcoral')
        axes[0, 1].set_title('Model Comparison - RMSE')
        axes[0, 1].set_ylabel('RMSE (Rs.)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # MAE comparison
        axes[1, 0].bar(results_df['Model'], results_df['Test_MAE'], color='lightgreen')
        axes[1, 0].set_title('Model Comparison - MAE')
        axes[1, 0].set_ylabel('MAE (Rs.)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Train vs Test R² comparison
        x = np.arange(len(results_df))
        width = 0.35
        axes[1, 1].bar(x - width/2, results_df['Train_R2'], width, label='Train R²', color='orange')
        axes[1, 1].bar(x + width/2, results_df['Test_R2'], width, label='Test R²', color='blue')
        axes[1, 1].set_title('Train vs Test R² Score')
        axes[1, 1].set_ylabel('R² Score')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(results_df['Model'], rotation=45)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function to run the complete ML pipeline"""
    print("🚗 CAR PRICE PREDICTION - MACHINE LEARNING PIPELINE")
    print("=" * 70)
    
    # Initialize model
    model = CarPricePredictionModel()
    
    # Load and prepare data
    X, y = model.load_and_prepare_data()
    
    # Encode categorical features
    X_encoded = model.encode_categorical_features()
    
    # Split and scale data
    X_train, X_test, y_train, y_test = model.split_and_scale_data()
    
    # Train models
    model.train_linear_regression()
    model.train_random_forest()
    model.train_xgboost()
    
    # Evaluate models
    results_df = model.evaluate_models()
    
    # Create model comparison plot
    model.create_model_comparison_plot(results_df)
    
    # Perform clustering
    clusters, cluster_analysis = model.perform_clustering()
    
    # Generate both SHAP and LIME explanations
    model.explain_model_predictions()
    model.explain_with_lime()
        
    print(f"\n🎉 ML Pipeline Complete!")
    print(f"🏆 Best Model: {model.best_model_name}")
    print(f"📊 Best R² Score: {model.model_scores[model.best_model_name]['test_r2']:.4f}")
    print(f"💾 Models saved in 'models/' directory")
    print(f"📈 Visualizations saved in 'visualizations/' directory")
    
    return model, results_df

if __name__ == "__main__":
    model, results = main()
