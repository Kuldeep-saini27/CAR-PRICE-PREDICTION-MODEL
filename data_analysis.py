import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class CarDataAnalyzer:
    def __init__(self, data_path):
        """Initialize the analyzer with the dataset path"""
        self.data_path = data_path
        self.df = None
        self.df_cleaned = None
        
    def load_data(self):
        """Load and display basic information about the dataset"""
        print("🚗 Loading Car Dataset...")
        self.df = pd.read_csv(self.data_path)
        
        print(f"\n📊 Dataset Shape: {self.df.shape}")
        print(f"📋 Columns: {list(self.df.columns)}")
        
        # Display first few rows
        print("\n🔍 First 5 rows:")
        print(self.df.head())
        
        # Basic info
        print("\n📈 Dataset Info:")
        print(self.df.info())
        
        # Statistical summary
        print("\n📊 Statistical Summary:")
        print(self.df.describe())
        
        return self.df
    
    def check_data_quality(self):
        """Analyze data quality issues"""
        print("\n🔍 DATA QUALITY ANALYSIS")
        print("=" * 50)
        
        # Missing values
        missing_values = self.df.isnull().sum()
        missing_percent = (missing_values / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_values.index,
            'Missing_Count': missing_values.values,
            'Missing_Percentage': missing_percent.values
        }).sort_values('Missing_Count', ascending=False)
        
        print("📋 Missing Values Analysis:")
        print(missing_df[missing_df['Missing_Count'] > 0])
        
        # Duplicate rows
        duplicates = self.df.duplicated().sum()
        print(f"\n🔄 Duplicate Rows: {duplicates}")
        
        # Data types
        print("\n📊 Data Types:")
        print(self.df.dtypes)
        
        # Unique values in categorical columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        print("\n🏷️ Unique Values in Categorical Columns:")
        for col in categorical_cols:
            print(f"{col}: {self.df[col].nunique()} unique values")
            if self.df[col].nunique() < 20:
                print(f"  Values: {sorted(self.df[col].unique())}")
        
        return missing_df
    
    def clean_data(self):
        """Clean and preprocess the dataset"""
        print("\n🧹 DATA CLEANING")
        print("=" * 50)
        
        self.df_cleaned = self.df.copy()
        
        # Clean column names
        self.df_cleaned.columns = self.df_cleaned.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('-', '_')
        
        # Clean price column - remove commas and convert to numeric
        price_col = 'Ex_Showroom_PriceRs.'
        if price_col in self.df_cleaned.columns:
            self.df_cleaned[price_col] = pd.to_numeric(self.df_cleaned[price_col], errors='coerce')
        
        print(f"✅ Cleaned column names: {list(self.df_cleaned.columns)}")
        
        # Clean mileage column - handle ranges and convert to numeric
        mileage_col = 'ARAI_Certified_Mileage'
        if mileage_col in self.df_cleaned.columns:
            # Handle ranges like "9.8-10.0"
            self.df_cleaned[mileage_col] = self.df_cleaned[mileage_col].astype(str)
            self.df_cleaned[mileage_col] = self.df_cleaned[mileage_col].str.replace(' ', '')
            
            # For ranges, take the average
            def clean_mileage(x):
                if '-' in str(x):
                    try:
                        parts = str(x).split('-')
                        return (float(parts[0]) + float(parts[1])) / 2
                    except:
                        return np.nan
                else:
                    try:
                        return float(x)
                    except:
                        return np.nan
            
            self.df_cleaned[mileage_col] = self.df_cleaned[mileage_col].apply(clean_mileage)
        
        # Handle missing values
        # For numerical columns, fill with median
        numerical_cols = self.df_cleaned.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if self.df_cleaned[col].isnull().sum() > 0:
                median_val = self.df_cleaned[col].median()
                self.df_cleaned[col].fillna(median_val, inplace=True)
                print(f"✅ Filled {col} missing values with median: {median_val}")
        
        # For categorical columns, fill with mode
        categorical_cols = self.df_cleaned.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.df_cleaned[col].isnull().sum() > 0:
                mode_val = self.df_cleaned[col].mode()[0] if len(self.df_cleaned[col].mode()) > 0 else 'Unknown'
                self.df_cleaned[col].fillna(mode_val, inplace=True)
                print(f"✅ Filled {col} missing values with mode: {mode_val}")
        
        # Remove duplicates
        initial_rows = len(self.df_cleaned)
        self.df_cleaned.drop_duplicates(inplace=True)
        final_rows = len(self.df_cleaned)
        print(f"✅ Removed {initial_rows - final_rows} duplicate rows")
        
        print(f"\n📊 Cleaned Dataset Shape: {self.df_cleaned.shape}")
        return self.df_cleaned
    
    def detect_outliers(self, column):
        """Detect outliers using IQR method"""
        Q1 = self.df_cleaned[column].quantile(0.25)
        Q3 = self.df_cleaned[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = self.df_cleaned[(self.df_cleaned[column] < lower_bound) | 
                                  (self.df_cleaned[column] > upper_bound)]
        return outliers, lower_bound, upper_bound
    
    def exploratory_data_analysis(self):
        """Perform comprehensive EDA"""
        print("\n📊 EXPLORATORY DATA ANALYSIS")
        print("=" * 50)
        
        # Create visualizations directory
        import os
        if not os.path.exists('visualizations'):
            os.makedirs('visualizations')
        
        # 1. Price Distribution
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.hist(self.df_cleaned['Ex_Showroom_PriceRs.'], bins=50, alpha=0.7, color='skyblue')
        plt.title('Price Distribution')
        plt.xlabel('Price (Rs.)')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 3, 2)
        plt.hist(np.log1p(self.df_cleaned['Ex_Showroom_PriceRs.']), bins=50, alpha=0.7, color='lightgreen')
        plt.title('Log Price Distribution')
        plt.xlabel('Log(Price + 1)')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 3, 3)
        plt.boxplot(self.df_cleaned['Ex_Showroom_PriceRs.'])
        plt.title('Price Box Plot')
        plt.ylabel('Price (Rs.)')
        
        plt.tight_layout()
        plt.savefig('visualizations/price_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Brand Analysis
        plt.figure(figsize=(15, 8))
        brand_counts = self.df_cleaned['Brand'].value_counts().head(15)
        
        plt.subplot(2, 2, 1)
        brand_counts.plot(kind='bar', color='coral')
        plt.title('Top 15 Brands by Count')
        plt.xlabel('Brand')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 2)
        brand_price = self.df_cleaned.groupby('Brand')['Ex_Showroom_PriceRs.'].mean().sort_values(ascending=False).head(15)
        brand_price.plot(kind='bar', color='lightblue')
        plt.title('Top 15 Brands by Average Price')
        plt.xlabel('Brand')
        plt.ylabel('Average Price (Rs.)')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 3)
        fuel_counts = self.df_cleaned['Fuel_Type'].value_counts()
        plt.pie(fuel_counts.values, labels=fuel_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Fuel Type Distribution')
        
        plt.subplot(2, 2, 4)
        body_counts = self.df_cleaned['Body_Type'].value_counts().head(10)
        body_counts.plot(kind='bar', color='gold')
        plt.title('Top 10 Body Types')
        plt.xlabel('Body Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('visualizations/categorical_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Correlation Analysis
        plt.figure(figsize=(12, 10))
        numerical_cols = self.df_cleaned.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df_cleaned[numerical_cols].corr()
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix of Numerical Features')
        plt.tight_layout()
        plt.savefig('visualizations/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Feature vs Price Analysis
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Mileage vs Price
        axes[0, 0].scatter(self.df_cleaned['ARAI_Certified_Mileage'], 
                          self.df_cleaned['Ex_Showroom_PriceRs.'], alpha=0.6)
        axes[0, 0].set_xlabel('Mileage (km/l)')
        axes[0, 0].set_ylabel('Price (Rs.)')
        axes[0, 0].set_title('Mileage vs Price')
        
        # Cylinders vs Price
        cylinder_price = self.df_cleaned.groupby('Cylinders')['Ex_Showroom_PriceRs.'].mean()
        axes[0, 1].bar(cylinder_price.index, cylinder_price.values, color='lightcoral')
        axes[0, 1].set_xlabel('Number of Cylinders')
        axes[0, 1].set_ylabel('Average Price (Rs.)')
        axes[0, 1].set_title('Cylinders vs Average Price')
        
        # Seating Capacity vs Price
        seating_price = self.df_cleaned.groupby('Seating_Capacity')['Ex_Showroom_PriceRs.'].mean()
        axes[0, 2].bar(seating_price.index, seating_price.values, color='lightgreen')
        axes[0, 2].set_xlabel('Seating Capacity')
        axes[0, 2].set_ylabel('Average Price (Rs.)')
        axes[0, 2].set_title('Seating Capacity vs Average Price')
        
        # Fuel Type vs Price
        fuel_price = self.df_cleaned.groupby('Fuel_Type')['Ex_Showroom_PriceRs.'].mean().sort_values(ascending=False)
        axes[1, 0].bar(range(len(fuel_price)), fuel_price.values, color='gold')
        axes[1, 0].set_xticks(range(len(fuel_price)))
        axes[1, 0].set_xticklabels(fuel_price.index, rotation=45)
        axes[1, 0].set_ylabel('Average Price (Rs.)')
        axes[1, 0].set_title('Fuel Type vs Average Price')
        
        # Transmission Type vs Price
        transmission_price = self.df_cleaned.groupby('Type')['Ex_Showroom_PriceRs.'].mean()
        axes[1, 1].bar(transmission_price.index, transmission_price.values, color='skyblue')
        axes[1, 1].set_xlabel('Transmission Type')
        axes[1, 1].set_ylabel('Average Price (Rs.)')
        axes[1, 1].set_title('Transmission vs Average Price')
        
        # Boot Space vs Price
        axes[1, 2].scatter(self.df_cleaned['Boot_Space'], 
                          self.df_cleaned['Ex_Showroom_PriceRs.'], alpha=0.6, color='purple')
        axes[1, 2].set_xlabel('Boot Space (L)')
        axes[1, 2].set_ylabel('Price (Rs.)')
        axes[1, 2].set_title('Boot Space vs Price')
        
        plt.tight_layout()
        plt.savefig('visualizations/feature_price_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 5. Outlier Analysis
        print("\n🔍 OUTLIER ANALYSIS")
        print("=" * 30)
        
        numerical_features = ['Ex_Showroom_PriceRs.', 'ARAI_Certified_Mileage', 'Cylinders', 'Fuel_Capacity']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, feature in enumerate(numerical_features):
            if feature in self.df_cleaned.columns:
                outliers, lower, upper = self.detect_outliers(feature)
                
                axes[i].boxplot(self.df_cleaned[feature])
                axes[i].set_title(f'{feature} - Outliers: {len(outliers)}')
                axes[i].set_ylabel(feature)
                
                print(f"{feature}: {len(outliers)} outliers detected")
                print(f"  Range: [{lower:.2f}, {upper:.2f}]")
        
        plt.tight_layout()
        plt.savefig('visualizations/outlier_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return self.df_cleaned
    
    def generate_insights(self):
        """Generate key insights from the analysis"""
        print("\n💡 KEY INSIGHTS")
        print("=" * 50)
        
        # Price insights
        avg_price = self.df_cleaned['Ex_Showroom_PriceRs.'].mean()
        median_price = self.df_cleaned['Ex_Showroom_PriceRs.'].median()
        max_price = self.df_cleaned['Ex_Showroom_PriceRs.'].max()
        min_price = self.df_cleaned['Ex_Showroom_PriceRs.'].min()
        
        print(f"💰 Price Statistics:")
        print(f"  Average Price: ₹{avg_price:,.0f}")
        print(f"  Median Price: ₹{median_price:,.0f}")
        print(f"  Price Range: ₹{min_price:,.0f} - ₹{max_price:,.0f}")
        
        # Most expensive and cheapest cars
        most_expensive = self.df_cleaned.loc[self.df_cleaned['Ex_Showroom_PriceRs.'].idxmax()]
        cheapest = self.df_cleaned.loc[self.df_cleaned['Ex_Showroom_PriceRs.'].idxmin()]
        
        print(f"\n🏆 Most Expensive: {most_expensive['Brand']} {most_expensive['Model']} - ₹{most_expensive['Ex_Showroom_PriceRs.']:,.0f}")
        print(f"💸 Cheapest: {cheapest['Brand']} {cheapest['Model']} - ₹{cheapest['Ex_Showroom_PriceRs.']:,.0f}")
        
        # Brand insights
        top_brand = self.df_cleaned['Brand'].value_counts().index[0]
        brand_count = self.df_cleaned['Brand'].value_counts().iloc[0]
        
        print(f"\n🚗 Most Popular Brand: {top_brand} ({brand_count} models)")
        
        # Fuel type insights
        fuel_distribution = self.df_cleaned['Fuel_Type'].value_counts()
        print(f"\n⛽ Fuel Type Distribution:")
        for fuel, count in fuel_distribution.items():
            percentage = (count / len(self.df_cleaned)) * 100
            print(f"  {fuel}: {count} cars ({percentage:.1f}%)")
        
        # Mileage insights
        avg_mileage = self.df_cleaned['ARAI_Certified_Mileage'].mean()
        best_mileage = self.df_cleaned.loc[self.df_cleaned['ARAI_Certified_Mileage'].idxmax()]
        
        print(f"\n⚡ Average Mileage: {avg_mileage:.1f} km/l")
        print(f"🏆 Best Mileage: {best_mileage['Brand']} {best_mileage['Model']} - {best_mileage['ARAI_Certified_Mileage']:.1f} km/l")
        
        # Correlation insights
        price_correlations = self.df_cleaned.select_dtypes(include=[np.number]).corr()['Ex_Showroom_PriceRs.'].sort_values(ascending=False)
        print(f"\n📊 Features Most Correlated with Price:")
        for feature, corr in price_correlations.items():
            if feature != 'Ex_Showroom_PriceRs.' and abs(corr) > 0.1:
                print(f"  {feature}: {corr:.3f}")
        
        return {
            'avg_price': avg_price,
            'median_price': median_price,
            'price_range': (min_price, max_price),
            'most_expensive': most_expensive,
            'cheapest': cheapest,
            'top_brand': top_brand,
            'fuel_distribution': fuel_distribution,
            'avg_mileage': avg_mileage,
            'price_correlations': price_correlations
        }

def main():
    """Main function to run the complete analysis"""
    print("🚗 CAR PRICE PREDICTION - DATA ANALYSIS")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = CarDataAnalyzer('Car Dataset - Sheet1.csv')
    
    # Load data
    df = analyzer.load_data()
    
    # Check data quality
    missing_info = analyzer.check_data_quality()
    
    # Clean data
    df_cleaned = analyzer.clean_data()
    
    # Perform EDA
    df_final = analyzer.exploratory_data_analysis()
    
    # Generate insights
    insights = analyzer.generate_insights()
    
    # Save cleaned data
    df_cleaned.to_csv('cleaned_car_data.csv', index=False)
    print(f"\n✅ Cleaned dataset saved as 'cleaned_car_data.csv'")
    
    print(f"\n🎉 Analysis Complete! Check the 'visualizations' folder for charts.")
    
    return df_cleaned, insights

if __name__ == "__main__":
    df_cleaned, insights = main()
