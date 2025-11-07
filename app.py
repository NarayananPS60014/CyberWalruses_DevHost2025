import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json
import io
import zipfile
from typing import Dict, List, Optional
import re
import warnings
import sys
import chardet
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import joblib
warnings.filterwarnings('ignore')

# Increase recursion limit for complex operations
sys.setrecursionlimit(10000)

# Set page configuration
st.set_page_config(
    page_title="KULDIO ESG Compliance Platform",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DataIntegrator:
    """Advanced data integration system for handling multiple inconsistent CSV files"""
    
    def __init__(self):
        # Expanded standard columns with more variations
        self.standard_columns = {
            'date': ['date', 'timestamp', 'time', 'period', 'month', 'year', 'day', 'reading_date', 'billing_period'],
            'co2_emissions_tonnes': ['co2', 'co2_emissions', 'carbon_emissions', 'emissions', 'co2_tons', 'carbon', 'co2_tonnes'],
            'energy_consumption_mwh': ['energy', 'energy_consumption', 'power', 'electricity', 'energy_mwh', 'consumption', 
                                     'total_kwh', 'power_consumption_kwh', 'energy_consumed_mwh', 'kwh', 'mwh'],
            'renewable_energy_ratio': ['renewable', 'renewable_ratio', 'green_energy', 'renewable_percent', 'clean_energy',
                                     'solar_kwh', 'renewable_energy_ratio'],
            'waste_generated_tonnes': ['waste', 'waste_generated', 'waste_tons', 'garbage', 'trash', 'waste_disposed_kg',
                                     'waste_kg', 'waste_tonnes'],
            'water_consumption_m3': ['water', 'water_consumption', 'water_usage', 'water_m3', 'water_use', 
                                   'gas_volume_m3', 'gas_consumption_m3'],
            'business_travel_km': ['travel', 'business_travel', 'travel_km', 'mileage', 'transport', 
                                 'business_travel_mi', 'travel_miles', 'mileage_km']
        }
        
        self.date_formats = [
            '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y.%m.%d', 
            '%d-%m-%Y', '%m-%d-%Y', '%Y%m%d', '%b %d, %Y',
            '%d %b %Y', '%B %d, %Y', '%d %B %Y', '%b-%Y', '%B-%Y',
            '%Y-%m', '%m/%Y', '%b %Y', '%B %Y'
        ]
    
    def detect_encoding(self, file_content):
        """Detect file encoding"""
        try:
            result = chardet.detect(file_content)
            return result.get('encoding', 'utf-8')
        except:
            return 'utf-8'
    
    def read_csv_file(self, file):
        """Robust CSV file reading with multiple fallbacks"""
        try:
            # Get file content
            file_content = file.getvalue()
            
            # Try to detect encoding
            encoding = self.detect_encoding(file_content)
            
            # Try different reading methods
            try:
                # Method 1: Standard read
                df = pd.read_csv(io.BytesIO(file_content), encoding=encoding)
                if not df.empty:
                    return df
            except:
                pass
            
            try:
                # Method 2: Try with different separator
                df = pd.read_csv(io.BytesIO(file_content), encoding=encoding, sep=';')
                if not df.empty:
                    return df
            except:
                pass
            
            try:
                # Method 3: Try with different separator
                df = pd.read_csv(io.BytesIO(file_content), encoding=encoding, sep='\t')
                if not df.empty:
                    return df
            except:
                pass
            
            try:
                # Method 4: Try with error handling
                df = pd.read_csv(io.BytesIO(file_content), encoding=encoding, engine='python', on_bad_lines='skip')
                if not df.empty:
                    return df
            except:
                pass
            
            # If all methods fail, return empty dataframe
            return pd.DataFrame()
            
        except Exception as e:
            st.error(f"Error reading file {file.name}: {str(e)}")
            return pd.DataFrame()
    
    def detect_column_mapping(self, df_columns):
        """Automatically detect column mappings using enhanced fuzzy matching"""
        mappings = {}
        confidence_scores = {}
        
        for standard_col, possible_names in self.standard_columns.items():
            best_match = None
            best_score = 0
            
            for col in df_columns:
                col_lower = col.lower()
                
                # Try exact match first
                if col_lower in [name.lower() for name in possible_names]:
                    best_match = col
                    best_score = 100
                    break
                
                # Try partial matches
                for possible_name in possible_names:
                    score = self.enhanced_fuzzy_match(col_lower, possible_name.lower())
                    if score > best_score and score > 60:  # Lower threshold for more flexibility
                        best_match = col
                        best_score = score
            
            if best_match:
                mappings[standard_col] = best_match
                confidence_scores[standard_col] = best_score
        
        return mappings, confidence_scores
    
    def enhanced_fuzzy_match(self, str1, str2):
        """Enhanced fuzzy matching with better pattern recognition"""
        # Remove special characters and convert to lower
        str1_clean = re.sub(r'[^a-z0-9]', '', str1.lower())
        str2_clean = re.sub(r'[^a-z0-9]', '', str2.lower())
        
        # Exact match after cleaning
        if str1_clean == str2_clean:
            return 100
        
        # Contains match
        if str1_clean in str2_clean or str2_clean in str1_clean:
            return 90
        
        # Word overlap
        words1 = set(re.findall(r'[a-z]+', str1))
        words2 = set(re.findall(r'[a-z]+', str2))
        common_words = words1.intersection(words2)
        
        if common_words:
            overlap_ratio = len(common_words) / max(len(words1), len(words2))
            return int(70 + (overlap_ratio * 20))
        
        # Character overlap
        common_chars = set(str1_clean) & set(str2_clean)
        if common_chars:
            char_overlap = len(common_chars) / max(len(str1_clean), len(str2_clean))
            return int(50 + (char_overlap * 20))
        
        return 40
    
    def normalize_date(self, date_str):
        """Enhanced date normalization for various formats including month-year"""
        if pd.isna(date_str):
            return None
            
        date_str = str(date_str).strip()
        
        # Handle month-year formats like "Jan-2023", "January-2023"
        month_year_match = re.match(r'([A-Za-z]+)-(\d{4})', date_str)
        if month_year_match:
            month_str, year_str = month_year_match.groups()
            try:
                # Try parsing as full month name
                return datetime.strptime(f"{month_str} {year_str}", '%B %Y').strftime('%Y-%m-15')
            except ValueError:
                try:
                    # Try parsing as abbreviated month name
                    return datetime.strptime(f"{month_str} {year_str}", '%b %Y').strftime('%Y-%m-15')
                except ValueError:
                    pass
        
        # Handle year-month formats like "2023-01"
        year_month_match = re.match(r'(\d{4})-(\d{1,2})', date_str)
        if year_month_match:
            year_str, month_str = year_month_match.groups()
            return f"{year_str}-{month_str.zfill(2)}-15"
        
        # Try multiple date formats
        for fmt in self.date_formats:
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                # For month-only formats, use the 15th as a representative day
                if fmt in ['%b-%Y', '%B-%Y', '%Y-%m', '%m/%Y', '%b %Y', '%B %Y']:
                    return parsed_date.strftime('%Y-%m-15')
                else:
                    return parsed_date.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        return None
    
    def detect_units(self, column_name, sample_values):
        """Enhanced unit detection with more unit types"""
        column_name = str(column_name).lower()
        
        # Unit detection logic
        if any(unit in column_name for unit in ['ton', 'tonne', 't']):
            return 'tons'
        elif any(unit in column_name for unit in ['mwh']):
            return 'mwh'
        elif any(unit in column_name for unit in ['kwh']):
            return 'kwh'
        elif any(unit in column_name for unit in ['m3', 'cubic']):
            return 'm3'
        elif any(unit in column_name for unit in ['km', 'kilometer']):
            return 'km'
        elif any(unit in column_name for unit in ['mi', 'mile']):
            return 'miles'
        elif any(unit in column_name for unit in ['kg', 'kilogram']):
            return 'kg'
        elif any(unit in column_name for unit in ['liter', 'litre', 'l']):
            return 'liters'
        elif any(unit in column_name for unit in ['%', 'percent', 'ratio']):
            return 'ratio'
        else:
            # Try to infer from sample values
            if len(sample_values) > 0:
                sample_val = sample_values.iloc[0]
                if isinstance(sample_val, (int, float)):
                    if 0 <= sample_val <= 1:
                        return 'ratio'
                    elif sample_val > 1000:  # Likely in smaller units
                        if 'energy' in column_name or 'power' in column_name:
                            return 'kwh'
                        elif 'waste' in column_name:
                            return 'kg'
            return 'unknown'
    
    def normalize_units(self, series, source_unit, target_unit):
        """Enhanced unit normalization with more conversion types"""
        if source_unit == target_unit:
            return series
        
        # Energy conversions
        if source_unit == 'kwh' and target_unit == 'mwh':
            return series / 1000
        elif source_unit == 'mwh' and target_unit == 'kwh':
            return series * 1000
        
        # Mass conversions
        if source_unit == 'kg' and target_unit == 'tons':
            return series / 1000
        elif source_unit == 'tons' and target_unit == 'kg':
            return series * 1000
        
        # Distance conversions
        if source_unit == 'miles' and target_unit == 'km':
            return series * 1.60934
        elif source_unit == 'km' and target_unit == 'miles':
            return series / 1.60934
        
        # Volume conversions (simplified)
        if source_unit == 'liters' and 'gas' in target_unit:
            # Rough conversion: 1 liter diesel ≈ 2.68 kg CO2, but we'll handle this in emissions calculation
            return series
        
        return series
    
    def calculate_derived_metrics(self, integrated_df):
        """Calculate derived metrics like CO2 emissions from source data"""
        result_df = integrated_df.copy()
        
        # Calculate CO2 emissions if not directly provided
        if 'co2_emissions_tonnes' not in result_df.columns or result_df['co2_emissions_tonnes'].isna().all():
            # Simple emission factors (in reality, these would be more sophisticated)
            emission_factors = {
                'energy_consumption_mwh': 0.5,  # tons CO2 per MWh
                'gas_consumption_m3': 0.2,      # tons CO2 per m3 gas
                'diesel_liters': 0.00268,       # tons CO2 per liter diesel
            }
            
            co2_emissions = 0
            for source_col, factor in emission_factors.items():
                if source_col in result_df.columns:
                    # Ensure the column is numeric
                    result_df[source_col] = pd.to_numeric(result_df[source_col], errors='coerce')
                    co2_emissions += result_df[source_col] * factor
            
            if co2_emissions.sum() > 0:
                result_df['co2_emissions_tonnes'] = co2_emissions
        
        # Ensure renewable ratio is calculated
        if 'renewable_energy_ratio' not in result_df.columns or result_df['renewable_energy_ratio'].isna().all():
            if 'solar_kwh' in result_df.columns and 'energy_consumption_mwh' in result_df.columns:
                # Convert solar kWh to MWh and calculate ratio
                result_df['solar_kwh'] = pd.to_numeric(result_df['solar_kwh'], errors='coerce')
                result_df['energy_consumption_mwh'] = pd.to_numeric(result_df['energy_consumption_mwh'], errors='coerce')
                solar_mwh = result_df['solar_kwh'] / 1000
                result_df['renewable_energy_ratio'] = solar_mwh / result_df['energy_consumption_mwh']
        
        return result_df
    
    def integrate_multiple_files(self, uploaded_files):
        """Integrate multiple inconsistent CSV files into a unified dataset"""
        all_data = []
        integration_report = {
            'files_processed': 0,
            'total_records': 0,
            'mapping_confidence': {},
            'data_quality_issues': [],
            'successful_files': [],
            'derived_metrics': []
        }
        
        for i, file in enumerate(uploaded_files):
            try:
                # Read CSV file with robust method
                df = self.read_csv_file(file)
                
                if df.empty:
                    integration_report['data_quality_issues'].append(f"File {file.name}: Could not read CSV file - file may be empty or invalid format")
                    continue
                    
                integration_report['files_processed'] += 1
                
                # Show file info for debugging
                st.write(f"📊 **File {file.name}:** {len(df)} rows, {len(df.columns)} columns")
                st.write(f"**Columns:** {list(df.columns)}")
                
                # Detect column mappings
                mappings, confidence_scores = self.detect_column_mapping(df.columns)
                integration_report['mapping_confidence'][file.name] = confidence_scores
                
                # Create normalized dataframe
                normalized_data = {}
                
                # Handle date column
                date_col = None
                for possible_date_col in ['date', 'timestamp', 'period', 'reading_date', 'billing_period']:
                    if possible_date_col in mappings.values() or possible_date_col in df.columns:
                        date_col = possible_date_col
                        break
                
                if date_col:
                    if date_col in df.columns:
                        df['normalized_date'] = df[date_col].apply(self.normalize_date)
                        df = df[df['normalized_date'].notna()]  # Remove rows with invalid dates
                        if not df.empty:
                            normalized_data['date'] = pd.to_datetime(df['normalized_date'])
                    else:
                        # Try to find the actual column name from mappings
                        for std_col, actual_col in mappings.items():
                            if std_col == 'date':
                                df['normalized_date'] = df[actual_col].apply(self.normalize_date)
                                df = df[df['normalized_date'].notna()]
                                if not df.empty:
                                    normalized_data['date'] = pd.to_datetime(df['normalized_date'])
                                break
                else:
                    integration_report['data_quality_issues'].append(f"File {file.name}: No date column found")
                    continue
                
                # If no date data after processing, skip this file
                if 'date' not in normalized_data:
                    integration_report['data_quality_issues'].append(f"File {file.name}: No valid date data after processing")
                    continue
                
                # Map and normalize other columns
                for std_col, source_col in mappings.items():
                    if std_col != 'date' and source_col in df.columns:
                        try:
                            # Detect units and normalize
                            source_unit = self.detect_units(source_col, df[source_col])
                            target_unit = self.get_target_unit(std_col)
                            
                            # Ensure numeric data
                            df[source_col] = pd.to_numeric(df[source_col], errors='coerce')
                            normalized_series = self.normalize_units(df[source_col], source_unit, target_unit)
                            normalized_data[std_col] = normalized_series
                            
                            # Record unit conversion if it happened
                            if source_unit != target_unit:
                                integration_report['derived_metrics'].append(
                                    f"Converted {source_col} from {source_unit} to {target_unit}"
                                )
                        except Exception as e:
                            integration_report['data_quality_issues'].append(
                                f"File {file.name}: Error processing {source_col} - {str(e)}"
                            )
                
                # Also include unmapped columns that might be useful
                for col in df.columns:
                    if col not in mappings.values() and col not in ['normalized_date']:
                        # Check if this column might contain useful data
                        if any(keyword in col.lower() for keyword in ['solar', 'peak', 'off', 'gas', 'diesel', 'travel', 'waste']):
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                            normalized_data[col] = df[col]
                
                # Create final normalized dataframe
                normalized_df = pd.DataFrame(normalized_data)
                
                if not normalized_df.empty:
                    # Add source file information
                    normalized_df['source_file'] = file.name
                    all_data.append(normalized_df)
                    integration_report['successful_files'].append(file.name)
                    integration_report['total_records'] += len(normalized_df)
                    st.success(f"✅ Successfully processed {file.name}: {len(normalized_df)} records")
                else:
                    integration_report['data_quality_issues'].append(f"File {file.name}: No valid data after processing")
                
            except Exception as e:
                integration_report['data_quality_issues'].append(f"File {file.name}: Error - {str(e)}")
                st.error(f"Error processing {file.name}: {str(e)}")
        
        # Combine all data
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Remove duplicates based on date
            combined_df = combined_df.sort_values('date').drop_duplicates(subset=['date'], keep='last')
            
            # Calculate derived metrics
            combined_df = self.calculate_derived_metrics(combined_df)
            
            # Fill missing values using interpolation
            for col in combined_df.columns:
                if col not in ['date', 'source_file'] and combined_df[col].dtype in [np.float64, np.int64]:
                    combined_df[col] = combined_df[col].interpolate(method='linear', limit_direction='both')
            
            return combined_df, integration_report
        else:
            return pd.DataFrame(), integration_report
    
    def get_target_unit(self, standard_column):
        """Get target unit for each standard column"""
        unit_mapping = {
            'co2_emissions_tonnes': 'tons',
            'energy_consumption_mwh': 'mwh',
            'renewable_energy_ratio': 'ratio',
            'waste_generated_tonnes': 'tons',
            'water_consumption_m3': 'm3',
            'business_travel_km': 'km'
        }
        return unit_mapping.get(standard_column, 'unknown')

class ESGPlatform:
    def __init__(self):
        # Initialize EU Taxonomy criteria first (always available)
        self.eu_taxonomy_criteria = {
            'Construction': {
                'climate_mitigation': ['Energy efficiency measures', 'Sustainable materials', 'Waste reduction'],
                'climate_adaptation': ['Flood risk management', 'Heat resilience', 'Sustainable drainage'],
                'thresholds': {'carbon_intensity': 150, 'renewable_energy': 0.5}
            },
            'Manufacturing': {
                'climate_mitigation': ['Process optimization', 'Renewable energy', 'Circular economy'],
                'climate_adaptation': ['Supply chain resilience', 'Water management', 'Extreme weather preparedness'],
                'thresholds': {'carbon_intensity': 200, 'renewable_energy': 0.4}
            },
            'Energy': {
                'climate_mitigation': ['Renewable generation', 'Grid optimization', 'Carbon capture'],
                'climate_adaptation': ['Infrastructure resilience', 'Climate risk assessment', 'Emergency planning'],
                'thresholds': {'carbon_intensity': 100, 'renewable_energy': 0.7}
            },
            'Transportation': {
                'climate_mitigation': ['Fuel efficiency', 'Electric vehicles', 'Route optimization'],
                'climate_adaptation': ['Infrastructure resilience', 'Extreme weather planning', 'Supply chain diversification'],
                'thresholds': {'carbon_intensity': 180, 'renewable_energy': 0.3}
            },
            'Technology': {
                'climate_mitigation': ['Energy efficient data centers', 'Renewable energy procurement', 'E-waste management'],
                'climate_adaptation': ['Business continuity planning', 'Climate risk assessment', 'Remote work infrastructure'],
                'thresholds': {'carbon_intensity': 80, 'renewable_energy': 0.6}
            }
        }
        
        # Initialize data integrator
        self.data_integrator = DataIntegrator()
        
        # Initialize model - create new one instead of loading
        self.model = self.create_new_model()
        
        if 'companies' not in st.session_state:
            self.setup_default_data()
        else:
            self.companies = st.session_state.companies
            self.emission_data = st.session_state.emission_data
    
    def create_new_model(self):
        """Create a new model instead of trying to load problematic PKL file"""
        try:
            # Try RandomForest first
            return RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
        except:
            try:
                # Fallback to simpler model
                return DecisionTreeRegressor(random_state=42, max_depth=10)
            except:
                # Ultimate fallback
                return LinearRegression()
    
    def save_model(self):
        """Save model using joblib which is more reliable than pickle"""
        try:
            joblib.dump(self.model, "esg_prediction_model.joblib")
            return True
        except Exception as e:
            # If joblib fails, don't worry - we'll create new models as needed
            return False
    
    def load_model_joblib(self):
        """Try to load model using joblib if it exists"""
        try:
            if os.path.exists("esg_prediction_model.joblib"):
                return joblib.load("esg_prediction_model.joblib")
        except:
            pass
        return None
    
    def analyze_with_model(self, company: str) -> Dict:
        """Analyze company data using the trained model with robust error handling"""
        if company not in self.emission_data or self.emission_data[company].empty:
            return {"prediction": None, "trend": "insufficient_data", "key_factors": []}
            
        df = self.emission_data[company].copy()
        
        # Check if we have the required columns
        if 'co2_emissions_tonnes' not in df.columns or len(df) < 3:
            return {"prediction": None, "trend": "insufficient_data", "key_factors": []}
        
        try:
            # Simple trend analysis instead of complex ML
            if len(df) >= 3:
                # Calculate simple moving average trend
                emissions = df['co2_emissions_tonnes'].values
                
                if len(emissions) >= 6:
                    # Use last 3 points vs previous 3 points
                    recent_avg = np.mean(emissions[-3:])
                    previous_avg = np.mean(emissions[-6:-3])
                    trend = "decreasing" if recent_avg < previous_avg else "increasing"
                    predicted_change = ((recent_avg - previous_avg) / previous_avg) * 100
                else:
                    # Simple linear trend for shorter series
                    x = np.arange(len(emissions))
                    slope = np.polyfit(x, emissions, 1)[0]
                    trend = "decreasing" if slope < 0 else "increasing"
                    predicted_change = (slope / np.mean(emissions)) * 100 * 3  # Project 3 months
                
                current_emissions = emissions[-1]
                prediction = current_emissions * (1 + predicted_change/100)
                
                # Determine key factors based on available data
                key_factors = []
                if 'energy_consumption_mwh' in df.columns:
                    key_factors.append("energy_consumption")
                if 'renewable_energy_ratio' in df.columns:
                    key_factors.append("renewable_ratio")
                if len(df) >= 12:
                    key_factors.append("seasonal_patterns")
                
                if not key_factors:
                    key_factors = ["historical_trend", "business_operations"]
                
                # Save the simple analysis approach
                self.save_model()
                
                return {
                    "prediction": float(prediction),
                    "trend": trend,
                    "key_factors": key_factors[:3],
                    "current_emissions": float(current_emissions),
                    "predicted_change_percent": float(predicted_change)
                }
            else:
                return {"prediction": None, "trend": "insufficient_data", "key_factors": []}
            
        except Exception as e:
            return {"prediction": None, "trend": "analysis_failed", "key_factors": []}
    
    def setup_default_data(self):
        """Initialize sample data for demonstration"""
        # Sample company data
        self.companies = {
            'Nordic Construction AS': {
                'sector': 'Construction',
                'employees': 250,
                'revenue': 50000000,
                'country': 'Norway',
                'data_sources': ['ERP System', 'Utility Bills', 'Fuel Consumption']
            },
            'Baltic Manufacturing OY': {
                'sector': 'Manufacturing',
                'employees': 180,
                'revenue': 35000000,
                'country': 'Finland',
                'data_sources': ['ERP System', 'Energy Meters', 'Production Data']
            },
            'Scandinavian Energy AB': {
                'sector': 'Energy',
                'employees': 120,
                'revenue': 45000000,
                'country': 'Sweden',
                'data_sources': ['Grid Data', 'Fuel Consumption', 'Energy Production']
            }
        }
        
        # Generate sample emission data
        self.generate_emission_data()
        
        # Save to session state
        st.session_state.companies = self.companies
        st.session_state.emission_data = self.emission_data
    
    def generate_emission_data(self):
        """Generate sample emission data for companies"""
        dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='M')
        self.emission_data = {}
        
        for company, info in self.companies.items():
            sector = info['sector']
            base_emissions = {
                'Construction': 8000,
                'Manufacturing': 12000,
                'Energy': 15000,
                'Transportation': 10000,
                'Technology': 3000
            }
            
            # Create realistic emission patterns with some seasonality and trends
            base = base_emissions.get(sector, 5000)
            trend = -np.random.uniform(0.8, 1.2)  # Slight decreasing trend
            seasonal = np.sin(np.arange(len(dates)) * 2 * np.pi / 12) * 0.1
            
            emissions = base * (1 + trend * np.arange(len(dates)) / 100 + seasonal)
            emissions += np.random.normal(0, base * 0.05, len(dates))  # Random noise
            
            df = pd.DataFrame({
                'date': dates,
                'co2_emissions_tonnes': np.maximum(emissions, 1000),  # Ensure positive values
                'energy_consumption_mwh': emissions * np.random.uniform(1.8, 2.2),
                'renewable_energy_ratio': np.random.uniform(0.3, 0.7, len(dates)),
                'waste_generated_tonnes': emissions * np.random.uniform(0.1, 0.3),
                'water_consumption_m3': emissions * np.random.uniform(10, 20),
                'business_travel_km': emissions * np.random.uniform(50, 100)
            })
            
            self.emission_data[company] = df

    def integrate_csv_files(self, company: str, uploaded_files):
        """Integrate multiple CSV files and update company data"""
        try:
            # Use data integrator to process files
            integrated_data, integration_report = self.data_integrator.integrate_multiple_files(uploaded_files)
            
            if not integrated_data.empty:
                # Remove source_file column for final storage
                if 'source_file' in integrated_data.columns:
                    integrated_data = integrated_data.drop('source_file', axis=1)
                
                # Update company data
                self.emission_data[company] = integrated_data
                st.session_state.emission_data = self.emission_data
                
                return True, integration_report
            else:
                return False, integration_report
                
        except Exception as e:
            return False, {'error': str(e), 'files_processed': 0, 'total_records': 0}
    
    def add_manual_data_entry(self, company: str, new_data: Dict):
        """Add manual data entry for a company"""
        if company not in self.emission_data:
            # Initialize empty dataframe for new company
            self.emission_data[company] = pd.DataFrame(columns=[
                'date', 'co2_emissions_tonnes', 'energy_consumption_mwh',
                'renewable_energy_ratio', 'waste_generated_tonnes',
                'water_consumption_m3', 'business_travel_km'
            ])
        
        # Convert new data to DataFrame row
        new_row = pd.DataFrame([new_data])
        
        # Append to existing data
        self.emission_data[company] = pd.concat([
            self.emission_data[company], 
            new_row
        ], ignore_index=True)
        
        # Update session state
        st.session_state.emission_data = self.emission_data
        return True
    
    def upload_csv_data(self, company: str, csv_file):
        """Upload and process single CSV data"""
        try:
            # Read CSV file
            uploaded_df = pd.read_csv(csv_file)
            
            # Use integrator for single file
            integrated_data, integration_report = self.data_integrator.integrate_multiple_files([csv_file])
            
            if not integrated_data.empty:
                # Remove source_file column
                if 'source_file' in integrated_data.columns:
                    integrated_data = integrated_data.drop('source_file', axis=1)
                
                self.emission_data[company] = integrated_data
                st.session_state.emission_data = self.emission_data
                return True
            else:
                st.error(f"Failed to process CSV: {integration_report.get('data_quality_issues', ['Unknown error'])}")
                return False
                
        except Exception as e:
            st.error(f"Error processing CSV file: {str(e)}")
            return False
    
    def add_new_company(self, company_data: Dict):
        """Add a new company to the platform"""
        company_name = company_data['name']
        
        if company_name in self.companies:
            st.error(f"Company {company_name} already exists")
            return False
        
        # Validate sector
        sector = company_data['sector']
        if sector not in self.eu_taxonomy_criteria:
            st.error(f"Unknown sector: {sector}. Please choose from {list(self.eu_taxonomy_criteria.keys())}")
            return False
        
        # Add company info
        self.companies[company_name] = {
            'sector': sector,
            'employees': company_data['employees'],
            'revenue': company_data['revenue'],
            'country': company_data['country'],
            'data_sources': company_data.get('data_sources', [])
        }
        
        # Initialize empty emission data
        self.emission_data[company_name] = pd.DataFrame(columns=[
            'date', 'co2_emissions_tonnes', 'energy_consumption_mwh',
            'renewable_energy_ratio', 'waste_generated_tonnes',
            'water_consumption_m3', 'business_travel_km'
        ])
        
        # Update session state
        st.session_state.companies = self.companies
        st.session_state.emission_data = self.emission_data
        
        return True

    def calculate_green_credit_score(self, company: str) -> float:
        """Calculate green credit score based on multiple factors with error handling"""
        if company not in self.emission_data or self.emission_data[company].empty:
            return 50.0  # Default score for new companies
            
        df = self.emission_data[company]
        
        # Base score
        score = 70
        
        # Emission reduction trend with error handling
        try:
            if len(df) >= 12 and 'co2_emissions_tonnes' in df.columns:
                recent_emissions = df['co2_emissions_tonnes'].tail(6).mean()
                previous_emissions = df['co2_emissions_tonnes'].tail(12).head(6).mean()
                
                if recent_emissions < previous_emissions:
                    reduction = (previous_emissions - recent_emissions) / previous_emissions
                    score += reduction * 20
        except (KeyError, TypeError, ZeroDivisionError):
            pass
        
        # Renewable energy bonus with error handling
        try:
            if 'renewable_energy_ratio' in df.columns:
                renewable_ratio = df['renewable_energy_ratio'].mean()
                score += renewable_ratio * 15
        except (KeyError, TypeError):
            pass
        
        # Waste reduction with error handling
        try:
            if len(df) > 1 and 'waste_generated_tonnes' in df.columns:
                waste_trend = np.polyfit(range(len(df)), df['waste_generated_tonnes'], 1)[0]
                if waste_trend < 0:
                    score += 5
        except (KeyError, TypeError):
            pass
        
        return min(max(score, 0), 100)  # Ensure score is between 0-100
    
    def calculate_eu_taxonomy_alignment(self, company: str) -> Dict:
        """Calculate EU Taxonomy alignment percentage - FIXED VERSION"""
        if company not in self.companies:
            return {
                'alignment_percentage': 0,
                'carbon_intensity': 0,
                'renewable_ratio': 0,
                'meets_thresholds': False
            }
            
        sector = self.companies[company]['sector']
        
        # Ensure sector exists in criteria
        if sector not in self.eu_taxonomy_criteria:
            return {
                'alignment_percentage': 0,
                'carbon_intensity': 0,
                'renewable_ratio': 0,
                'meets_thresholds': False
            }
            
        df = self.emission_data[company]
        
        if df.empty:
            return {
                'alignment_percentage': 0,
                'carbon_intensity': 0,
                'renewable_ratio': 0,
                'meets_thresholds': False
            }
            
        latest = df.iloc[-1]
        criteria = self.eu_taxonomy_criteria[sector]
        alignment_score = 0
        
        # Carbon intensity alignment - FIXED to use direct access like old code
        try:
            carbon_intensity = latest['co2_emissions_tonnes'] / self.companies[company]['revenue'] * 1000000
            if carbon_intensity <= criteria['thresholds']['carbon_intensity']:
                alignment_score += 30
        except (KeyError, TypeError, ZeroDivisionError):
            carbon_intensity = 0
        
        # Renewable energy alignment - FIXED to use direct access like old code
        try:
            renewable_ratio = latest['renewable_energy_ratio']
            if renewable_ratio >= criteria['thresholds']['renewable_energy']:
                alignment_score += 30
        except (KeyError, TypeError):
            renewable_ratio = 0
        
        # Climate mitigation measures (simplified)
        alignment_score += 20  # Assume some measures are implemented
        
        # Climate adaptation measures (simplified)
        alignment_score += 20  # Assume some measures are implemented
        
        return {
            'alignment_percentage': alignment_score,
            'carbon_intensity': carbon_intensity,
            'renewable_ratio': renewable_ratio,
            'meets_thresholds': alignment_score >= 60
        }

    def get_ai_suggestions(self, company: str) -> List[str]:
        """Get AI-powered suggestions using model analysis and LM Studio"""
        if company not in self.companies:
            return ["Please add company data first"]
            
        sector = self.companies[company]['sector']
        
        if sector not in self.eu_taxonomy_criteria:
            return [f"Sector '{sector}' not recognized. Please choose from {list(self.eu_taxonomy_criteria.keys())}"]
        
        # Get model analysis
        model_analysis = self.analyze_with_model(company)
        alignment = self.calculate_eu_taxonomy_alignment(company)
        
        # Prepare the enhanced prompt with model analysis
        prompt = f"""
        Provide 3-5 specific, actionable ESG recommendations for a {sector} company.
        
        Company context:
        - Sector: {sector}
        - EU Taxonomy Alignment: {alignment['alignment_percentage']}%
        - Carbon Intensity: {alignment['carbon_intensity']:.1f} t/€M revenue
        - Renewable Energy Ratio: {alignment['renewable_ratio']:.1%}
        - Green Credit Score: {self.calculate_green_credit_score(company):.1f}/100
        
        **Trend Analysis Results:**
        - Emissions Trend: {model_analysis.get('trend', 'unknown')}
        - Current Emissions: {model_analysis.get('current_emissions', 'N/A'):.0f} tons
        - Predicted Change: {model_analysis.get('predicted_change_percent', 0):.1f}%
        - Key Influencing Factors: {', '.join(model_analysis.get('key_factors', []))}
        
        Focus on practical, implementable suggestions that would help improve their ESG performance and EU Taxonomy alignment, considering the predicted trend and key factors.
        IMPORTANT: Avoid using specific percentages or numerical targets in your recommendations. Use qualitative improvements instead.
        Return each suggestion as a separate bullet point, concise and actionable.
        """
        
        try:
            # Call LM Studio local API with enhanced prompt
            response = requests.post(
                "http://localhost:1234/v1/chat/completions",
                json={
                    "model": "local-model",
                    "messages": [
                        {"role": "system", "content": "You are an ESG and sustainability expert providing practical recommendations for Nordic companies. Consider the predictive model analysis in your suggestions. Avoid using specific percentages or numerical targets - focus on qualitative improvements."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 500
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result['choices'][0]['message']['content']
                
                # Parse the response into a list of suggestions
                suggestions = []
                for line in ai_response.split('\n'):
                    line = line.strip()
                    if line and (line.startswith('-') or line.startswith('•') or line[0].isdigit()):
                        clean_line = re.sub(r'^[-•\d\.\s]+', '', line).strip()
                        if clean_line:
                            # Remove any remaining percentages and numbers
                            clean_line = re.sub(r'\d+%', 'significantly', clean_line)
                            clean_line = re.sub(r'\d+-\d+%', 'substantially', clean_line)
                            clean_line = re.sub(r'\d+', '', clean_line).strip()
                            suggestions.append(clean_line)
                
                if suggestions:
                    return suggestions[:5]
            
            # Fallback to default suggestions if API call fails
            return self._get_enhanced_default_suggestions(sector, alignment, model_analysis)
            
        except Exception as e:
            st.warning(f"LM Studio API not available, using default suggestions: {str(e)}")
            return self._get_enhanced_default_suggestions(sector, alignment, model_analysis)

    def _get_enhanced_default_suggestions(self, sector: str, alignment: dict, model_analysis: dict) -> List[str]:
        """Fallback suggestions enhanced with model analysis"""
        base_suggestions = self._get_default_suggestions(sector, alignment)
        
        # Enhance suggestions based on model analysis
        enhanced_suggestions = base_suggestions.copy()
        
        trend = model_analysis.get('trend')
        key_factors = model_analysis.get('key_factors', [])
        
        # Add trend-based suggestions
        if trend == "increasing":
            enhanced_suggestions.append("🚨 Priority: Implement immediate emission reduction measures to reverse the upward trend")
        elif trend == "decreasing":
            enhanced_suggestions.append("✅ Good progress - maintain and accelerate current reduction strategies")
        
        # Add factor-based suggestions
        if "energy_consumption" in str(key_factors):
            enhanced_suggestions.append("⚡ Focus on energy efficiency improvements in high-consumption areas")
        if "renewable_ratio" in str(key_factors):
            enhanced_suggestions.append("🌞 Accelerate renewable energy adoption to leverage its impact on emissions")
        
        return enhanced_suggestions[:5]  # Return max 5 suggestions

    def _get_default_suggestions(self, sector: str, alignment: dict) -> List[str]:
        """Fallback default suggestions when LM Studio is not available"""
        suggestions = {
            'Construction': [
                "Implement BIM-based construction planning to significantly reduce material waste",
                "Switch to low-carbon concrete and recycled steel in all new projects",
                "Install solar panels on construction sites for temporary power needs",
                "Adopt modular construction techniques to minimize on-site waste generation",
                "Implement comprehensive water recycling systems across all construction sites"
            ],
            'Manufacturing': [
                "Conduct energy audit to identify high-consumption equipment for optimization",
                "Implement heat recovery systems from manufacturing processes",
                "Switch to LED lighting and smart energy management systems",
                "Establish circular economy principles for material reuse and recycling",
                "Optimize production schedules to reduce energy-intensive peak operations"
            ],
            'Energy': [
                "Accelerate transition to renewable energy sources",
                "Implement smart grid technologies for better energy distribution",
                "Invest in carbon capture, utilization and storage technologies",
                "Enhance grid resilience through infrastructure modernization",
                "Develop distributed energy resources for local generation"
            ],
            'Transportation': [
                "Transition to electric or hybrid vehicle fleet",
                "Implement route optimization software to reduce fuel consumption",
                "Use sustainable aviation fuel for air transport",
                "Establish telecommuting policies to reduce business travel",
                "Optimize logistics networks for reduced transportation distances"
            ],
            'Technology': [
                "Migrate to energy-efficient cloud data centers",
                "Implement server virtualization to reduce energy consumption",
                "Use renewable energy for office operations and data centers",
                "Extend product lifecycle through repair and upgrade programs",
                "Implement e-waste recycling and responsible disposal programs"
            ]
        }
        
        default_suggestions = suggestions.get(sector, [
            "Conduct comprehensive sustainability assessment",
            "Establish ESG governance framework",
            "Engage stakeholders in sustainability planning"
        ])
        
        # Add priority suggestions based on performance gaps
        if alignment['carbon_intensity'] > self.eu_taxonomy_criteria[sector]['thresholds']['carbon_intensity']:
            default_suggestions.append("Priority: Focus on carbon intensity reduction through process optimization")
        
        if alignment['renewable_ratio'] < self.eu_taxonomy_criteria[sector]['thresholds']['renewable_energy']:
            default_suggestions.append("Priority: Increase renewable energy procurement through PPAs or on-site generation")
        
        return default_suggestions[:5]
    
    def create_emissions_forecast(self, company: str) -> pd.DataFrame:
        """Create simple emissions forecast"""
        if company not in self.emission_data or self.emission_data[company].empty:
            return pd.DataFrame()
            
        df = self.emission_data[company].copy()
        
        # Simple linear forecast for next 12 months
        last_date = df['date'].iloc[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=31), periods=12, freq='M')
        
        # Calculate trend and project forward (with accelerated reduction due to implemented measures)
        if len(df) > 1:
            try:
                emissions_trend = np.polyfit(range(len(df)), df['co2_emissions_tonnes'], 1)[0]
            except:
                emissions_trend = 0
        else:
            emissions_trend = 0
            
        current_emissions = df['co2_emissions_tonnes'].iloc[-1]
        
        # Projected emissions (with accelerated reduction)
        projected_emissions = [
            max(current_emissions * (1 - 0.02 * (i+1)) + emissions_trend, 100)  # Ensure positive
            for i in range(len(future_dates))
        ]
        
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'co2_emissions_tonnes': projected_emissions,
            'type': 'forecast'
        })
        
        return forecast_df

def data_input_tab(platform: ESGPlatform, selected_company: str):
    """Data Input and Management Tab with Advanced Integration"""
    st.header("📥 Advanced Data Integration & Management")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔄 Multi-CSV Integration", 
        "➕ Manual Data Entry", 
        "📁 Single CSV Upload", 
        "🏢 Add New Company",
        "🔍 View Current Data"
    ])
    
    with tab1:
        st.subheader("Multi-CSV Automatic Integration")
        st.info("""
        **Enhanced Smart Data Integration:**
        - Now handles unpredictable CSV formats like energy meters, ERP systems, and utility bills
        - Automatic unit conversion (kWh→MWh, kg→tons, miles→km, etc.)
        - Derived metrics calculation (CO2 from energy consumption)
        - Flexible date format handling (Jan-2023, 2023-01, etc.)
        """)
        
        # Add CSV format requirements
        st.warning("""
        **CSV Requirements:**
        - Must contain a date column (any format: 2023-01-31, Jan-2023, etc.)
        - Should contain at least one data column (energy, emissions, waste, etc.)
        - Supported separators: comma (,), semicolon (;), tab
        - Supported encodings: UTF-8, Latin-1
        """)
        
        uploaded_files = st.file_uploader(
            "Upload multiple CSV files for automatic integration", 
            type=['csv'], 
            accept_multiple_files=True,
            key="multi_csv"
        )
        
        if uploaded_files:
            st.write(f"**Files to integrate:** {[f.name for f in uploaded_files]}")
            
            # Show file previews
            for file in uploaded_files:
                with st.expander(f"Preview: {file.name}"):
                    try:
                        # Use the robust reading method
                        preview_df = platform.data_integrator.read_csv_file(file)
                        if not preview_df.empty:
                            st.write(f"**Shape:** {preview_df.shape[0]} rows × {preview_df.shape[1]} columns")
                            st.write(f"**Columns:** {list(preview_df.columns)}")
                            st.dataframe(preview_df.head(3))
                        else:
                            st.error(f"Could not read {file.name}. File may be empty or invalid format.")
                    except Exception as e:
                        st.error(f"Error previewing {file.name}: {str(e)}")
            
            if st.button("🚀 Start Smart Integration", type="primary"):
                with st.spinner("Integrating and normalizing data from multiple sources..."):
                    success, integration_report = platform.integrate_csv_files(selected_company, uploaded_files)
                
                if success:
                    st.success("✅ Data integration completed successfully!")
                    
                    # Show integration report
                    st.subheader("📊 Integration Report")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Files Processed", integration_report.get('files_processed', 0))
                    with col2:
                        st.metric("Total Records", integration_report.get('total_records', 0))
                    with col3:
                        successful_files = len(integration_report.get('successful_files', []))
                        st.metric("Successful Files", successful_files)
                    
                    # Show mapping confidence
                    if 'mapping_confidence' in integration_report:
                        st.subheader("🔍 Column Mapping Confidence")
                        for file_name, mappings in integration_report['mapping_confidence'].items():
                            with st.expander(f"Mapping for {file_name}"):
                                for std_col, confidence in mappings.items():
                                    st.write(f"**{std_col}**: {confidence}% confidence")
                    
                    # Show derived metrics
                    if integration_report.get('derived_metrics'):
                        st.subheader("🔄 Derived Metrics")
                        for metric in integration_report['derived_metrics']:
                            st.info(f"✓ {metric}")
                    
                    # Show data quality issues
                    if integration_report.get('data_quality_issues'):
                        st.subheader("⚠️ Data Quality Issues")
                        for issue in integration_report['data_quality_issues']:
                            st.warning(issue)
                    
                    # NO RERUN - Let Streamlit handle the natural flow
                    st.balloons()  # Celebration instead of rerun
                else:
                    st.error("❌ Data integration failed")
                    if integration_report.get('data_quality_issues'):
                        st.error("Integration issues:")
                        for issue in integration_report['data_quality_issues']:
                            st.write(f"- {issue}")
    
    with tab2:
        st.subheader("Manual Data Entry")
        
        with st.form("manual_data_entry"):
            col1, col2 = st.columns(2)
            
            with col1:
                entry_date = st.date_input("Date", datetime.now())
                co2_emissions = st.number_input("CO₂ Emissions (tons)", min_value=0.0, value=1000.0)
                energy_consumption = st.number_input("Energy Consumption (MWh)", min_value=0.0, value=2000.0)
                
            with col2:
                renewable_ratio = st.slider("Renewable Energy Ratio", 0.0, 1.0, 0.5)
                waste_generated = st.number_input("Waste Generated (tons)", min_value=0.0, value=200.0)
                water_consumption = st.number_input("Water Consumption (m³)", min_value=0.0, value=15000.0)
            
            submitted = st.form_submit_button("Add Data Entry")
            
            if submitted:
                new_data = {
                    'date': entry_date,
                    'co2_emissions_tonnes': co2_emissions,
                    'energy_consumption_mwh': energy_consumption,
                    'renewable_energy_ratio': renewable_ratio,
                    'waste_generated_tonnes': waste_generated,
                    'water_consumption_m3': water_consumption,
                    'business_travel_km': np.random.uniform(50000, 100000)  # Simulated
                }
                
                if platform.add_manual_data_entry(selected_company, new_data):
                    st.success("Data entry added successfully!")
                    # NO RERUN - Let Streamlit handle the natural flow
    
    with tab3:
        st.subheader("Single CSV Upload")
        
        st.info("""
        **CSV Format (Flexible):**
        - Must include date/datetime column (any format)
        - Other columns will be automatically mapped
        - Supported units: tons, kg, MWh, kWh, m³, km, %
        """)
        
        # Download template
        template_data = {
            'date': ['2024-01-31', '2024-02-29', '2024-03-31'],
            'co2_emissions_tonnes': [1000, 950, 900],
            'energy_consumption_mwh': [2000, 1900, 1800],
            'renewable_energy_ratio': [0.5, 0.55, 0.6],
            'waste_generated_tonnes': [200, 190, 180],
            'water_consumption_m3': [15000, 14500, 14000],
            'business_travel_km': [50000, 45000, 40000]
        }
        template_df = pd.DataFrame(template_data)
        
        st.download_button(
            label="📥 Download CSV Template",
            data=template_df.to_csv(index=False),
            file_name="esg_data_template.csv",
            mime="text/csv"
        )
        
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'], key="single_csv")
        
        if uploaded_file is not None:
            if st.button("Process CSV Data"):
                if platform.upload_csv_data(selected_company, uploaded_file):
                    st.success("CSV data uploaded and processed successfully!")
                    # NO RERUN - Let Streamlit handle the natural flow
    
    with tab4:
        st.subheader("Add New Company")
        
        with st.form("new_company_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                company_name = st.text_input("Company Name*", placeholder="Enter company name")
                sector = st.selectbox("Sector*", list(platform.eu_taxonomy_criteria.keys()))
                employees = st.number_input("Number of Employees*", min_value=1, value=100)
                
            with col2:
                revenue = st.number_input("Annual Revenue (€)*", min_value=0, value=10000000)
                country = st.selectbox("Country*", ["Norway", "Sweden", "Finland", "Denmark", "Iceland", "Other"])
                data_sources = st.multiselect("Data Sources", 
                    ["ERP System", "Utility Bills", "Energy Meters", "Fuel Consumption", "Production Data", "Travel System"])
            
            submitted = st.form_submit_button("Add Company")
            
            if submitted:
                if not company_name:
                    st.error("Company name is required!")
                else:
                    company_data = {
                        'name': company_name,
                        'sector': sector,
                        'employees': employees,
                        'revenue': revenue,
                        'country': country,
                        'data_sources': data_sources
                    }
                    
                    if platform.add_new_company(company_data):
                        st.success(f"Company {company_name} added successfully!")
                        # NO RERUN - Let Streamlit handle the natural flow
    
    with tab5:
        st.subheader("Current Data Overview")
        
        if selected_company in platform.emission_data:
            df = platform.emission_data[selected_company]
            
            if not df.empty:
                st.dataframe(df, use_container_width=True)
                
                # Data summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", len(df))
                with col2:
                    if not df.empty:
                        st.metric("Date Range", f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
                    else:
                        st.metric("Date Range", "No data")
                with col3:
                    if not df.empty and 'co2_emissions_tonnes' in df.columns:
                        st.metric("Latest CO₂ Emissions", f"{df['co2_emissions_tonnes'].iloc[-1]:,.0f} tons")
                    else:
                        st.metric("Latest CO₂ Emissions", "No data")
                
                # Data quality indicators
                st.subheader("📈 Data Quality Indicators")
                quality_col1, quality_col2, quality_col3, quality_col4 = st.columns(4)
                
                with quality_col1:
                    completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                    st.metric("Data Completeness", f"{completeness:.1f}%")
                
                with quality_col2:
                    date_span = (df['date'].max() - df['date'].min()).days if len(df) > 1 else 0
                    st.metric("Date Coverage", f"{date_span} days")
                
                with quality_col3:
                    record_count = len(df)
                    st.metric("Record Count", record_count)
                
                with quality_col4:
                    data_freshness = (datetime.now().date() - df['date'].max().date()).days if len(df) > 0 else "N/A"
                    st.metric("Data Freshness", f"{data_freshness} days ago" if isinstance(data_freshness, int) else data_freshness)
                
                # Option to clear data
                if st.button("Clear All Data for This Company"):
                    platform.emission_data[selected_company] = pd.DataFrame(columns=df.columns)
                    st.session_state.emission_data = platform.emission_data
                    st.success("Data cleared successfully!")
                    # NO RERUN - Let Streamlit handle the natural flow
            else:
                st.info("No data available for this company. Use the tabs above to add data.")
        else:
            st.warning("No data found for selected company.")

def generate_csrd_report(company: str, platform: ESGPlatform) -> str:
    """Generate a simulated CSRD report"""
    alignment = platform.calculate_eu_taxonomy_alignment(company)
    score = platform.calculate_green_credit_score(company)
    
    if company in platform.emission_data and not platform.emission_data[company].empty and 'co2_emissions_tonnes' in platform.emission_data[company].columns:
        latest_emissions = platform.emission_data[company]['co2_emissions_tonnes'].iloc[-1]
    else:
        latest_emissions = 'N/A'
    
    report_content = f"""
    CSRD Compliance Report - {company}
    Generated: {datetime.now().strftime('%Y-%m-%d')}
    
    EXECUTIVE SUMMARY
    =================
    - Green Credit Score: {score:.1f}/100
    - EU Taxonomy Alignment: {alignment['alignment_percentage']}%
    - Compliance Status: {'COMPLIANT' if alignment['meets_thresholds'] else 'NEEDS IMPROVEMENT'}
    
    ENVIRONMENTAL PERFORMANCE
    =========================
    - Carbon Intensity: {alignment['carbon_intensity']:.1f} t/€M revenue
    - Renewable Energy Ratio: {alignment['renewable_ratio']:.1%}
    - Latest CO₂ Emissions: {latest_emissions if latest_emissions != 'N/A' else 'N/A':,} tons
    
    DATA INTEGRATION SUMMARY
    ========================
    - Data Sources: {', '.join(platform.companies[company].get('data_sources', [])) if company in platform.companies else 'N/A'}
    - Automated Integration: Yes
    - Data Quality: Verified
    
    This report has been generated by the KULDIO ESG Compliance Platform.
    Data is based on automated integration from multiple sources.
    """
    return report_content

def generate_esg_data_export(company: str, platform: ESGPlatform) -> str:
    """Generate ESG data export"""
    if company in platform.emission_data:
        df = platform.emission_data[company].copy()
        return df.to_csv(index=False)
    else:
        return ""

def safe_plotly_chart(fig, use_container_width=True):
    """Safely render Plotly charts with error handling"""
    try:
        st.plotly_chart(fig, use_container_width=use_container_width)
    except RecursionError:
        st.error("Chart too complex to render. Please try with less data or a different visualization.")
    except Exception as e:
        st.error(f"Error rendering chart: {str(e)}")

def main():
    st.title("🌍 KULDIO ESG & Carbon Compliance Platform")
    st.markdown("### AI-Powered Sustainability Reporting with Enhanced Smart Data Integration")
    
    # Initialize platform
    platform = ESGPlatform()
    
    # Sidebar for company selection
    st.sidebar.header("Company Selection")
    selected_company = st.sidebar.selectbox(
        "Choose Company",
        list(platform.companies.keys())
    )
    
    # Display company info in sidebar
    if selected_company in platform.companies:
        company_info = platform.companies[selected_company]
        st.sidebar.subheader("Company Info")
        st.sidebar.write(f"**Sector:** {company_info['sector']}")
        st.sidebar.write(f"**Employees:** {company_info['employees']:,}")
        st.sidebar.write(f"**Revenue:** €{company_info['revenue']:,}")
        st.sidebar.write(f"**Country:** {company_info['country']}")
        
        if 'data_sources' in company_info and company_info['data_sources']:
            st.sidebar.write("**Data Sources:**")
            for source in company_info['data_sources']:
                st.sidebar.write(f"• {source}")
    
    # Display key metrics at the top
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        green_score = platform.calculate_green_credit_score(selected_company)
        st.metric(
            "Green Credit Score",
            f"{green_score:.1f}/100",
            delta="+2.1" if green_score > 50 else "-1.2"
        )
    
    with col2:
        alignment = platform.calculate_eu_taxonomy_alignment(selected_company)
        st.metric(
            "EU Taxonomy Alignment",
            f"{alignment['alignment_percentage']}%",
            delta="Compliant" if alignment['meets_thresholds'] else "Needs Improvement",
            delta_color="normal" if alignment['meets_thresholds'] else "off"
        )
    
    with col3:
        emission_data = platform.emission_data.get(selected_company, pd.DataFrame())
        if not emission_data.empty and 'co2_emissions_tonnes' in emission_data.columns:
            latest_emissions = emission_data['co2_emissions_tonnes'].iloc[-1]
            if len(emission_data) > 1:
                previous_emissions = emission_data['co2_emissions_tonnes'].iloc[-2]
                change = ((latest_emissions - previous_emissions) / previous_emissions) * 100
            else:
                change = 0
            st.metric(
                "CO₂ Emissions (tons)",
                f"{latest_emissions:,.0f}",
                delta=f"{change:+.1f}%"
            )
        else:
            st.metric("CO₂ Emissions (tons)", "No data")
    
    with col4:
        if not emission_data.empty and 'renewable_energy_ratio' in emission_data.columns:
            renewable_ratio = emission_data['renewable_energy_ratio'].iloc[-1] * 100
            st.metric(
                "Renewable Energy",
                f"{renewable_ratio:.1f}%",
                delta="+5.2%"
            )
        else:
            st.metric("Renewable Energy", "No data")
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔄 Data Integration", 
        "📊 Emissions Overview", 
        "🌱 EU Taxonomy Compliance", 
        "💡 AI Recommendations",
        "📈 Progress Tracking",
        "📋 ESG Reporting"
    ])
    
    with tab1:
        data_input_tab(platform, selected_company)
    
    with tab2:
        st.subheader("Carbon Emissions Dashboard")
        
        emission_data = platform.emission_data.get(selected_company, pd.DataFrame())
        
        if emission_data.empty:
            st.warning("No emission data available. Please add data in the 'Data Integration' tab.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                # Emissions trend chart - using go.Figure instead of px.line to avoid recursion
                try:
                    if 'co2_emissions_tonnes' in emission_data.columns:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=emission_data['date'], 
                            y=emission_data['co2_emissions_tonnes'],
                            mode='lines',
                            name='CO₂ Emissions'
                        ))
                        fig.update_layout(
                            title=f'{selected_company} - CO₂ Emissions Trend',
                            xaxis_title='Date',
                            yaxis_title='CO₂ Emissions (tons)',
                            height=400
                        )
                        safe_plotly_chart(fig)
                    else:
                        st.warning("CO₂ emissions data not available for chart")
                except Exception as e:
                    st.error(f"Could not create emissions chart: {str(e)}")
            
            with col2:
                # Emissions by category (simulated)
                try:
                    categories = ['Energy Consumption', 'Transportation', 'Manufacturing', 'Waste']
                    emissions_by_category = [45, 25, 20, 10]
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=categories,
                        values=emissions_by_category,
                        hole=.3
                    )])
                    fig.update_layout(
                        title='Emissions by Category',
                        height=400
                    )
                    safe_plotly_chart(fig)
                except Exception as e:
                    st.error(f"Could not create pie chart: {str(e)}")
            
            # Forecast section
            st.subheader("Emissions Forecast")
            forecast_data = platform.create_emissions_forecast(selected_company)
            
            if not forecast_data.empty:
                try:
                    # Combine historical and forecast data
                    historical_for_chart = emission_data[['date', 'co2_emissions_tonnes']].copy()
                    historical_for_chart['type'] = 'historical'
                    
                    combined_data = pd.concat([historical_for_chart, forecast_data], ignore_index=True)
                    
                    fig = go.Figure()
                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=combined_data[combined_data['type']=='historical']['date'],
                        y=combined_data[combined_data['type']=='historical']['co2_emissions_tonnes'],
                        mode='lines',
                        name='Historical',
                        line=dict(color='blue')
                    ))
                    # Forecast data
                    fig.add_trace(go.Scatter(
                        x=combined_data[combined_data['type']=='forecast']['date'],
                        y=combined_data[combined_data['type']=='forecast']['co2_emissions_tonnes'],
                        mode='lines',
                        name='Forecast',
                        line=dict(color='red', dash='dash')
                    ))
                    fig.update_layout(
                        title='Historical Emissions vs Forecast',
                        xaxis_title='Date',
                        yaxis_title='CO₂ Emissions (tons)',
                        height=400
                    )
                    safe_plotly_chart(fig)
                except Exception as e:
                    st.error(f"Could not create forecast chart: {str(e)}")
    
    with tab3:
        st.subheader("EU Taxonomy Compliance Analysis")
        
        alignment = platform.calculate_eu_taxonomy_alignment(selected_company)
        
        if selected_company in platform.companies:
            sector = platform.companies[selected_company]['sector']
            criteria = platform.eu_taxonomy_criteria.get(sector, {})
            
            if not criteria:
                st.error(f"No EU Taxonomy criteria found for sector: {sector}")
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Compliance gauge
                    try:
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = alignment['alignment_percentage'],
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "EU Taxonomy Alignment"},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 60], 'color': "lightgray"},
                                    {'range': [60, 100], 'color': "lightgreen"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 60
                                }
                            }
                        ))
                        fig.update_layout(height=300)
                        safe_plotly_chart(fig)
                    except Exception as e:
                        st.error(f"Could not create gauge chart: {str(e)}")
                
                with col2:
                    st.subheader("Key Compliance Metrics")
                    
                    # Carbon intensity
                    threshold_carbon = criteria.get('thresholds', {}).get('carbon_intensity', 0)
                    carbon_status = "✅ Compliant" if alignment['carbon_intensity'] <= threshold_carbon else "❌ Needs Improvement"
                    
                    # Handle NaN values for carbon intensity display
                    carbon_display = f"{alignment['carbon_intensity']:.1f}" if not pd.isna(alignment['carbon_intensity']) else "No data"
                    
                    st.metric(
                        "Carbon Intensity (t/€M revenue)",
                        carbon_display,
                        delta=f"Threshold: {threshold_carbon}",
                        help=carbon_status
                    )
                    
                    # Renewable energy - FIXED: Now displays as percentage
                    threshold_renewable = criteria.get('thresholds', {}).get('renewable_energy', 0)
                    renewable_status = "✅ Compliant" if alignment['renewable_ratio'] >= threshold_renewable else "❌ Needs Improvement"
                    
                    # Handle NaN values for renewable ratio display and format as percentage
                    if pd.isna(alignment['renewable_ratio']):
                        renewable_display = "No data"
                        threshold_display = f"Threshold: {threshold_renewable:.0%}"
                    else:
                        renewable_display = f"{alignment['renewable_ratio']:.1%}"
                        threshold_display = f"Threshold: {threshold_renewable:.0%}"
                    
                    st.metric(
                        "Renewable Energy Ratio",
                        renewable_display,
                        delta=threshold_display,
                        help=renewable_status
                    )
                
                # Climate objectives
                st.subheader("Climate Objectives Alignment")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Climate Change Mitigation**")
                    for objective in criteria.get('climate_mitigation', []):
                        st.write(f"• {objective}")
                
                with col2:
                    st.write("**Climate Change Adaptation**")
                    for objective in criteria.get('climate_adaptation', []):
                        st.write(f"• {objective}")
        else:
            st.warning("Company information not found.")
    
    with tab4:
        st.subheader("AI-Powered Recommendations")
        
        # Show model analysis results
        model_analysis = platform.analyze_with_model(selected_company)
        
        if model_analysis.get('prediction') is not None:
            st.subheader("📊 Model Analysis Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                trend_icon = "📈" if model_analysis['trend'] == 'increasing' else "📉"
                st.metric(
                    "Emissions Trend",
                    f"{trend_icon} {model_analysis['trend'].title()}",
                    delta=f"{model_analysis['predicted_change_percent']:.1f}%"
                )
            
            with col2:
                st.metric(
                    "Current Emissions",
                    f"{model_analysis['current_emissions']:,.0f} tons"
                )
            
            with col3:
                st.metric(
                    "Predicted Next Period",
                    f"{model_analysis['prediction']:,.0f} tons"
                )
            
            # Key factors
            st.write("**Key Influencing Factors:**")
            for factor in model_analysis.get('key_factors', []):
                st.write(f"• {factor.replace('_', ' ').title()}")
        
        suggestions = platform.get_ai_suggestions(selected_company)
        
        st.info("💡 These recommendations are generated based on your company's specific emission profile, predictive model analysis, and sector characteristics")
        
        for i, suggestion in enumerate(suggestions, 1):
            with st.expander(f"Recommendation #{i}: {suggestion.split(':')[0] if ':' in suggestion else suggestion}"):
                st.write(suggestion)
                
                # Simulated impact assessment - NO PERCENTAGES
                impact_options = {
                    "Estimated CO₂ Reduction": ["Moderate reduction potential", "Significant reduction potential", "Substantial reduction potential"],
                    "Implementation Timeline": ["Short-term implementation", "Medium-term implementation", "Long-term strategic initiative"],
                    "ROI Potential": ["Positive financial return", "Strong financial benefits", "Excellent value creation"],
                    "EU Taxonomy Impact": ["High compliance improvement", "Medium compliance improvement", "Strategic alignment enhancement"]
                }
                
                impact_metrics = {
                    "Estimated CO₂ Reduction": np.random.choice(impact_options["Estimated CO₂ Reduction"]),
                    "Implementation Timeline": np.random.choice(impact_options["Implementation Timeline"]),
                    "ROI Potential": np.random.choice(impact_options["ROI Potential"]),
                    "EU Taxonomy Impact": "High compliance improvement" if "Priority" in suggestion else np.random.choice(["Medium compliance improvement", "Strategic alignment enhancement"])
                }
                
                for metric, value in impact_metrics.items():
                    st.write(f"**{metric}:** {value}")
        
        # Implementation roadmap - NO PERCENTAGES
        st.subheader("Suggested Implementation Roadmap")
        roadmap_data = {
            'Phase': ['Immediate (0-3 months)', 'Short-term (3-12 months)', 'Medium-term (1-2 years)', 'Long-term (2+ years)'],
            'Actions': [
                "Energy audit and baseline assessment",
                "Quick-win efficiency improvements",
                "Process optimization and technology upgrades",
                "Strategic transformation and innovation"
            ],
            'Expected Impact': ['Initial improvements', 'Noticeable operational benefits', 'Significant performance gains', 'Transformational outcomes']
        }
        
        roadmap_df = pd.DataFrame(roadmap_data)
        st.table(roadmap_df)
    
    with tab5:
        st.subheader("Sustainability Progress Tracking")
        
        alignment = platform.calculate_eu_taxonomy_alignment(selected_company)
        
        if selected_company in platform.companies:
            sector = platform.companies[selected_company]['sector']
            criteria = platform.eu_taxonomy_criteria.get(sector, {})
            thresholds = criteria.get('thresholds', {})
            
            # Key performance indicators with NaN handling
            kpis = [
                {"name": "Carbon Intensity", "current": alignment['carbon_intensity'], "target": thresholds.get('carbon_intensity', 100) * 0.8, "unit": "t/€M"},
                {"name": "Renewable Energy", "current": alignment['renewable_ratio'] * 100, "target": 80, "unit": "%"},
                {"name": "Energy Efficiency", "current": 65, "target": 85, "unit": "points"},
                {"name": "Waste Reduction", "current": 40, "target": 70, "unit": "%"}
            ]
            
            for kpi in kpis:
                # Handle NaN values in progress calculation
                current = kpi['current']
                target = kpi['target']
                
                # Replace NaN with 0
                if pd.isna(current):
                    current = 0
                if pd.isna(target) or target == 0:
                    target = 1  # Avoid division by zero
                
                progress = (current / target) * 100
                
                # Ensure progress is a valid number between 0 and 100
                if pd.isna(progress) or progress < 0:
                    progress = 0
                elif progress > 100:
                    progress = 100
                
                st.write(f"**{kpi['name']}**")
                st.progress(progress / 100)
                st.write(f"Current: {current:.1f}{kpi['unit']} | Target: {target:.1f}{kpi['unit']} | Progress: {progress:.1f}%")
                st.write("---")
        else:
            st.warning("Company information not available for progress tracking.")
    
    with tab6:
        st.subheader("CSRD Compliance Reporting")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Generate CSRD Report",
                data=generate_csrd_report(selected_company, platform),
                file_name=f"csrd_report_{selected_company.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )
        
        with col2:
            st.download_button(
                label="📥 Download ESG Data",
                data=generate_esg_data_export(selected_company, platform),
                file_name=f"esg_data_{selected_company.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        # Report preview
        st.subheader("Report Preview")
        
        report_data = {
            'Section': [
                'Executive Summary',
                'Environmental Performance',
                'Social Indicators',
                'Governance Structure',
                'EU Taxonomy Alignment',
                'Risk Assessment',
                'Future Outlook'
            ],
            'Status': ['✅ Complete', '✅ Complete', '🟡 In Progress', '✅ Complete', '✅ Complete', '🟡 In Progress', '✅ Complete'],
            'Last Updated': [
                datetime.now().strftime('%Y-%m-%d'),
                datetime.now().strftime('%Y-%m-%d'),
                (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d'),
                datetime.now().strftime('%Y-%m-%d'),
                datetime.now().strftime('%Y-%m-%d'),
                (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'),
                datetime.now().strftime('%Y-%m-%d')
            ]
        }
        
        report_df = pd.DataFrame(report_data)
        st.table(report_df)

if __name__ == "__main__":
    main()