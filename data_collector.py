"""
Data Collector for Vancouver Airbnb Properties
This module collects comprehensive data from various sources
"""

import pandas as pd
import requests
import json
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import re
import numpy as np

class VancouverAirbnbDataCollector:
    def __init__(self):
        self.base_url = "https://www.airbnb.com"
        self.data = []
        
    def setup_driver(self):
        """Setup Chrome driver with appropriate options"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        return driver
    
    def collect_sample_data(self):
        """Create comprehensive sample data for Vancouver Airbnb properties"""
        np.random.seed(42)  # For reproducible results
        
        # Vancouver neighborhoods
        neighborhoods = [
            "Downtown", "West End", "Gastown", "Yaletown", "Kitsilano", 
            "Mount Pleasant", "Commercial Drive", "Main Street", "False Creek",
            "Coal Harbour", "English Bay", "Davie Village", "Chinatown",
            "Olympic Village", "Fairview", "Riley Park", "Sunset", "Renfrew"
        ]
        
        # Property types
        property_types = ["Entire home/apt", "Private room", "Shared room"]
        
        # Room types
        room_types = ["Entire home/apt", "Private room", "Shared room"]
        
        # Bedroom counts
        bedrooms = [0, 1, 2, 3, 4, 5]
        
        # Sample data generation
        sample_data = []
        
        for i in range(2000):  # Generate 2000 sample listings
            # Vancouver coordinates (approximate bounds)
            lat = np.random.uniform(49.2, 49.3)
            lng = np.random.uniform(-123.3, -123.0)
            
            # Price based on location and property type
            base_price = np.random.uniform(80, 400)
            if lat > 49.25:  # Downtown area
                base_price *= 1.3
            if lat > 49.28:  # Very central
                base_price *= 1.2
                
            # Property type affects price
            prop_type = np.random.choice(property_types)
            if prop_type == "Entire home/apt":
                base_price *= 1.5
            elif prop_type == "Shared room":
                base_price *= 0.4
                
            # Bedroom count affects price
            bedroom_count = np.random.choice(bedrooms, p=[0.1, 0.4, 0.3, 0.15, 0.04, 0.01])
            base_price *= (1 + bedroom_count * 0.3)
            
            # Generate realistic data
            listing = {
                'id': f"vancouver_{i+1}",
                'name': f"Beautiful {np.random.choice(['Modern', 'Cozy', 'Stylish', 'Charming', 'Luxury'])} {prop_type.lower()} in {np.random.choice(neighborhoods)}",
                'host_id': f"host_{np.random.randint(1, 500)}",
                'host_name': f"Host_{np.random.randint(1, 500)}",
                'neighbourhood': np.random.choice(neighborhoods),
                'neighbourhood_group': np.random.choice(['Downtown', 'East Side', 'West Side', 'North Vancouver']),
                'latitude': lat,
                'longitude': lng,
                'room_type': prop_type,
                'price': round(base_price, 2),
                'minimum_nights': np.random.choice([1, 2, 3, 7, 14, 30], p=[0.4, 0.3, 0.15, 0.1, 0.04, 0.01]),
                'number_of_reviews': np.random.poisson(25),
                'last_review': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(1, 365)),
                'reviews_per_month': np.random.uniform(0.5, 8),
                'calculated_host_listings_count': np.random.randint(1, 10),
                'availability_365': np.random.randint(50, 365),
                'number_of_reviews_ltm': np.random.poisson(15),
                'license': np.random.choice(['Exempt', 'Pending', 'Licensed', None], p=[0.3, 0.2, 0.4, 0.1]),
                'instant_bookable': np.random.choice([True, False], p=[0.6, 0.4]),
                'host_is_superhost': np.random.choice([True, False], p=[0.3, 0.7]),
                'host_identity_verified': np.random.choice([True, False], p=[0.8, 0.2]),
                'accommodates': bedroom_count + np.random.randint(1, 3),
                'bedrooms': bedroom_count,
                'beds': bedroom_count + np.random.randint(0, 2),
                'bathrooms': max(1, bedroom_count // 2 + np.random.randint(0, 2)),
                'amenities': np.random.choice([
                    "WiFi,Kitchen,Washer,Dryer,Air conditioning,Heating",
                    "WiFi,Kitchen,Parking,Pool,Gym",
                    "WiFi,Kitchen,Pet friendly,Garden",
                    "WiFi,Kitchen,Hot tub,Sauna",
                    "WiFi,Kitchen,Workspace,High chair"
                ]),
                'property_type': np.random.choice(['Apartment', 'House', 'Condominium', 'Townhouse', 'Loft']),
                'cancellation_policy': np.random.choice(['flexible', 'moderate', 'strict', 'super_strict_30']),
                'host_response_rate': np.random.uniform(0.8, 1.0),
                'host_response_time': np.random.choice(['within an hour', 'within a few hours', 'within a day', 'a few days or more']),
                'host_acceptance_rate': np.random.uniform(0.7, 1.0),
                'host_total_listings_count': np.random.randint(1, 20),
                'host_has_profile_pic': np.random.choice([True, False], p=[0.95, 0.05]),
                'host_identity_verified': np.random.choice([True, False], p=[0.8, 0.2]),
                'is_location_exact': np.random.choice([True, False], p=[0.9, 0.1]),
                'review_scores_rating': np.random.uniform(3.5, 5.0),
                'review_scores_accuracy': np.random.uniform(3.5, 5.0),
                'review_scores_cleanliness': np.random.uniform(3.5, 5.0),
                'review_scores_checkin': np.random.uniform(3.5, 5.0),
                'review_scores_communication': np.random.uniform(3.5, 5.0),
                'review_scores_location': np.random.uniform(3.5, 5.0),
                'review_scores_value': np.random.uniform(3.5, 5.0),
                'requires_license': np.random.choice([True, False], p=[0.7, 0.3]),
                'jurisdiction_names': 'Vancouver, BC, Canada',
                'airbnb_url': f"https://www.airbnb.com/rooms/{np.random.randint(1000000, 9999999)}",
                'calculated_days': np.random.randint(30, 365),
                'revenue_ltm': round(base_price * np.random.uniform(0.3, 0.8) * 365, 2),
                'occupancy_rate': np.random.uniform(0.3, 0.9),
                'adr': round(base_price, 2),
                'revenue_per_available_room': round(base_price * np.random.uniform(0.3, 0.8), 2)
            }
            
            sample_data.append(listing)
        
        return pd.DataFrame(sample_data)
    
    def save_data(self, df, filename="vancouver_airbnb_data.csv"):
        """Save data to CSV file"""
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
        return filename

def main():
    """Main function to collect and save data"""
    collector = VancouverAirbnbDataCollector()
    
    print("Generating comprehensive Vancouver Airbnb dataset...")
    df = collector.collect_sample_data()
    
    print(f"Generated {len(df)} listings")
    print(f"Columns: {list(df.columns)}")
    
    # Save the data
    filename = collector.save_data(df)
    
    # Display sample statistics
    print("\nSample Statistics:")
    print(f"Average Price: ${df['price'].mean():.2f}")
    print(f"Average Rating: {df['review_scores_rating'].mean():.2f}")
    print(f"Average Occupancy Rate: {df['occupancy_rate'].mean():.2f}")
    print(f"Average Revenue (LTM): ${df['revenue_ltm'].mean():.2f}")
    
    return df

if __name__ == "__main__":
    main()
