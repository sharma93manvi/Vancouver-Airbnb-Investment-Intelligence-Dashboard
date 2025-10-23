"""
Vancouver Airbnb Data Visualization Dashboard
Comprehensive insights and analysis of Vancouver Airbnb market
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Vancouver Airbnb Analytics Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff7f0e;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the Airbnb data"""
    try:
        df = pd.read_csv('vancouver_airbnb_data.csv')
        # Convert date columns
        df['last_review'] = pd.to_datetime(df['last_review'])
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please run data_collector.py first.")
        return None

def create_price_distribution_chart(df):
    """Create price distribution visualization"""
    fig = px.histogram(
        df, 
        x='price', 
        nbins=50,
        title="Price Distribution",
        labels={'price': 'Price ($)', 'count': 'Number of Listings'},
        color_discrete_sequence=['#1f77b4']
    )
    fig.update_layout(
        xaxis_title="Price ($)",
        yaxis_title="Number of Listings",
        showlegend=False
    )
    return fig

def create_neighborhood_analysis(df):
    """Create neighborhood price analysis"""
    neighborhood_stats = df.groupby('neighbourhood').agg({
        'price': ['mean', 'median', 'count'],
        'review_scores_rating': 'mean',
        'occupancy_rate': 'mean'
    }).round(2)
    
    neighborhood_stats.columns = ['Avg Price', 'Median Price', 'Count', 'Avg Rating', 'Avg Occupancy']
    neighborhood_stats = neighborhood_stats.sort_values('Avg Price', ascending=False)
    
    fig = px.bar(
        neighborhood_stats.reset_index(),
        x='neighbourhood',
        y='Avg Price',
        title="Average Price by Neighborhood",
        labels={'neighbourhood': 'Neighborhood', 'Avg Price': 'Average Price ($)'},
        color='Avg Price',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        height=500
    )
    return fig, neighborhood_stats

def create_property_type_analysis(df):
    """Analyze property types and their characteristics"""
    property_analysis = df.groupby('property_type').agg({
        'price': ['mean', 'count'],
        'review_scores_rating': 'mean',
        'occupancy_rate': 'mean',
        'revenue_ltm': 'mean'
    }).round(2)
    
    property_analysis.columns = ['Avg Price', 'Count', 'Avg Rating', 'Avg Occupancy', 'Avg Revenue']
    
    # Create subplot for property type analysis
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Average Price by Property Type', 'Count by Property Type', 
                       'Average Rating by Property Type', 'Average Revenue by Property Type'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Price analysis
    fig.add_trace(
        go.Bar(x=property_analysis.index, y=property_analysis['Avg Price'], 
               name='Avg Price', marker_color='#1f77b4'),
        row=1, col=1
    )
    
    # Count analysis
    fig.add_trace(
        go.Bar(x=property_analysis.index, y=property_analysis['Count'], 
               name='Count', marker_color='#ff7f0e'),
        row=1, col=2
    )
    
    # Rating analysis
    fig.add_trace(
        go.Bar(x=property_analysis.index, y=property_analysis['Avg Rating'], 
               name='Avg Rating', marker_color='#2ca02c'),
        row=2, col=1
    )
    
    # Revenue analysis
    fig.add_trace(
        go.Bar(x=property_analysis.index, y=property_analysis['Avg Revenue'], 
               name='Avg Revenue', marker_color='#d62728'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False)
    return fig, property_analysis

def create_revenue_analysis(df):
    """Create revenue and occupancy analysis"""
    # Revenue vs Price scatter
    fig1 = px.scatter(
        df, 
        x='price', 
        y='revenue_ltm',
        color='occupancy_rate',
        size='number_of_reviews',
        title="Revenue vs Price (colored by Occupancy Rate)",
        labels={'price': 'Price ($)', 'revenue_ltm': 'Revenue LTM ($)', 'occupancy_rate': 'Occupancy Rate'},
        color_continuous_scale='Viridis'
    )
    
    # Occupancy rate distribution
    fig2 = px.histogram(
        df,
        x='occupancy_rate',
        nbins=30,
        title="Occupancy Rate Distribution",
        labels={'occupancy_rate': 'Occupancy Rate', 'count': 'Number of Listings'},
        color_discrete_sequence=['#ff7f0e']
    )
    
    return fig1, fig2

def create_review_analysis(df):
    """Analyze review scores and ratings"""
    review_cols = ['review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
                   'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value']
    
    review_data = df[review_cols].mean().reset_index()
    review_data.columns = ['Category', 'Average Score']
    
    fig = px.bar(
        review_data,
        x='Category',
        y='Average Score',
        title="Average Review Scores by Category",
        labels={'Category': 'Review Category', 'Average Score': 'Average Score'},
        color='Average Score',
        color_continuous_scale='RdYlGn'
    )
    fig.update_layout(xaxis_tickangle=-45)
    
    return fig

def create_amenity_analysis(df):
    """Analyze amenities and their impact"""
    # Split amenities and count frequency
    all_amenities = []
    for amenities in df['amenities'].dropna():
        all_amenities.extend(amenities.split(','))
    
    amenity_counts = pd.Series(all_amenities).value_counts().head(15)
    
    fig = px.bar(
        x=amenity_counts.values,
        y=amenity_counts.index,
        orientation='h',
        title="Top 15 Most Common Amenities",
        labels={'x': 'Count', 'y': 'Amenity'},
        color=amenity_counts.values,
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=600)
    
    return fig

@st.cache_data
def create_interactive_map(df):
    """Create interactive map with property locations"""
    # Create base map centered on Vancouver
    m = folium.Map(
        location=[49.2827, -123.1207],  # Vancouver coordinates
        zoom_start=11,
        tiles='OpenStreetMap'
    )
    
    # Use a stable sample instead of random sampling to prevent blinking
    # Take every nth row to get a representative sample
    sample_size = min(500, len(df))
    step = max(1, len(df) // sample_size)
    sample_df = df.iloc[::step].head(sample_size)
    
    # Add property markers
    for idx, row in sample_df.iterrows():
        # Color based on price
        if row['price'] < 200:
            color = 'green'
        elif row['price'] < 400:
            color = 'orange'
        else:
            color = 'red'
        
        # Create popup with proper HTML escaping
        popup_html = f"""
        <div style="width: 200px;">
            <b>{row['name'][:50]}{'...' if len(row['name']) > 50 else ''}</b><br>
            <b>Price:</b> ${row['price']}<br>
            <b>Neighborhood:</b> {row['neighbourhood']}<br>
            <b>Rating:</b> {row['review_scores_rating']:.2f}<br>
            <b>Type:</b> {row['property_type']}<br>
            <b>Occupancy:</b> {row['occupancy_rate']:.1%}
        </div>
        """
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=6,
            popup=folium.Popup(popup_html, max_width=250),
            color=color,
            fill=True,
            fillOpacity=0.7,
            weight=2
        ).add_to(m)
    
    return m

def main():
    """Main dashboard function"""
    # Header
    st.markdown('<h1 class="main-header">🏠 Vancouver Airbnb Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Neighborhood filter
    neighborhoods = ['All'] + sorted(df['neighbourhood'].unique().tolist())
    selected_neighborhood = st.sidebar.selectbox("Select Neighborhood", neighborhoods)
    
    # Property type filter
    property_types = ['All'] + sorted(df['property_type'].unique().tolist())
    selected_property_type = st.sidebar.selectbox("Select Property Type", property_types)
    
    # Price range filter
    price_range = st.sidebar.slider(
        "Price Range ($)",
        min_value=int(df['price'].min()),
        max_value=int(df['price'].max()),
        value=(int(df['price'].min()), int(df['price'].max()))
    )
    
    # Apply filters
    filtered_df = df.copy()
    if selected_neighborhood != 'All':
        filtered_df = filtered_df[filtered_df['neighbourhood'] == selected_neighborhood]
    if selected_property_type != 'All':
        filtered_df = filtered_df[filtered_df['property_type'] == selected_property_type]
    filtered_df = filtered_df[
        (filtered_df['price'] >= price_range[0]) & 
        (filtered_df['price'] <= price_range[1])
    ]
    
    # Key metrics
    st.header("📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Listings",
            value=f"{len(filtered_df):,}",
            delta=f"{len(filtered_df) - len(df)}" if len(filtered_df) != len(df) else None
        )
    
    with col2:
        st.metric(
            label="Average Price",
            value=f"${filtered_df['price'].mean():.2f}",
            delta=f"${filtered_df['price'].mean() - df['price'].mean():.2f}" if len(filtered_df) != len(df) else None
        )
    
    with col3:
        st.metric(
            label="Average Rating",
            value=f"{filtered_df['review_scores_rating'].mean():.2f}",
            delta=f"{filtered_df['review_scores_rating'].mean() - df['review_scores_rating'].mean():.2f}" if len(filtered_df) != len(df) else None
        )
    
    with col4:
        st.metric(
            label="Average Occupancy",
            value=f"{filtered_df['occupancy_rate'].mean():.1%}",
            delta=f"{filtered_df['occupancy_rate'].mean() - df['occupancy_rate'].mean():.1%}" if len(filtered_df) != len(df) else None
        )
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🗺️ Interactive Map", "💰 Price Analysis", "🏘️ Neighborhood Insights", 
        "🏠 Property Analysis", "📈 Revenue & Performance", "⭐ Reviews & Amenities"
    ])
    
    with tab1:
        st.header("Interactive Property Map")
        st.write("Explore Vancouver Airbnb properties on the map. Green = Low Price, Orange = Medium Price, Red = High Price")
        
        # Create a unique key based on filtered data to prevent unnecessary re-rendering
        map_key = f"map_{len(filtered_df)}_{selected_neighborhood}_{selected_property_type}_{price_range[0]}_{price_range[1]}"
        
        # Use a container to prevent re-rendering
        with st.container():
            map_fig = create_interactive_map(filtered_df)
            st_folium(map_fig, width=700, height=500, key=map_key)
    
    with tab2:
        st.header("Price Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            price_dist_fig = create_price_distribution_chart(filtered_df)
            st.plotly_chart(price_dist_fig, use_container_width=True)
        
        with col2:
            # Price by room type
            room_type_price = filtered_df.groupby('room_type')['price'].mean().reset_index()
            room_type_fig = px.bar(
                room_type_price,
                x='room_type',
                y='price',
                title="Average Price by Room Type",
                labels={'room_type': 'Room Type', 'price': 'Average Price ($)'},
                color='price',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(room_type_fig, use_container_width=True)
    
    with tab3:
        st.header("Neighborhood Analysis")
        neighborhood_fig, neighborhood_stats = create_neighborhood_analysis(filtered_df)
        st.plotly_chart(neighborhood_fig, use_container_width=True)
        
        st.subheader("Neighborhood Statistics")
        st.dataframe(neighborhood_stats, use_container_width=True)
    
    with tab4:
        st.header("Property Type Analysis")
        property_fig, property_stats = create_property_type_analysis(filtered_df)
        st.plotly_chart(property_fig, use_container_width=True)
        
        st.subheader("Property Type Statistics")
        st.dataframe(property_stats, use_container_width=True)
    
    with tab5:
        st.header("Revenue & Performance Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            revenue_fig1, revenue_fig2 = create_revenue_analysis(filtered_df)
            st.plotly_chart(revenue_fig1, use_container_width=True)
        
        with col2:
            st.plotly_chart(revenue_fig2, use_container_width=True)
        
        # Revenue insights
        st.subheader("Revenue Insights")
        top_revenue = filtered_df.nlargest(10, 'revenue_ltm')[['name', 'price', 'revenue_ltm', 'occupancy_rate', 'neighbourhood']]
        st.dataframe(top_revenue, use_container_width=True)
    
    with tab6:
        st.header("Reviews & Amenities Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            review_fig = create_review_analysis(filtered_df)
            st.plotly_chart(review_fig, use_container_width=True)
        
        with col2:
            amenity_fig = create_amenity_analysis(filtered_df)
            st.plotly_chart(amenity_fig, use_container_width=True)
        
        # Top rated properties
        st.subheader("Top Rated Properties")
        top_rated = filtered_df.nlargest(10, 'review_scores_rating')[['name', 'review_scores_rating', 'price', 'neighbourhood']]
        st.dataframe(top_rated, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("**Data Source:** Generated Vancouver Airbnb Dataset | **Last Updated:** " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    main()
