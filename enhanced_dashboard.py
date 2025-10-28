"""
Enhanced Vancouver Airbnb Investment & Operations Dashboard
Comprehensive market analysis for investment and operational strategies
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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import webbrowser

# Import the real data loader
from real_data_loader import load_real_airbnb_data
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Vancouver Airbnb Investment Intelligence Dashboard",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 2rem;
        color: #ff7f0e;
        margin: 1rem 0;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .insight-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .success-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the real Airbnb data with enhanced processing"""
    try:
        # Use the real data loader
        df = load_real_airbnb_data()
        
        if df is None:
            st.error("Failed to load real Airbnb data. Please check the Data folder.")
            return None
        
        # Additional processing for enhanced analysis
        df['profit_margin'] = (df['revenue_ltm'] - (df['price'] * 30 * 12 * 0.3)) / df['revenue_ltm'] * 100
        
        # Competitive analysis metrics
        df['price_vs_market'] = df['price'] / df.groupby('neighbourhood')['price'].transform('mean')
        df['rating_vs_market'] = df['review_scores_rating'] / df.groupby('neighbourhood')['review_scores_rating'].transform('mean')
        
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please run data_collector.py first.")
        return None

def create_investment_heatmap(df):
    """Create investment opportunity heatmap by neighborhood"""
    investment_data = df.groupby('neighbourhood').agg({
        'roi_potential': 'mean',
        'market_score': 'mean',
        'price': 'mean',
        'occupancy_rate': 'mean',
        'revenue_ltm': 'mean'
    }).round(2)
    
    # Create heatmap
    fig = px.imshow(
        investment_data.T,
        labels=dict(x="Neighborhood", y="Metric", color="Value"),
        title="Investment Opportunity Heatmap by Neighborhood",
        color_continuous_scale='RdYlGn',
        aspect="auto"
    )
    fig.update_layout(height=600)
    return fig, investment_data

def create_roi_analysis(df):
    """Create comprehensive ROI analysis"""
    # ROI distribution
    fig1 = px.histogram(
        df, 
        x='roi_potential', 
        nbins=30,
        title="ROI Distribution Across All Properties",
        labels={'roi_potential': 'ROI Potential (%)', 'count': 'Number of Properties'},
        color_discrete_sequence=['#1f77b4']
    )
    
    # ROI vs Price scatter
    fig2 = px.scatter(
        df,
        x='price',
        y='roi_potential',
        color='investment_grade',
        size='revenue_ltm',
        title="ROI vs Price (colored by Investment Grade)",
        labels={'price': 'Price ($)', 'roi_potential': 'ROI Potential (%)'},
        hover_data=['neighbourhood', 'property_type', 'occupancy_rate']
    )
    
    # Top ROI properties
    top_roi = df.nlargest(20, 'roi_potential')[['name', 'neighbourhood', 'price', 'roi_potential', 'revenue_ltm', 'occupancy_rate']]
    
    return fig1, fig2, top_roi

def create_competitive_analysis(df):
    """Create competitive analysis with market positioning"""
    # Market positioning scatter
    fig1 = px.scatter(
        df,
        x='price_vs_market',
        y='rating_vs_market',
        color='investment_grade',
        size='revenue_ltm',
        title="Market Positioning: Price vs Rating (vs Market Average)",
        labels={'price_vs_market': 'Price vs Market Average', 'rating_vs_market': 'Rating vs Market Average'},
        hover_data=['name', 'neighbourhood', 'price', 'review_scores_rating']
    )
    fig1.add_hline(y=1, line_dash="dash", line_color="red", annotation_text="Market Average Rating")
    fig1.add_vline(x=1, line_dash="dash", line_color="red", annotation_text="Market Average Price")
    
    # Competitive clusters - handle cases with insufficient data
    competitive_features = ['price', 'review_scores_rating', 'occupancy_rate', 'revenue_ltm']
    X = df[competitive_features].fillna(df[competitive_features].mean())
    
    # Only perform clustering if we have enough samples
    if len(X) >= 4:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Use fewer clusters if we have limited data
        n_clusters = min(4, len(X))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['competitive_cluster'] = kmeans.fit_predict(X_scaled)
        
        cluster_names = {i: f'Cluster {i+1}' for i in range(n_clusters)}
        if n_clusters == 4:
            cluster_names = {0: 'Budget Leaders', 1: 'Premium Performers', 2: 'Underperformers', 3: 'Balanced Players'}
        df['cluster_name'] = df['competitive_cluster'].map(cluster_names)
    else:
        # If insufficient data, create simple categories
        df['competitive_cluster'] = 0
        df['cluster_name'] = 'All Properties'
    
    fig2 = px.scatter(
        df,
        x='price',
        y='revenue_ltm',
        color='cluster_name',
        title="Competitive Clusters: Price vs Revenue",
        labels={'price': 'Price ($)', 'revenue_ltm': 'Revenue LTM ($)'},
        hover_data=['name', 'neighbourhood', 'review_scores_rating', 'occupancy_rate']
    )
    
    return fig1, fig2, df

def create_operational_insights(df):
    """Create operational efficiency insights"""
    # Occupancy vs Revenue analysis
    fig1 = px.scatter(
        df,
        x='occupancy_rate',
        y='revenue_ltm',
        color='property_type',
        size='price',
        title="Operational Efficiency: Occupancy vs Revenue",
        labels={'occupancy_rate': 'Occupancy Rate', 'revenue_ltm': 'Revenue LTM ($)'},
        hover_data=['name', 'neighbourhood', 'price', 'review_scores_rating']
    )
    
    # Host performance analysis (using id as proxy for host_id if not available)
    host_col = 'host_id' if 'host_id' in df.columns else 'id'
    if host_col in df.columns:
        host_performance = df.groupby(host_col).agg({
            'revenue_ltm': 'sum',
            'review_scores_rating': 'mean',
            'occupancy_rate': 'mean',
            'price': 'mean',
            'name': 'count'
        }).round(2)
        host_performance.columns = ['Total Revenue', 'Avg Rating', 'Avg Occupancy', 'Avg Price', 'Property Count']
    else:
        # Create a dummy host performance if no host column is available
        host_performance = pd.DataFrame({
            'Total Revenue': [df['revenue_ltm'].sum()],
            'Avg Rating': [df['review_scores_rating'].mean()],
            'Avg Occupancy': [df['occupancy_rate'].mean()],
            'Avg Price': [df['price'].mean()],
            'Property Count': [len(df)]
        })
    host_performance = host_performance.sort_values('Total Revenue', ascending=False).head(20)
    
    # Operational recommendations
    operational_insights = {
        'High Occupancy, Low Revenue': df[(df['occupancy_rate'] > 0.7) & (df['revenue_ltm'] < df['revenue_ltm'].quantile(0.5))],
        'Low Occupancy, High Revenue': df[(df['occupancy_rate'] < 0.5) & (df['revenue_ltm'] > df['revenue_ltm'].quantile(0.7))],
        'Optimization Opportunities': df[(df['occupancy_rate'] < 0.6) & (df['revenue_ltm'] < df['revenue_ltm'].quantile(0.6))]
    }
    
    return fig1, host_performance, operational_insights

def create_market_forecasting(df):
    """Create market forecasting and trends"""
    # Price trends by neighborhood
    neighborhood_trends = df.groupby(['neighbourhood', 'property_type']).agg({
        'price': ['mean', 'std'],
        'occupancy_rate': 'mean',
        'revenue_ltm': 'mean'
    }).round(2)
    
    # Market saturation analysis
    saturation_analysis = df.groupby('neighbourhood').agg({
        'name': 'count',
        'price': 'mean',
        'occupancy_rate': 'mean',
        'revenue_ltm': 'mean'
    }).round(2)
    saturation_analysis.columns = ['Property Count', 'Avg Price', 'Avg Occupancy', 'Avg Revenue']
    saturation_analysis['market_saturation'] = saturation_analysis['Property Count'] / saturation_analysis['Property Count'].max()
    
    # Growth potential scoring - only if not already exists
    if 'growth_potential' not in df.columns:
        df['growth_potential'] = (
            (df['occupancy_rate'] * 0.3) +
            (df['review_scores_rating'] / 5 * 0.3) +
            (1 - df['price'] / df['price'].max() * 0.4)
        ) * 100
    
    return neighborhood_trends, saturation_analysis, df

def create_enhanced_map(df):
    """Create enhanced interactive map with investment insights"""
    # Create base map
    m = folium.Map(
        location=[49.2827, -123.1207],
        zoom_start=11,
        tiles='OpenStreetMap'
    )
    
    # Add investment grade markers
    for idx, row in df.iterrows():
        # Color based on investment grade
        color_map = {'Poor': 'red', 'Fair': 'orange', 'Good': 'blue', 'Excellent': 'green'}
        color = color_map.get(row['investment_grade'], 'gray')
        
        # Size based on ROI potential
        size = max(5, min(15, row['roi_potential'] / 10))
        
        # Enhanced popup with investment insights - smaller, more readable text
        popup_html = f"""
        <div style="width: 280px; font-family: Arial, sans-serif; font-size: 12px;">
            <h4 style="color: #1f77b4; margin: 0 0 8px 0; font-size: 14px; line-height: 1.2;">{row['name'][:35]}{'...' if len(row['name']) > 35 else ''}</h4>
            <hr style="margin: 5px 0; border: 1px solid #ddd;">
            <div style="line-height: 1.4;">
                <p style="margin: 3px 0; font-size: 11px;"><strong>💰 Price:</strong> ${row['price']:,.0f}</p>
                <p style="margin: 3px 0; font-size: 11px;"><strong>📈 ROI:</strong> {row['roi_potential']:.1f}%</p>
                <p style="margin: 3px 0; font-size: 11px;"><strong>⭐ Grade:</strong> {row['investment_grade']}</p>
                <p style="margin: 3px 0; font-size: 11px;"><strong>🏠 Type:</strong> {row['property_type']}</p>
                <p style="margin: 3px 0; font-size: 11px;"><strong>📍 Area:</strong> {row['neighbourhood']}</p>
                <p style="margin: 3px 0; font-size: 11px;"><strong>📊 Occupancy:</strong> {row['occupancy_rate']:.1%}</p>
                <p style="margin: 3px 0; font-size: 11px;"><strong>💵 Revenue:</strong> ${row['revenue_ltm']:,.0f}</p>
                <p style="margin: 3px 0; font-size: 11px;"><strong>⭐ Rating:</strong> {row['review_scores_rating']:.2f}/5</p>
            </div>
            <hr style="margin: 8px 0; border: 1px solid #ddd;">
            <a href="{row['airbnb_url']}" target="_blank" style="background: #1f77b4; color: white; padding: 4px 8px; text-decoration: none; border-radius: 3px; font-size: 10px; display: inline-block;">
                🔗 View on Airbnb
            </a>
        </div>
        """
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=size,
            popup=folium.Popup(popup_html, max_width=350),
            color=color,
            fill=True,
            fillOpacity=0.7,
            weight=2
        ).add_to(m)
    
    # Add legend with smaller, more readable text
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 180px; height: 100px; 
                background-color: white; border:2px solid #666; z-index:9999; 
                font-size:11px; padding: 8px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.2)">
    <p style="margin: 0 0 5px 0; font-weight: bold; font-size: 12px;">Investment Grade:</p>
    <p style="margin: 2px 0; font-size: 10px;"><span style="color:green; font-weight: bold;">●</span> Excellent</p>
    <p style="margin: 2px 0; font-size: 10px;"><span style="color:blue; font-weight: bold;">●</span> Good</p>
    <p style="margin: 2px 0; font-size: 10px;"><span style="color:orange; font-weight: bold;">●</span> Fair</p>
    <p style="margin: 2px 0; font-size: 10px;"><span style="color:red; font-weight: bold;">●</span> Poor</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

def create_neighborhood_analysis(df):
    """Create comprehensive neighborhood-level analysis"""
    # Neighborhood performance metrics
    neighborhood_metrics = df.groupby('neighbourhood').agg({
        'price': ['mean', 'median', 'std', 'count'],
        'roi_potential': ['mean', 'max'],
        'occupancy_rate': 'mean',
        'revenue_ltm': ['mean', 'sum'],
        'review_scores_rating': 'mean',
        'market_score': 'mean',
        'investment_grade': lambda x: (x == 'Excellent').sum()
    }).round(2)
    
    # Flatten column names
    neighborhood_metrics.columns = [
        'Avg Price', 'Median Price', 'Price Std', 'Property Count',
        'Avg ROI', 'Max ROI', 'Avg Occupancy', 'Avg Revenue', 'Total Revenue',
        'Avg Rating', 'Avg Market Score', 'Excellent Count'
    ]
    
    # Calculate additional metrics
    neighborhood_metrics['ROI Rank'] = neighborhood_metrics['Avg ROI'].rank(ascending=False)
    neighborhood_metrics['Price Rank'] = neighborhood_metrics['Avg Price'].rank(ascending=False)
    neighborhood_metrics['Revenue Rank'] = neighborhood_metrics['Total Revenue'].rank(ascending=False)
    neighborhood_metrics['Excellence Rate'] = (neighborhood_metrics['Excellent Count'] / neighborhood_metrics['Property Count'] * 100).round(1)
    
    # Sort by ROI potential
    neighborhood_metrics = neighborhood_metrics.sort_values('Avg ROI', ascending=False)
    
    return neighborhood_metrics

def create_neighborhood_visualizations(df):
    """Create neighborhood-specific visualizations"""
    # Top neighborhoods by ROI
    top_neighborhoods = df.groupby('neighbourhood')['roi_potential'].mean().nlargest(10)
    
    fig1 = px.bar(
        x=top_neighborhoods.values,
        y=top_neighborhoods.index,
        orientation='h',
        title="Top 10 Neighborhoods by Average ROI",
        labels={'x': 'Average ROI (%)', 'y': 'Neighborhood'},
        color=top_neighborhoods.values,
        color_continuous_scale='Viridis'
    )
    fig1.update_layout(height=500)
    
    # Price vs ROI by neighborhood
    neighborhood_analysis = df.groupby('neighbourhood').agg({
        'price': 'mean',
        'roi_potential': 'mean',
        'occupancy_rate': 'mean',
        'name': 'count'
    }).round(2)
    neighborhood_analysis.columns = ['Avg Price', 'Avg ROI', 'Avg Occupancy', 'Property Count']
    
    fig2 = px.scatter(
        neighborhood_analysis,
        x='Avg Price',
        y='Avg ROI',
        size='Property Count',
        color='Avg Occupancy',
        title="Neighborhood Performance: Price vs ROI",
        labels={'Avg Price': 'Average Price ($)', 'Avg ROI': 'Average ROI (%)'},
        hover_data=['Property Count', 'Avg Occupancy'],
        color_continuous_scale='RdYlGn'
    )
    
    # Investment grade distribution by neighborhood
    grade_distribution = df.groupby(['neighbourhood', 'investment_grade']).size().unstack(fill_value=0)
    grade_distribution_pct = grade_distribution.div(grade_distribution.sum(axis=1), axis=0) * 100
    
    fig3 = px.bar(
        grade_distribution_pct,
        title="Investment Grade Distribution by Neighborhood (%)",
        labels={'neighbourhood': 'Neighborhood', 'value': 'Percentage (%)'},
        color_discrete_map={
            'Excellent': '#2E8B57',
            'Good': '#4682B4', 
            'Fair': '#FF8C00',
            'Poor': '#DC143C'
        }
    )
    fig3.update_layout(height=500, xaxis_tickangle=-45)
    
    return fig1, fig2, fig3, neighborhood_analysis

def create_market_summary(df):
    """Create comprehensive market summary and insights"""
    # Key market metrics
    total_properties = len(df)
    avg_price = df['price'].mean()
    avg_roi = df['roi_potential'].mean()
    avg_occupancy = df['occupancy_rate'].mean()
    total_revenue = df['revenue_ltm'].sum()
    
    # Market segments
    excellent_investments = len(df[df['investment_grade'] == 'Excellent'])
    good_investments = len(df[df['investment_grade'] == 'Good'])
    
    # Top performing neighborhoods
    top_neighborhoods = df.groupby('neighbourhood').agg({
        'roi_potential': 'mean',
        'market_score': 'mean',
        'revenue_ltm': 'mean'
    }).sort_values('roi_potential', ascending=False).head(5)
    
    # Market opportunities - safely check for required columns
    opportunities = {
        'Undervalued Properties': len(df[(df.get('price_vs_market', 1) < 0.8) & (df.get('rating_vs_market', 1) > 1.1)]) if 'price_vs_market' in df.columns and 'rating_vs_market' in df.columns else 0,
        'High ROI Opportunities': len(df[df['roi_potential'] > df['roi_potential'].quantile(0.8)]) if 'roi_potential' in df.columns else 0,
        'Growth Potential': len(df[df.get('growth_potential', pd.Series([0] * len(df))) > 70]) if 'growth_potential' in df.columns else 0
    }
    
    return {
        'total_properties': total_properties,
        'avg_price': avg_price,
        'avg_roi': avg_roi,
        'avg_occupancy': avg_occupancy,
        'total_revenue': total_revenue,
        'excellent_investments': excellent_investments,
        'good_investments': good_investments,
        'top_neighborhoods': top_neighborhoods,
        'opportunities': opportunities
    }

def main():
    """Enhanced main dashboard function"""
    # Header
    st.markdown('<h1 class="main-header">Vancouver Airbnb Investment Intelligence Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Real Vancouver Airbnb Data Analysis for Investment & Operations Strategy</p>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Sidebar filters
    st.sidebar.header("🔍 Investment Filters")
    
    # Investment grade filter with enhanced options
    investment_grades = ['All'] + sorted(list(df['investment_grade'].unique()))
    selected_grade = st.sidebar.selectbox(
        "Investment Grade", 
        investment_grades,
        help="Filter properties by investment quality grade"
    )
    
    # Investment grade color coding
    grade_colors = {
        'Excellent': '🟢',
        'Good': '🔵', 
        'Fair': '🟠',
        'Poor': '🔴'
    }
    
    # Show grade distribution
    st.sidebar.markdown("**Grade Distribution:**")
    grade_counts = df['investment_grade'].value_counts()
    for grade in sorted(df['investment_grade'].unique()):
        count = grade_counts.get(grade, 0)
        color = grade_colors.get(grade, '⚪')
        st.sidebar.markdown(f"{color} {grade}: {count}")
    
    
    # ROI range filter
    roi_range = st.sidebar.slider(
        "ROI Range (%)",
        min_value=float(df['roi_potential'].min()),
        max_value=float(df['roi_potential'].max()),
        value=(float(df['roi_potential'].min()), float(df['roi_potential'].max())),
        help="Filter by ROI potential range"
    )
    
    # Price range filter
    price_range = st.sidebar.slider(
        "Price Range ($)",
        min_value=int(df['price'].min()),
        max_value=int(df['price'].max()),
        value=(int(df['price'].min()), int(df['price'].max())),
        help="Filter by property price range"
    )
    
    # Additional filters
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Additional Filters:**")
    
    # Neighborhood filter
    neighborhoods = ['All'] + sorted(df['neighbourhood'].unique().tolist())
    selected_neighborhood = st.sidebar.selectbox(
        "Neighborhood", 
        neighborhoods,
        help="Filter by specific neighborhood"
    )
    
    # Property type filter
    property_types = ['All'] + sorted(df['property_type'].unique().tolist())
    selected_property_type = st.sidebar.selectbox(
        "Property Type", 
        property_types,
        help="Filter by property type"
    )
    
    # Apply filters with enhanced logic
    filtered_df = df.copy()
    
    # Investment grade filtering
    if selected_grade != 'All':
        filtered_df = filtered_df[filtered_df['investment_grade'] == selected_grade]
    
    # ROI and Price range filtering
    filtered_df = filtered_df[
        (filtered_df['roi_potential'] >= roi_range[0]) & 
        (filtered_df['roi_potential'] <= roi_range[1]) &
        (filtered_df['price'] >= price_range[0]) & 
        (filtered_df['price'] <= price_range[1])
    ]
    
    # Additional filters
    if selected_neighborhood != 'All':
        filtered_df = filtered_df[filtered_df['neighbourhood'] == selected_neighborhood]
    if selected_property_type != 'All':
        filtered_df = filtered_df[filtered_df['property_type'] == selected_property_type]
    
    # Ensure all required columns exist
    required_columns = ['growth_potential', 'price_vs_market', 'rating_vs_market']
    for col in required_columns:
        if col not in filtered_df.columns:
            if col == 'growth_potential':
                filtered_df[col] = (
                    (filtered_df['occupancy_rate'] * 0.3) +
                    (filtered_df['review_scores_rating'] / 5 * 0.3) +
                    (1 - filtered_df['price'] / filtered_df['price'].max() * 0.4)
                ) * 100
            elif col == 'price_vs_market':
                market_avg_price = filtered_df['price'].mean()
                filtered_df[col] = filtered_df['price'] / market_avg_price
            elif col == 'rating_vs_market':
                market_avg_rating = filtered_df['review_scores_rating'].mean()
                filtered_df[col] = filtered_df['review_scores_rating'] / market_avg_rating
    
    # Market Summary
    market_summary = create_market_summary(filtered_df)
    
    # Filter summary
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Active Filters:**")
    active_filters = []
    if selected_grade != 'All':
        active_filters.append(f"Grade: {selected_grade}")
    if selected_neighborhood != 'All':
        active_filters.append(f"Neighborhood: {selected_neighborhood}")
    if selected_property_type != 'All':
        active_filters.append(f"Type: {selected_property_type}")
    if roi_range[0] != df['roi_potential'].min() or roi_range[1] != df['roi_potential'].max():
        active_filters.append(f"ROI: {roi_range[0]:.1f}%-{roi_range[1]:.1f}%")
    if price_range[0] != df['price'].min() or price_range[1] != df['price'].max():
        active_filters.append(f"Price: ${price_range[0]:,}-${price_range[1]:,}")
    
    if active_filters:
        for filter_text in active_filters:
            st.sidebar.markdown(f"• {filter_text}")
        st.sidebar.markdown("---")
        if st.sidebar.button("🔄 Clear All Filters", help="Reset all filters to show all properties"):
            st.rerun()
    else:
        st.sidebar.markdown("• No filters applied")
    
    # Key metrics
    st.header("Market Overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Properties", f"{market_summary['total_properties']:,}")
    with col2:
        st.metric("Avg Price", f"${market_summary['avg_price']:,.0f}")
    with col3:
        st.metric("Avg ROI", f"{market_summary['avg_roi']:.1f}%")
    with col4:
        st.metric("Avg Occupancy", f"{market_summary['avg_occupancy']:.1%}")
    with col5:
        st.metric("Total Revenue", f"${market_summary['total_revenue']:,.0f}")
    
    # Investment opportunities
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("### Investment Opportunities")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Excellent Investments", market_summary['excellent_investments'])
    with col2:
        st.metric("Good Investments", market_summary['good_investments'])
    with col3:
        st.metric("High ROI Opportunities", market_summary['opportunities']['High ROI Opportunities'])
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🗺️ Investment Map", "💰 ROI Analysis", "🏆 Competitive Intelligence", 
        "⚙️ Operations Insights", "📈 Market Forecasting", "🎯 Investment Opportunities", 
        "🏘️ Neighborhood Analysis", "📋 Executive Summary"
    ])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Investment Opportunity Map</h2>', unsafe_allow_html=True)
        st.write("Interactive map showing investment opportunities. Click markers for detailed analysis and direct links to Airbnb listings.")
        
        # Create enhanced map
        map_fig = create_enhanced_map(filtered_df)
        st_folium(map_fig, width=900, height=600)
        
        # Map insights with smaller, more readable text
        st.markdown("### Map Insights")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Color Coding:**")
            st.markdown("- 🟢 **Green:** Excellent Investment")
            st.markdown("- 🔵 **Blue:** Good Investment") 
            st.markdown("- 🟠 **Orange:** Fair Investment")
            st.markdown("- 🔴 **Red:** Poor Investment")
        with col2:
            st.markdown("**Marker Size:**")
            st.markdown("- **Larger markers** = Higher ROI potential")
            st.markdown("- **Click markers** for detailed property info")
            st.markdown("- **Direct links** to Airbnb listings included")
    
    with tab2:
        st.markdown('<h2 class="sub-header">ROI Analysis</h2>', unsafe_allow_html=True)
        
        # ROI visualizations
        fig1, fig2, top_roi = create_roi_analysis(filtered_df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            st.plotly_chart(fig2, use_container_width=True)
        
        # Top ROI properties
        st.markdown("### 🏆 Top ROI Properties")
        st.dataframe(top_roi, use_container_width=True)
    
    with tab3:
        st.markdown('<h2 class="sub-header">Competitive Intelligence</h2>', unsafe_allow_html=True)
        
        fig1, fig2, df_with_clusters = create_competitive_analysis(filtered_df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            st.plotly_chart(fig2, use_container_width=True)
        
        # Competitive insights
        st.markdown("### 🎯 Competitive Clusters")
        cluster_analysis = df_with_clusters.groupby('cluster_name').agg({
            'price': 'mean',
            'revenue_ltm': 'mean',
            'occupancy_rate': 'mean',
            'roi_potential': 'mean'
        }).round(2)
        st.dataframe(cluster_analysis, use_container_width=True)
    
    with tab4:
        st.markdown('<h2 class="sub-header">Operations Insights</h2>', unsafe_allow_html=True)
        
        fig1, host_performance, operational_insights = create_operational_insights(filtered_df)
        
        st.plotly_chart(fig1, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🏆 Top Performing Hosts")
            st.dataframe(host_performance, use_container_width=True)
        
        with col2:
            st.markdown("### ⚠️ Operational Alerts")
            for insight_type, properties in operational_insights.items():
                if len(properties) > 0:
                    st.markdown(f"**{insight_type}:** {len(properties)} properties")
    
    with tab5:
        st.markdown('<h2 class="sub-header">Market Forecasting</h2>', unsafe_allow_html=True)
        
        neighborhood_trends, saturation_analysis, df_with_growth = create_market_forecasting(filtered_df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Market Saturation Analysis")
            st.dataframe(saturation_analysis, use_container_width=True)
        
        with col2:
            st.markdown("### Growth Potential Leaders")
            growth_leaders = df_with_growth.nlargest(10, 'growth_potential')[['name', 'neighbourhood', 'growth_potential', 'roi_potential', 'price']]
            st.dataframe(growth_leaders, use_container_width=True)
    
    with tab6:
        st.markdown('<h2 class="sub-header">Investment Opportunities</h2>', unsafe_allow_html=True)
        
        # Investment heatmap
        heatmap_fig, investment_data = create_investment_heatmap(filtered_df)
        st.plotly_chart(heatmap_fig, use_container_width=True)
        
        # Top opportunities
        st.markdown("### Top Investment Opportunities")
        opportunities_df = filtered_df.nlargest(20, 'market_score')[['name', 'neighbourhood', 'price', 'roi_potential', 'market_score', 'investment_grade', 'airbnb_url']]
        st.dataframe(opportunities_df, use_container_width=True)
    
    with tab7:
        st.markdown('<h2 class="sub-header">Neighborhood Analysis</h2>', unsafe_allow_html=True)
        
        # Create neighborhood analysis
        neighborhood_metrics = create_neighborhood_analysis(filtered_df)
        fig1, fig2, fig3, neighborhood_analysis = create_neighborhood_visualizations(filtered_df)
        
        # Neighborhood performance overview
        st.markdown("### 📊 Neighborhood Performance Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.plotly_chart(fig2, use_container_width=True)
        
        # Investment grade distribution
        st.plotly_chart(fig3, use_container_width=True)
        
        # Detailed neighborhood metrics
        st.markdown("### 📈 Detailed Neighborhood Metrics")
        st.dataframe(neighborhood_metrics, use_container_width=True)
        
        # Top performing neighborhoods
        st.markdown("### 🏆 Top Performing Neighborhoods")
        top_performers = neighborhood_metrics.head(10)[['Avg ROI', 'Avg Price', 'Avg Occupancy', 'Excellence Rate', 'Property Count']]
        st.dataframe(top_performers, use_container_width=True)
        
        # Neighborhood insights
        st.markdown("### 💡 Neighborhood Insights")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            best_roi_neighborhood = neighborhood_metrics.index[0]
            best_roi_value = neighborhood_metrics.iloc[0]['Avg ROI']
            st.metric("Best ROI Neighborhood", f"{best_roi_neighborhood}", f"{best_roi_value:.1f}%")
        
        with col2:
            highest_price_neighborhood = neighborhood_metrics.loc[neighborhood_metrics['Price Rank'] == 1].index[0]
            highest_price = neighborhood_metrics.loc[neighborhood_metrics['Price Rank'] == 1]['Avg Price'].iloc[0]
            st.metric("Highest Price Neighborhood", f"{highest_price_neighborhood}", f"${highest_price:,.0f}")
        
        with col3:
            most_excellent_neighborhood = neighborhood_metrics.loc[neighborhood_metrics['Excellence Rate'].idxmax()]
            excellence_rate = neighborhood_metrics.loc[neighborhood_metrics['Excellence Rate'].idxmax()]['Excellence Rate']
            st.metric("Highest Excellence Rate", f"{neighborhood_metrics.loc[neighborhood_metrics['Excellence Rate'].idxmax()].name}", f"{excellence_rate:.1f}%")
    
    with tab8:
        st.markdown('<h2 class="sub-header">Executive Summary</h2>', unsafe_allow_html=True)
        
        # Key insights
        st.markdown("### 📈 Key Market Insights")
        
        insights = [
            f"**Market Size:** {market_summary['total_properties']:,} total properties generating ${market_summary['total_revenue']:,.0f} in annual revenue",
            f"**Average Performance:** ${market_summary['avg_price']:,.0f} average price with {market_summary['avg_roi']:.1f}% ROI potential",
            f"**Investment Quality:** {market_summary['excellent_investments']} excellent and {market_summary['good_investments']} good investment opportunities",
            f"**Market Opportunities:** {market_summary['opportunities']['High ROI Opportunities']} high ROI opportunities identified"
        ]
        
        for insight in insights:
            st.markdown(f"• {insight}")
        
        # Strategic recommendations
        st.markdown("### 🎯 Strategic Recommendations")
        
        recommendations = [
            "**Focus on Excellent/Good Grade Properties:** Prioritize properties with proven market performance",
            "**Geographic Diversification:** Consider multiple neighborhoods to spread risk",
            "**ROI Optimization:** Target properties with ROI > 15% for better returns",
            "**Operational Efficiency:** Monitor occupancy rates and adjust pricing strategies",
            "**Competitive Positioning:** Analyze market positioning relative to competitors"
        ]
        
        for rec in recommendations:
            st.markdown(f"• {rec}")
        
        # Market trends
        st.markdown("### 📊 Market Trends")
        st.markdown("• **Price Trends:** Monitor neighborhood price movements for timing opportunities")
        st.markdown("• **Occupancy Patterns:** Identify seasonal and location-based occupancy trends")
        st.markdown("• **Revenue Optimization:** Focus on properties with high revenue potential")
        st.markdown("• **Competitive Landscape:** Stay aware of market saturation and competition")
    
    # Footer
    st.markdown("---")
    st.markdown("**💼 Investment Intelligence Dashboard** | **Data Source:** Vancouver Airbnb Market Analysis | **Last Updated:** " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Data source attribution
    st.markdown("**Data Source:** This analysis uses data from [Inside Airbnb](https://insideairbnb.com/get-the-data/) - a mission-driven project that provides data and advocacy about Airbnb's impact on residential communities.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check the data files and try again.")
