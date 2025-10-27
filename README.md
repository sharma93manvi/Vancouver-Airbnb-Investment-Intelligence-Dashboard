# Vancouver Airbnb Investment Intelligence Dashboard

A comprehensive data visualization and investment analysis platform for Vancouver's Airbnb market, built with Streamlit and powered by real market data.

![Dashboard Preview](https://img.shields.io/badge/Status-Live-brightgreen) ![Python](https://img.shields.io/badge/Python-3.11+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red) ![License](https://img.shields.io/badge/License-MIT-yellow)

## Overview

This project provides investors, real estate professionals, and market analysts with deep insights into Vancouver's Airbnb market through an interactive dashboard that combines data visualization, investment analysis, and competitive intelligence.

## Key Features

### **Market Intelligence**
- **Real-time Market Overview** - Key metrics including average prices, ROI potential, and market opportunities
- **Investment Grading System** - Automated classification of properties (Excellent, Good, Fair, Poor)
- **Market Saturation Analysis** - Identify oversaturated vs. opportunity-rich areas
- **Growth Potential Scoring** - Predictive analysis for future market performance

###  **Interactive Mapping**
- **Geographic Visualization** - Interactive map with property locations and key metrics
- **Neighborhood Analysis** - Comprehensive neighborhood-level performance metrics
- **Competitive Positioning** - Market positioning analysis relative to competitors
- **One-click Property Access** - Direct links to Airbnb listings for detailed analysis

###  **Investment Analysis**
- **ROI Calculations** - Automated return on investment analysis
- **Revenue Forecasting** - Last 12 months revenue analysis and projections
- **Occupancy Rate Analysis** - Performance metrics and optimization insights
- **Price vs. Market Analysis** - Competitive pricing intelligence

###  **Neighborhood Intelligence**
- **Top Performing Areas** - Ranked neighborhoods by ROI and performance
- **Market Trends** - Price movements and occupancy patterns
- **Investment Opportunities** - Undervalued properties and high-potential areas
- **Risk Assessment** - Market saturation and competitive landscape analysis

##    Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://vancouver-airbnb-dashboard.streamlit.app/)

## 📁 Project Structure

```
Vancouver-Airbnb-Investment-Intelligence-Dashboard/
├── enhanced_dashboard.py          # Main Streamlit dashboard application
├── real_data_loader.py           # Data processing and loading utilities
├── data_collector.py             # Data collection scripts (legacy)
├── dashboard.py                  # Basic dashboard (legacy)
├── requirements.txt              # Python dependencies
├── Procfile                      # Heroku deployment configuration
├── runtime.txt                   # Python version specification
├── setup.sh                      # Streamlit cloud configuration
├── Data/                         # Real Vancouver Airbnb data
│   ├── listings.csv             # Property listings data
│   ├── calendar.csv             # Availability calendar data
│   ├── reviews.csv              # Guest reviews data
│   └── neighbourhoods.csv       # Geographic boundaries
└── README.md                     # This file
```

## Installation & Setup

### Prerequisites
- Python 3.11+
- pip package manager

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/sharma93manvi/Vancouver-Airbnb-Investment-Intelligence-Dashboard.git
   cd Vancouver-Airbnb-Investment-Intelligence-Dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the dashboard**
   ```bash
   streamlit run enhanced_dashboard.py
   ```

4. **Access the dashboard**
   - Open your browser to `http://localhost:8501`

### Cloud Deployment

#### Streamlit Community Cloud
1. Fork this repository
2. Connect your GitHub account to [Streamlit Community Cloud](https://share.streamlit.io/)
3. Deploy directly from your forked repository

#### Heroku
1. Install [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli)
2. Create a new Heroku app
3. Deploy using the included `Procfile` and `runtime.txt`

## Data Sources

This project uses real Vancouver Airbnb data from [Inside Airbnb](https://insideairbnb.com/get-the-data/), a mission-driven project that provides data and advocacy about Airbnb's impact on residential communities.

**Data includes:**
- Property listings with detailed information
- Calendar availability data
- Guest reviews and ratings
- Geographic neighborhood boundaries
- Pricing and revenue metrics

## Technical Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Folium
- **Machine Learning**: Scikit-learn (KMeans clustering)
- **Geographic Analysis**: GeoPandas, Shapely
- **Web Scraping**: Selenium, BeautifulSoup, Requests

## Key Metrics & Analysis

### Investment Metrics
- **ROI Potential**: Calculated based on revenue, occupancy, and pricing
- **Market Score**: Composite score combining multiple performance factors
- **Investment Grade**: Automated classification system
- **Growth Potential**: Predictive scoring for future performance

### Market Analysis
- **Price vs. Market**: Competitive positioning analysis
- **Rating vs. Market**: Quality assessment relative to market
- **Occupancy Optimization**: Performance efficiency metrics
- **Revenue Analysis**: Last 12 months financial performance

### Competitive Intelligence
- **Market Clustering**: KMeans-based competitive grouping
- **Positioning Analysis**: Price vs. quality market positioning
- **Opportunity Identification**: Undervalued and high-potential properties

## Dashboard Features

### Interactive Filters
- **Neighborhood Selection**: Filter by specific areas
- **Property Type**: Apartment, house, condo, etc.
- **Price Range**: Customizable price brackets
- **Investment Grade**: Filter by investment quality
- **Clear All Filters**: Reset to default view

### Visualization Types
- **Interactive Maps**: Folium-based geographic visualization
- **Scatter Plots**: Market positioning and competitive analysis
- **Bar Charts**: Performance comparisons and rankings
- **Data Tables**: Detailed property information
- **KPI Cards**: Key performance indicators

## Sample Insights

The dashboard provides actionable insights such as:

- **Top Investment Opportunities**: Properties with highest ROI potential
- **Market Saturation Analysis**: Areas with growth potential vs. oversaturation
- **Competitive Clusters**: Market segments and positioning strategies
- **Neighborhood Rankings**: Best performing areas for investment
- **Operational Insights**: Host performance and optimization opportunities

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Inside Airbnb** for providing comprehensive Airbnb market data
- **Streamlit** for the excellent web app framework
- **Plotly** and **Folium** for interactive visualizations
- **Vancouver Open Data** for geographic and neighborhood information

## Contact

**Manvi Sharma** - [@sharma93manvi](https://github.com/sharma93manvi)

Project Link: [https://github.com/sharma93manvi/Vancouver-Airbnb-Investment-Intelligence-Dashboard](https://github.com/sharma93manvi/Vancouver-Airbnb-Investment-Intelligence-Dashboard)

## Future Enhancements

- [ ] Real-time data updates
- [ ] Advanced machine learning predictions
- [ ] Multi-city comparison analysis
- [ ] Mobile-responsive design improvements
- [ ] Export functionality for reports
- [ ] API integration for live data feeds
- [ ] Advanced filtering and search capabilities
- [ ] Investment portfolio tracking

---

**⭐ If you found this project helpful, please give it a star!**
