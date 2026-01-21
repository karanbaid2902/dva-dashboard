# 🛒 E-commerce Sales Dashboard

An interactive sales analytics dashboard built with Streamlit that provides comprehensive insights into e-commerce performance with advanced customer segmentation and financial metrics.

## 📊 Key Features

- **Executive Dashboard**: Real-time KPIs, revenue trends, category performance, regional analysis
- **Customer Analytics**: RFM segmentation, lifetime value distribution, satisfaction metrics
- **Advanced Metrics**: Monthly/quarterly trends, profit margin analysis, discount impact
- **Product Analysis**: Best-selling products, ABC classification, category profitability
- **Statistical Analysis**: Distribution analysis, correlation studies, summary statistics
- **Comparative Analysis**: Region vs category heatmaps, segment performance benchmarking
- **Performance Indicators**: Business health metrics, return analysis, monthly growth tracking
- **Interactive Filters**: Date range, categories, regions, customer segments
- **Period Comparison**: Year-over-year or custom period comparisons
- **Data Export**: Download filtered data as CSV for external analysis

## 🎯 Analytics Capabilities

- **RFM Segmentation**: Champions, Loyal, At Risk, Lost customer classification
- **Customer Lifetime Value**: Cohort-based LTV calculations
- **ABC Analysis**: Pareto-based product classification
- **Profit Margin Analysis**: Category-wise and product-wise margins
- **Return Rate Tracking**: By segment, category, and product
- **Satisfaction Correlation**: Relationship between satisfaction and purchase behavior
- **Regional Benchmarking**: Performance comparison across 5 regions
- **Seasonal Patterns**: Holiday, Prime Day, and New Year trends

## 🚀 Quick Start

### Local Installation
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Deployment on Streamlit Cloud
1. Push to GitHub: `https://github.com/YOUR_USERNAME/ecommerce-dashboard`
2. Go to `share.streamlit.io` and click "New app"
3. Connect your GitHub repository

## 📈 Data Features

- **100K+ Transactions**: Realistic sample data with real Amazon product prices
- **6 Product Categories**: Electronics, Books, Home & Kitchen, Fashion, Sports, Beauty
- **5 Regions**: US East, US West, Europe, Asia-Pacific, Latin America
- **3 Customer Segments**: Prime Members, Regular, New customers
- **365-Day History**: Full year of transactional data with seasonal patterns
- **Real Metrics**: Profit margins based on actual retail patterns

## 🔧 Technologies

- **Streamlit**: Interactive web framework
- **Plotly**: Advanced interactive visualizations
- **Pandas**: Data manipulation and analysis
- **NumPy/SciPy**: Numerical computing and statistics

## 📋 Dashboard Tabs

1. **Executive Summary** - Top-line metrics and trends
2. **Customer Analytics** - Segmentation and behavior analysis
3. **Advanced Metrics** - Detailed trend analysis
4. **Product & Segment** - Performance by product and category
5. **Statistical Analysis** - Distribution and summary metrics
6. **Comparative Analysis** - Cross-dimensional comparisons
7. **Performance & Health** - Business health indicators and growth metrics
4. Select your repository: `YOUR_USERNAME/ecommerce-dashboard`
5. Set:
   - **Branch**: `main`
   - **Main file path**: `app.py`
6. Click "Deploy"

Your dashboard will be live in a few minutes at: `https://YOUR_USERNAME-ecommerce-dashboard.streamlit.app`

## 💡 Key Insights from the Dashboard

### Revenue Analysis
- Track daily revenue trends to identify growth patterns
- Spot seasonal spikes (e.g., holiday shopping periods)
- Monitor month-over-month performance

### Category Performance
- **Electronics** typically generates highest revenue per order
- **Toys** and **Clothing** show strong holiday season performance
- Use insights to optimize inventory and marketing spend

### Regional Insights
- Identify high-performing regions for targeted campaigns
- Understand geographical customer distribution
- Optimize shipping and logistics based on demand

### Customer Behavior
- **Day of Week patterns**: Identify peak shopping days
- **Payment preferences**: Optimize checkout options
- **Average order value**: Set benchmarks for upselling

### Product Strategy
- Top 10 products drive significant revenue concentration
- Use best-sellers to inform procurement decisions
- Cross-sell opportunities with complementary products

## 📈 Business Recommendations

1. **Focus on high-revenue categories** for marketing investment
2. **Optimize inventory** for peak days and seasons
3. **Regional targeting** - increase presence in top-performing regions
4. **Payment optimization** - ensure preferred methods are seamless
5. **Product bundling** - combine top sellers for increased AOV

## 🛠️ Customization

To use your own data:
1. Replace the `generate_ecommerce_data()` function with your data loading code
2. Ensure your data has these columns: Date, Category, Product, Region, Payment_Method, Quantity, Total_Amount
3. Adjust visualizations as needed for your specific metrics

## 📝 Technical Notes

- Dashboard uses sample generated data (5,000 transactions over 12 months)
- Data includes realistic patterns: seasonality, category variations, regional distribution
- All visualizations are interactive with zoom, pan, and hover details
- Filters update all charts dynamically

## 📧 Support

For issues or questions, please open an issue in the GitHub repository.