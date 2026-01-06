# 🛒 E-commerce Sales Dashboard

An interactive sales analytics dashboard built with Streamlit that provides comprehensive insights into e-commerce performance.

## 📊 Features

- **Real-time KPI Metrics**: Track total revenue, orders, average order value, and unique customers
- **Interactive Filters**: Filter by date range, categories, and regions
- **Visual Analytics**:
  - Revenue trends over time
  - Category performance comparison
  - Regional sales distribution
  - Payment method analysis
  - Top-performing products
  - Sales patterns by day of week
- **Key Insights**: Automated insights highlighting best performers
- **Transaction Details**: Detailed view of individual orders

## 🚀 Deployment Steps

### 1. Create GitHub Repository

1. Go to [GitHub](https://github.com) and create a new repository
2. Name it something like `ecommerce-dashboard`
3. Make it public (required for Streamlit Cloud free tier)

### 2. Upload Files

Upload these files to your repository:
- `app.py` (main application)
- `requirements.txt` (dependencies)
- `README.md` (this file)

You can do this via:
- **GitHub Web Interface**: Use "Add file" → "Upload files"
- **Git Command Line**:
  ```bash
  git init
  git add .
  git commit -m "Initial commit"
  git branch -M main
  git remote add origin https://github.com/YOUR_USERNAME/ecommerce-dashboard.git
  git push -u origin main
  ```

### 3. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
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