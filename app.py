import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Page configuration
st.set_page_config(
    page_title="E-commerce Sales Dashboard",
    page_icon="🛒",
    layout="wide"
)

# Generate sample e-commerce data
@st.cache_data
def generate_ecommerce_data():
    np.random.seed(42)
    
    # Generate dates for the last 12 months
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Categories and products
    categories = ['Electronics', 'Clothing', 'Home & Garden', 'Books', 'Sports', 'Toys']
    products = {
        'Electronics': ['Laptop', 'Smartphone', 'Headphones', 'Tablet', 'Smart Watch'],
        'Clothing': ['T-Shirt', 'Jeans', 'Dress', 'Jacket', 'Shoes'],
        'Home & Garden': ['Lamp', 'Cushion', 'Plant', 'Kitchen Set', 'Rug'],
        'Books': ['Fiction', 'Non-Fiction', 'Educational', 'Comics', 'Biography'],
        'Sports': ['Yoga Mat', 'Dumbbells', 'Running Shoes', 'Bicycle', 'Protein Powder'],
        'Toys': ['Action Figure', 'Board Game', 'Puzzle', 'Doll', 'Building Blocks']
    }
    
    regions = ['North America', 'Europe', 'Asia', 'South America', 'Australia']
    payment_methods = ['Credit Card', 'Debit Card', 'PayPal', 'Cash on Delivery']
    
    # Generate transactions
    num_transactions = 5000
    data = []
    
    for _ in range(num_transactions):
        date = np.random.choice(dates)
        category = np.random.choice(categories)
        product = np.random.choice(products[category])
        region = np.random.choice(regions)
        payment = np.random.choice(payment_methods)
        
        # Price varies by category
        base_prices = {
            'Electronics': (200, 1500),
            'Clothing': (20, 150),
            'Home & Garden': (15, 300),
            'Books': (10, 50),
            'Sports': (25, 500),
            'Toys': (10, 100)
        }
        
        quantity = np.random.randint(1, 5)
        unit_price = np.random.uniform(*base_prices[category])
        total_amount = unit_price * quantity
        
        # Add some seasonality
        month = date.month
        if category == 'Clothing' and month in [11, 12]:
            total_amount *= 1.3  # Holiday boost
        if category == 'Toys' and month in [11, 12]:
            total_amount *= 1.5  # Holiday boost
        
        data.append({
            'Date': date,
            'Order_ID': f'ORD{_:05d}',
            'Category': category,
            'Product': product,
            'Region': region,
            'Payment_Method': payment,
            'Quantity': quantity,
            'Unit_Price': round(unit_price, 2),
            'Total_Amount': round(total_amount, 2),
            'Customer_ID': f'CUST{np.random.randint(1, 1000):04d}'
        })
    
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.to_period('M').astype(str)
    df['Year'] = df['Date'].dt.year
    df['Day_of_Week'] = df['Date'].dt.day_name()
    
    return df

# Load data
df = generate_ecommerce_data()

# Dashboard Title
st.title("🛒 E-commerce Sales Dashboard")
st.markdown("### Interactive Sales Analytics and Insights")

# Sidebar filters
st.sidebar.header("Filters")

# Date range filter
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(df['Date'].min(), df['Date'].max()),
    min_value=df['Date'].min(),
    max_value=df['Date'].max()
)

# Category filter
categories = st.sidebar.multiselect(
    "Select Categories",
    options=df['Category'].unique(),
    default=df['Category'].unique()
)

# Region filter
regions = st.sidebar.multiselect(
    "Select Regions",
    options=df['Region'].unique(),
    default=df['Region'].unique()
)

# Filter data
if len(date_range) == 2:
    mask = (
        (df['Date'] >= pd.to_datetime(date_range[0])) & 
        (df['Date'] <= pd.to_datetime(date_range[1])) &
        (df['Category'].isin(categories)) &
        (df['Region'].isin(regions))
    )
    filtered_df = df[mask]
else:
    filtered_df = df[
        (df['Category'].isin(categories)) &
        (df['Region'].isin(regions))
    ]

# Key Metrics
st.markdown("## 📊 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_revenue = filtered_df['Total_Amount'].sum()
    st.metric("Total Revenue", f"${total_revenue:,.2f}")

with col2:
    total_orders = len(filtered_df)
    st.metric("Total Orders", f"{total_orders:,}")

with col3:
    avg_order_value = filtered_df['Total_Amount'].mean()
    st.metric("Avg Order Value", f"${avg_order_value:,.2f}")

with col4:
    unique_customers = filtered_df['Customer_ID'].nunique()
    st.metric("Unique Customers", f"{unique_customers:,}")

st.markdown("---")

# Row 1: Revenue trends and category breakdown
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📈 Revenue Trend Over Time")
    revenue_by_date = filtered_df.groupby('Date')['Total_Amount'].sum().reset_index()
    fig_revenue = px.line(
        revenue_by_date,
        x='Date',
        y='Total_Amount',
        title='Daily Revenue Trend',
        labels={'Total_Amount': 'Revenue ($)', 'Date': 'Date'}
    )
    fig_revenue.update_traces(line_color='#1f77b4', line_width=2)
    st.plotly_chart(fig_revenue, use_container_width=True)

with col2:
    st.markdown("### 🏷️ Revenue by Category")
    revenue_by_category = filtered_df.groupby('Category')['Total_Amount'].sum().sort_values(ascending=False)
    fig_category = px.bar(
        x=revenue_by_category.index,
        y=revenue_by_category.values,
        title='Category Performance',
        labels={'x': 'Category', 'y': 'Revenue ($)'},
        color=revenue_by_category.values,
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig_category, use_container_width=True)

# Row 2: Regional performance and payment methods
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🌍 Regional Sales Distribution")
    revenue_by_region = filtered_df.groupby('Region')['Total_Amount'].sum()
    fig_region = px.pie(
        values=revenue_by_region.values,
        names=revenue_by_region.index,
        title='Sales by Region',
        hole=0.4
    )
    st.plotly_chart(fig_region, use_container_width=True)

with col2:
    st.markdown("### 💳 Payment Method Usage")
    payment_counts = filtered_df['Payment_Method'].value_counts()
    fig_payment = px.bar(
        x=payment_counts.index,
        y=payment_counts.values,
        title='Orders by Payment Method',
        labels={'x': 'Payment Method', 'y': 'Number of Orders'},
        color=payment_counts.values,
        color_continuous_scale='Greens'
    )
    st.plotly_chart(fig_payment, use_container_width=True)

# Row 3: Top products and day of week analysis
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🏆 Top 10 Products by Revenue")
    top_products = filtered_df.groupby('Product')['Total_Amount'].sum().sort_values(ascending=False).head(10)
    fig_products = px.bar(
        x=top_products.values,
        y=top_products.index,
        orientation='h',
        title='Best Selling Products',
        labels={'x': 'Revenue ($)', 'y': 'Product'},
        color=top_products.values,
        color_continuous_scale='Oranges'
    )
    st.plotly_chart(fig_products, use_container_width=True)

with col2:
    st.markdown("### 📅 Sales by Day of Week")
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    sales_by_day = filtered_df.groupby('Day_of_Week')['Total_Amount'].sum().reindex(day_order)
    fig_day = px.bar(
        x=sales_by_day.index,
        y=sales_by_day.values,
        title='Revenue by Day of Week',
        labels={'x': 'Day', 'y': 'Revenue ($)'},
        color=sales_by_day.values,
        color_continuous_scale='Purples'
    )
    st.plotly_chart(fig_day, use_container_width=True)

# Key Insights Section
st.markdown("---")
st.markdown("## 💡 Key Insights")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🎯 Best Category")
    best_category = filtered_df.groupby('Category')['Total_Amount'].sum().idxmax()
    best_category_revenue = filtered_df.groupby('Category')['Total_Amount'].sum().max()
    st.info(f"**{best_category}** leads with ${best_category_revenue:,.2f} in revenue")

with col2:
    st.markdown("### 🌟 Top Region")
    top_region = filtered_df.groupby('Region')['Total_Amount'].sum().idxmax()
    top_region_revenue = filtered_df.groupby('Region')['Total_Amount'].sum().max()
    st.success(f"**{top_region}** generates ${top_region_revenue:,.2f}")

with col3:
    st.markdown("### 📊 Peak Day")
    peak_day = filtered_df.groupby('Day_of_Week')['Total_Amount'].sum().idxmax()
    st.warning(f"**{peak_day}** has the highest sales")

# Detailed data table
st.markdown("---")
st.markdown("## 📋 Transaction Details")
st.dataframe(
    filtered_df[['Date', 'Order_ID', 'Category', 'Product', 'Region', 'Payment_Method', 'Quantity', 'Total_Amount']]
    .sort_values('Date', ascending=False)
    .head(100),
    use_container_width=True
)

# Footer
st.markdown("---")
st.markdown("*Dashboard created with Streamlit | Data updates in real-time based on filters*")