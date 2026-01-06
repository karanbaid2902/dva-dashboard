import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="E-commerce Sales Dashboard",
    page_icon="🛒",
    layout="wide"
)

# Generate sample e-commerce data
@st.cache_data
def generate_ecommerce_data():
    try:
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
        
        for i in range(num_transactions):
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
                total_amount *= 1.3
            if category == 'Toys' and month in [11, 12]:
                total_amount *= 1.5
            
            data.append({
                'Date': date,
                'Order_ID': f'ORD{i:05d}',
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
    except Exception as e:
        st.error(f"Error generating data: {str(e)}")
        return pd.DataFrame()

# Main app
try:
    # Load data
    df = generate_ecommerce_data()
    
    if df.empty:
        st.error("Failed to load data")
        st.stop()
    
    # Dashboard Title
    st.title("🛒 E-commerce Sales Dashboard")
    st.markdown("### Interactive Sales Analytics and Insights")
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Date range filter
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Category filter
    categories = st.sidebar.multiselect(
        "Select Categories",
        options=sorted(df['Category'].unique()),
        default=df['Category'].unique()
    )
    
    # Region filter
    regions = st.sidebar.multiselect(
        "Select Regions",
        options=sorted(df['Region'].unique()),
        default=df['Region'].unique()
    )
    
    # Filter data
    if len(date_range) == 2:
        mask = (
            (df['Date'].dt.date >= date_range[0]) & 
            (df['Date'].dt.date <= date_range[1]) &
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
            title='Daily Revenue Trend'
        )
        fig_revenue.update_traces(line_color='#1f77b4', line_width=2)
        fig_revenue.update_layout(
            xaxis_title="Date",
            yaxis_title="Revenue ($)",
            hovermode='x unified'
        )
        st.plotly_chart(fig_revenue, use_container_width=True)
    
    with col2:
        st.markdown("### 🏷️ Revenue by Category")
        revenue_by_category = filtered_df.groupby('Category')['Total_Amount'].sum().sort_values(ascending=False)
        fig_category = px.bar(
            x=revenue_by_category.index,
            y=revenue_by_category.values,
            title='Category Performance',
            color=revenue_by_category.values,
            color_continuous_scale='Blues'
        )
        fig_category.update_layout(
            xaxis_title="Category",
            yaxis_title="Revenue ($)",
            showlegend=False
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
            color=payment_counts.values,
            color_continuous_scale='Greens'
        )
        fig_payment.update_layout(
            xaxis_title="Payment Method",
            yaxis_title="Number of Orders",
            showlegend=False
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
            color=top_products.values,
            color_continuous_scale='Oranges'
        )
        fig_products.update_layout(
            xaxis_title="Revenue ($)",
            yaxis_title="Product",
            showlegend=False
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
            color=sales_by_day.values,
            color_continuous_scale='Purples'
        )
        fig_day.update_layout(
            xaxis_title="Day",
            yaxis_title="Revenue ($)",
            showlegend=False
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
    
    display_df = filtered_df[['Date', 'Order_ID', 'Category', 'Product', 'Region', 'Payment_Method', 'Quantity', 'Total_Amount']].copy()
    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
    display_df = display_df.sort_values('Date', ascending=False).head(100)
    
    st.dataframe(display_df, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("*Dashboard created with Streamlit | Data updates in real-time based on filters*")

except Exception as e:
    st.error(f"Application Error: {str(e)}")
    st.info("Please check the logs for more details or contact support.")