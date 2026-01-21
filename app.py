import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add Amazon logo at the very top
st.markdown('''
<div style="display: flex; align-items: center; justify-content: center; margin-bottom: 20px;">
    <img src="https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg" width="120" alt="Amazon Logo">
</div>
''', unsafe_allow_html=True)

st.markdown('<h1 style="text-align: center; margin-top: -10px;">Sales & Analytics Dashboard</h1>', unsafe_allow_html=True)
st.divider()

# Theme and styling
theme_color = st.sidebar.selectbox("🎨 Dashboard Theme", ["Light", "Dark", "Ocean", "Forest", "Sunset"], key="theme")

theme_styles = {
    "Light": """
        <style>
            .stMetric { background-color: #f0f2f6; padding: 15px; border-radius: 5px; color: #1f1f1f; border-left: 4px solid #FF9900; }
        </style>
    """,
    "Dark": """
        <style>
            .stMetric { background-color: #1a1a1a; padding: 15px; border-radius: 5px; color: #ffffff; }
        </style>
    """,
    "Ocean": """
        <style>
            .stMetric { background-color: #0a1428; padding: 15px; border-radius: 5px; color: #ffffff; border-left: 4px solid #0dcaf0; }
        </style>
    """,
    "Forest": """
        <style>
            .stMetric { background-color: #0d3b1f; padding: 15px; border-radius: 5px; color: #ffffff; border-left: 4px solid #198754; }
        </style>
    """,
    "Sunset": """
        <style>
            .stMetric { background-color: #3d1e2c; padding: 15px; border-radius: 5px; color: #ffffff; border-left: 4px solid #f66b6b; }
        </style>
    """
}

if theme_color in theme_styles:
    st.markdown(theme_styles[theme_color], unsafe_allow_html=True)

# Header
st.markdown('<h1 style="text-align: center;">📊 Amazon Sales & Analytics Dashboard</h1>', unsafe_allow_html=True)
st.divider()

@st.cache_data
def generate_ecommerce_data():
    try:
        np.random.seed(42)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        categories = ['Electronics', 'Books & Media', 'Home & Kitchen', 'Fashion', 'Sports & Outdoors', 'Beauty & Personal Care']
        
        products = {
            'Electronics': {'Fire TV Stick': 39.99, 'Echo Dot': 49.99, 'Kindle': 99.99, 'Ring Video Doorbell': 99.99, 'Echo Show': 149.99},
            'Books & Media': {'Science Fiction Novel': 14.99, 'Business Book': 19.99, 'Self-Help Guide': 16.99, 'Comic Book': 12.99, 'Audiobook': 17.99},
            'Home & Kitchen': {'Air Purifier': 89.99, 'Coffee Maker': 49.99, 'Bedding Set': 79.99, 'Cookware': 99.99, 'Smart Bulbs': 39.99},
            'Fashion': {'Running Shoes': 79.99, 'Denim Jeans': 49.99, 'Cotton T-Shirt': 19.99, 'Winter Jacket': 89.99, 'Backpack': 59.99},
            'Sports & Outdoors': {'Yoga Mat': 29.99, 'Dumbbells': 24.99, 'Tent': 129.99, 'Fitness Tracker': 149.99, 'Camping Gear': 89.99},
            'Beauty & Personal Care': {'Face Cream': 24.99, 'Shampoo': 12.99, 'Perfume': 49.99, 'Lipstick': 9.99, 'Moisturizer': 19.99}
        }
        
        regions = ['US East', 'US West', 'Europe', 'Asia-Pacific', 'Latin America']
        region_weights = [0.30, 0.20, 0.25, 0.18, 0.07]
        payment_methods = ['Amazon Wallet', 'Credit/Debit Card', 'Amazon Pay', 'COD']
        customer_segments = ['Prime Member', 'Regular', 'New']
        
        num_transactions = 100000
        data = []
        
        for i in range(num_transactions):
            date = pd.Timestamp(np.random.choice(dates))
            hour = np.random.randint(0, 24)
            minute = np.random.randint(0, 60)
            second = np.random.randint(0, 60)
            datetime_obj = date.replace(hour=hour, minute=minute, second=second)
            
            category_weights = [0.28, 0.12, 0.18, 0.20, 0.12, 0.10]
            category = np.random.choice(categories, p=category_weights)
            product = np.random.choice(list(products[category].keys()))
            unit_price = products[category][product] * np.random.uniform(0.95, 1.05)
            
            region = np.random.choice(regions, p=region_weights)
            payment = np.random.choice(payment_methods)
            customer_segment = np.random.choice(customer_segments, p=[0.45, 0.40, 0.15])
            
            profit_margins_by_category = {
                'Electronics': (0.35, 0.50), 'Books & Media': (0.40, 0.55), 'Home & Kitchen': (0.38, 0.52),
                'Fashion': (0.42, 0.58), 'Sports & Outdoors': (0.40, 0.55), 'Beauty & Personal Care': (0.38, 0.52)
            }
            
            cost_multiplier = np.random.uniform(*profit_margins_by_category[category])
            quantity = np.random.randint(1, 6)
            cost = unit_price * cost_multiplier
            
            discount_rate = 0
            if customer_segment == 'Prime Member':
                discount_rate = np.random.uniform(0.05, 0.12)
            elif customer_segment == 'Regular':
                discount_rate = np.random.uniform(0, 0.05)
            
            discount_amount = unit_price * quantity * discount_rate
            total_amount = (unit_price * quantity) - discount_amount
            profit = (total_amount - (cost * quantity))
            
            month = date.month
            if month in [11, 12]:
                if category == 'Electronics':
                    total_amount *= 2.2
                    profit *= 1.8
                elif category == 'Fashion':
                    total_amount *= 1.6
                    profit *= 1.4
                elif category == 'Home & Kitchen':
                    total_amount *= 1.4
                    profit *= 1.25
                else:
                    total_amount *= 1.35
                    profit *= 1.2
            elif month in [6, 7]:
                if category in ['Electronics', 'Sports & Outdoors']:
                    total_amount *= 1.4
                    profit *= 1.2
            elif month in [1]:
                if category in ['Sports & Outdoors', 'Beauty & Personal Care']:
                    total_amount *= 1.2
                    profit *= 1.1
            
            return_rate = 0.07 if customer_segment == 'New' else (0.03 if customer_segment == 'Regular' else 0.01)
            is_returned = np.random.random() < return_rate
            
            satisfaction_base = 4.6 if customer_segment == 'Prime Member' else (4.3 if customer_segment == 'Regular' else 4.0)
            if is_returned:
                satisfaction_score = np.random.uniform(2.0, 3.5)
            else:
                satisfaction_score = min(5.0, np.random.normal(satisfaction_base, 0.35))
            satisfaction_score = max(1, round(satisfaction_score, 1))
            
            data.append({
                'DateTime': datetime_obj,
                'Date': date.date(),
                'Time': f"{hour:02d}:{minute:02d}:{second:02d}",
                'Order_ID': f'AMZ{i:08d}',
                'Category': category,
                'Product': product,
                'Region': region,
                'Payment_Method': payment,
                'Customer_Segment': customer_segment,
                'Quantity': quantity,
                'Unit_Price': round(unit_price, 2),
                'Discount_Rate': round(discount_rate * 100, 2),
                'Total_Amount': round(total_amount, 2),
                'Cost': round(cost * quantity, 2),
                'Profit': round(profit, 2),
                'Profit_Margin': round((profit / total_amount * 100), 2) if total_amount > 0 else 0,
                'Is_Returned': is_returned,
                'Customer_ID': f'AMZN{np.random.randint(1, 5000):05d}',
                'Satisfaction_Score': satisfaction_score
            })
        
        df = pd.DataFrame(data)
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.to_period('M').astype(str)
        df['Day_of_Week'] = df['Date'].dt.day_name()
        df['Quarter'] = 'Q' + df['Date'].dt.quarter.astype(str)
        
        return df
    except Exception as e:
        st.error(f"Error generating data: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def calculate_rfm(df):
    today = df['Date'].max()
    rfm = df[~df['Is_Returned']].groupby('Customer_ID').agg({
        'Date': lambda x: (today - x.max()).days,
        'Order_ID': 'count',
        'Total_Amount': 'sum'
    }).reset_index()
    rfm.columns = ['Customer_ID', 'Recency', 'Frequency', 'Monetary']
    
    try:
        rfm['R_Score'] = pd.qcut(rfm['Recency'], q=4, labels=[4, 3, 2, 1], duplicates='drop')
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=4, labels=[1, 2, 3, 4], duplicates='drop')
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=4, labels=[1, 2, 3, 4], duplicates='drop')
    except:
        rfm['R_Score'] = 2
        rfm['F_Score'] = 2
        rfm['M_Score'] = 2
    
    rfm['RFM_Score'] = rfm['R_Score'].astype(int) + rfm['F_Score'].astype(int) + rfm['M_Score'].astype(int)
    
    def segment(score):
        if score >= 10: return 'Champions'
        elif score >= 8: return 'Loyal'
        elif score >= 6: return 'At Risk'
        else: return 'Lost'
    
    rfm['Segment'] = rfm['RFM_Score'].apply(segment)
    return rfm

def generate_insights(df):
    insights = []
    try:
        top_category = df.groupby('Category')['Total_Amount'].sum().idxmax()
        top_cat_pct = (df[df['Category'] == top_category]['Total_Amount'].sum() / df['Total_Amount'].sum() * 100)
        insights.append(f"📈 **{top_category}** leads with {top_cat_pct:.1f}% of revenue")
        
        top_region = df.groupby('Region')['Total_Amount'].sum().idxmax()
        insights.append(f"🌍 **{top_region}** is top region")
        
        total_profit = df['Profit'].sum()
        total_revenue = df['Total_Amount'].sum()
        profit_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0
        insights.append(f"💹 Profit margin: **{profit_margin:.1f}%**")
        
        return_rate = (df['Is_Returned'].sum() / len(df) * 100)
        insights.append(f"📦 Return rate: **{return_rate:.1f}%**")
        
        avg_satisfaction = df['Satisfaction_Score'].mean()
        insights.append(f"⭐ Satisfaction: **{avg_satisfaction:.1f}**/5")
        
        aov = df['Total_Amount'].mean()
        insights.append(f"🛍️ AOV: **${aov:.2f}**")
    except:
        insights = ["No insights available"]
    
    return insights

try:
    df = generate_ecommerce_data()
    
    if df.empty:
        st.error("Failed to load data")
        st.stop()
    
    st.sidebar.header("🔍 Global Filters")
    
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    
    date_range = st.sidebar.date_input("Select Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    categories_filter = st.sidebar.multiselect("Categories", sorted(df['Category'].unique()), default=df['Category'].unique())
    regions_filter = st.sidebar.multiselect("Regions", sorted(df['Region'].unique()), default=df['Region'].unique())
    segments_filter = st.sidebar.multiselect("Customer Segments", sorted(df['Customer_Segment'].unique()), default=df['Customer_Segment'].unique())
    
    if len(date_range) == 2:
        filtered_df = df[
            (df['Date'].dt.date >= date_range[0]) & 
            (df['Date'].dt.date <= date_range[1]) &
            (df['Category'].isin(categories_filter)) &
            (df['Region'].isin(regions_filter)) &
            (df['Customer_Segment'].isin(segments_filter))
        ]
    else:
        filtered_df = df[
            (df['Category'].isin(categories_filter)) &
            (df['Region'].isin(regions_filter)) &
            (df['Customer_Segment'].isin(segments_filter))
        ]
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Executive Dashboard",
        "👥 Customer Analytics", 
        "🎯 Products & Trends",
        "📊 Advanced Analysis",
        "⚡ Health Metrics",
        "🔮 Forecasting"
    ])
    
    with tab1:
        st.markdown("## Executive Summary & KPIs")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        total_revenue = filtered_df['Total_Amount'].sum()
        total_profit = filtered_df['Profit'].sum()
        
        with col1:
            st.metric("💰 Revenue", f"${total_revenue:,.0f}")
        with col2:
            st.metric("📈 Profit", f"${total_profit:,.0f}")
        with col3:
            st.metric("📦 Orders", f"{len(filtered_df):,}")
        with col4:
            st.metric("👥 Customers", f"{filtered_df['Customer_ID'].nunique():,}")
        with col5:
            st.metric("💹 Margin %", f"{(total_profit/total_revenue*100):.1f}%")
        
        st.markdown("---")
        st.markdown("### 💡 Key Insights")
        insights = generate_insights(filtered_df)
        cols = st.columns(3)
        for idx, insight in enumerate(insights):
            with cols[idx % 3]:
                st.info(insight)
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Weekly Revenue & Profit")
            weekly = filtered_df.copy()
            weekly['Week'] = weekly['Date'].dt.to_period('W').apply(lambda r: r.start_time)
            weekly_data = weekly.groupby('Week').agg({'Total_Amount': 'sum', 'Profit': 'sum'}).reset_index()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=weekly_data['Week'], y=weekly_data['Total_Amount'], name='Revenue', line=dict(color='#2E86AB', width=3), mode='lines+markers'))
            fig.add_trace(go.Scatter(x=weekly_data['Week'], y=weekly_data['Profit'], name='Profit', line=dict(color='#A23B72', width=3), mode='lines+markers'))
            fig.update_xaxes(title_text="Week")
            fig.update_yaxes(title_text="Amount ($)")
            fig.update_layout(hovermode='x unified', plot_bgcolor='rgba(20,20,30,0.3)', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Revenue by Category")
            cat_rev = filtered_df.groupby('Category')['Total_Amount'].sum().sort_values(ascending=False)
            fig = px.pie(values=cat_rev.values, names=cat_rev.index)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Top 10 Products")
            top_products = filtered_df.groupby('Product')['Total_Amount'].sum().nlargest(10)
            fig = px.bar(x=top_products.values, y=top_products.index, orientation='h', color=top_products.values, color_continuous_scale='Blues')
            fig.update_xaxes(title_text="Revenue ($)")
            fig.update_yaxes(title_text="Product Name")
            fig.update_layout(plot_bgcolor='rgba(20,20,30,0.3)', height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Orders by Day of Week")
            dow_data = filtered_df.groupby('Day_of_Week')['Total_Amount'].sum().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
            fig = px.bar(x=dow_data.index, y=dow_data.values, color=dow_data.values, color_continuous_scale='Reds')
            fig.update_xaxes(title_text="Day of Week")
            fig.update_yaxes(title_text="Revenue ($)")
            fig.update_layout(plot_bgcolor='rgba(20,20,30,0.3)', height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("## Customer Segmentation & Behavior")
        rfm = calculate_rfm(filtered_df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### RFM Segments Distribution")
            seg_dist = rfm['Segment'].value_counts()
            fig = px.pie(values=seg_dist.values, names=seg_dist.index, hole=0.4)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Total Value by Segment")
            seg_monetary = rfm.groupby('Segment')['Monetary'].sum().sort_values(ascending=False)
            fig = px.bar(x=seg_monetary.index, y=seg_monetary.values, color=seg_monetary.values, color_continuous_scale='Reds')
            fig.update_xaxes(title_text="Customer Segment")
            fig.update_yaxes(title_text="Total Value ($)")
            fig.update_layout(plot_bgcolor='rgba(20,20,30,0.3)', height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Satisfaction by Segment")
            scatter_data = filtered_df.groupby('Customer_Segment').agg({'Satisfaction_Score': 'mean', 'Total_Amount': 'sum'}).reset_index()
            fig = px.scatter(scatter_data, x='Customer_Segment', y='Satisfaction_Score', size='Total_Amount', color='Satisfaction_Score', color_continuous_scale='Greens')
            fig.update_xaxes(title_text="Customer Segment")
            fig.update_yaxes(title_text="Average Satisfaction Score")
            fig.update_layout(plot_bgcolor='rgba(20,20,30,0.3)', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Customer Frequency Range")
            freq_range = pd.cut(rfm['Frequency'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            freq_dist = freq_range.value_counts().sort_index()
            fig = px.bar(x=freq_dist.index.astype(str), y=freq_dist.values, color=freq_dist.values, color_continuous_scale='Blues')
            fig.update_xaxes(title_text="Frequency Level")
            fig.update_yaxes(title_text="Number of Customers")
            fig.update_layout(plot_bgcolor='rgba(20,20,30,0.3)', height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("## Products & Trends")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Top 10 Best-Selling Products")
            top_products = filtered_df['Product'].value_counts().head(10)
            fig = px.bar(x=top_products.values, y=top_products.index, orientation='h', color=top_products.values, color_continuous_scale='Teal')
            fig.update_xaxes(title_text="Number of Orders")
            fig.update_yaxes(title_text="Product Name")
            fig.update_layout(plot_bgcolor='rgba(20,20,30,0.3)', height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Monthly Revenue Trend")
            monthly = filtered_df.groupby('Month').agg({'Total_Amount': 'sum', 'Profit': 'sum'}).reset_index()
            fig = px.line(monthly, x='Month', y=['Total_Amount', 'Profit'], markers=True)
            fig.update_xaxes(title_text="Month")
            fig.update_yaxes(title_text="Amount ($)")
            fig.update_layout(plot_bgcolor='rgba(20,20,30,0.3)', hovermode='x unified', height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("## Advanced Analytics")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Revenue by Category")
            cat_revenue = filtered_df.groupby('Category')['Total_Amount'].sum().sort_values(ascending=False)
            fig = px.pie(values=cat_revenue.values, names=cat_revenue.index, hole=0.3)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Region Revenue")
            region_revenue = filtered_df.groupby('Region')['Total_Amount'].sum().sort_values(ascending=False)
            fig = px.bar(x=region_revenue.index, y=region_revenue.values, color=region_revenue.values, color_continuous_scale='Blues')
            fig.update_xaxes(title_text="Region")
            fig.update_yaxes(title_text="Revenue ($)")
            fig.update_layout(plot_bgcolor='rgba(20,20,30,0.3)', height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.markdown("## Health Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📦 Orders", f"{len(filtered_df):,}")
        with col2:
            st.metric("👥 Customers", f"{filtered_df['Customer_ID'].nunique():,}")
        with col3:
            st.metric("💰 COGS", f"${filtered_df['Cost'].sum():,.0f}")
        with col4:
            st.metric("⭐ Satisfaction", f"{filtered_df['Satisfaction_Score'].mean():.1f}/5.0")
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Return Rates by Category")
            returns_data = filtered_df.groupby('Category').agg({'Is_Returned': lambda x: (x.sum() / len(x) * 100), 'Total_Amount': 'sum'}).reset_index()
            returns_data.columns = ['Category', 'ReturnRate', 'Revenue']
            fig = px.scatter(returns_data, x='ReturnRate', y='Revenue', color='ReturnRate', text='Category', color_continuous_scale='Reds')
            fig.update_xaxes(title_text="Return Rate (%)")
            fig.update_yaxes(title_text="Revenue ($)")
            fig.update_layout(plot_bgcolor='rgba(20,20,30,0.3)', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Monthly Revenue Trend")
            monthly_revenue = filtered_df.groupby('Month')['Total_Amount'].sum()
            fig = px.line(x=monthly_revenue.index, y=monthly_revenue.values, markers=True, color_discrete_sequence=['#2E86AB'])
            fig.update_xaxes(title_text="Month")
            fig.update_yaxes(title_text="Revenue ($)")
            fig.update_layout(plot_bgcolor='rgba(20,20,30,0.3)', height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab6:
        st.markdown("## Forecasting & Predictions")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Payment Methods")
            payment_data = filtered_df.groupby('Payment_Method')['Total_Amount'].sum()
            fig = px.pie(values=payment_data.values, names=payment_data.index, hole=0.4)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Profit Margin by Category")
            margin_by_cat = filtered_df.groupby('Category')['Profit_Margin'].mean().sort_values(ascending=False)
            fig = px.bar(x=margin_by_cat.index, y=margin_by_cat.values, color=margin_by_cat.values, color_continuous_scale='RdYlGn')
            fig.update_xaxes(title_text="Category")
            fig.update_yaxes(title_text="Average Profit Margin (%)")
            fig.update_layout(plot_bgcolor='rgba(20,20,30,0.3)', height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    import traceback
    st.write(traceback.format_exc())