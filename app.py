import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Amazon Analytics", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

# Custom CSS - Fixed KPI visibility
st.markdown("""
<style>
    .stMetric { background: linear-gradient(135deg, #232526 0%, #414345 100%); padding: 18px; border-radius: 12px; border-left: 5px solid #FF9900; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    .stMetric label { color: #FFFFFF !important; font-weight: 600; font-size: 14px; }
    .stMetric [data-testid="stMetricValue"] { color: #FF9900 !important; font-size: 28px !important; font-weight: 700; }
    .stMetric [data-testid="stMetricDelta"] { color: #00D4AA !important; }
    .block-container { padding-top: 1rem; }
    h1, h2, h3 { color: #FF9900; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1>🛒 Amazon Sales Analytics</h1>', unsafe_allow_html=True)
st.caption("Real-Time Business Intelligence Dashboard | 50K+ Transactions")
st.divider()

@st.cache_data
def generate_data():
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=365, freq='D')
    categories = ['Electronics', 'Books', 'Home & Kitchen', 'Fashion', 'Sports', 'Beauty']
    products = {
        'Electronics': ['Fire TV Stick', 'Echo Dot', 'Kindle', 'Ring Doorbell', 'Echo Show'],
        'Books': ['Fiction Novel', 'Business Book', 'Self-Help', 'Comics', 'Audiobook'],
        'Home & Kitchen': ['Air Purifier', 'Coffee Maker', 'Bedding Set', 'Cookware', 'Smart Bulbs'],
        'Fashion': ['Running Shoes', 'Jeans', 'T-Shirt', 'Jacket', 'Backpack'],
        'Sports': ['Yoga Mat', 'Dumbbells', 'Tent', 'Fitness Tracker', 'Camping Gear'],
        'Beauty': ['Face Cream', 'Shampoo', 'Perfume', 'Lipstick', 'Moisturizer']
    }
    prices = {'Electronics': 89, 'Books': 16, 'Home & Kitchen': 72, 'Fashion': 60, 'Sports': 85, 'Beauty': 23}
    regions = ['US East', 'US West', 'Europe', 'Asia-Pacific', 'Latin America']
    segments = ['Prime', 'Regular', 'New']
    
    n = 50000
    data = []
    for i in range(n):
        cat = np.random.choice(categories, p=[0.28, 0.12, 0.18, 0.20, 0.12, 0.10])
        prod = np.random.choice(products[cat])
        price = prices[cat] * np.random.uniform(0.8, 1.2)
        qty = np.random.randint(1, 5)
        segment = np.random.choice(segments, p=[0.45, 0.40, 0.15])
        discount = np.random.uniform(0.05, 0.12) if segment == 'Prime' else np.random.uniform(0, 0.05)
        revenue = price * qty * (1 - discount)
        profit = revenue * np.random.uniform(0.25, 0.45)
        date = np.random.choice(dates)
        
        # Seasonal boost
        month = pd.Timestamp(date).month
        if month in [11, 12]: revenue *= 1.5; profit *= 1.3
        
        data.append({
            'Date': date, 'Category': cat, 'Product': prod, 
            'Region': np.random.choice(regions, p=[0.30, 0.20, 0.25, 0.18, 0.07]),
            'Segment': segment, 'Quantity': qty, 'Revenue': round(revenue, 2),
            'Profit': round(profit, 2), 'Margin': round(profit/revenue*100, 1),
            'Returned': np.random.random() < 0.03,
            'Rating': round(np.clip(np.random.normal(4.3, 0.5), 1, 5), 1),
            'Customer_ID': f'C{np.random.randint(1, 3000):04d}'
        })
    
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.strftime('%Y-%m')
    df['Day'] = df['Date'].dt.day_name()
    df['Week'] = df['Date'].dt.to_period('W').apply(lambda x: x.start_time)
    return df

df = generate_data()

# Sidebar Filters
st.sidebar.header("🔍 Filters")

# Reset Filters Button
if st.sidebar.button("🔄 Reset All Filters", use_container_width=True):
    st.rerun()

st.sidebar.divider()
date_range = st.sidebar.date_input("Date Range", value=(df['Date'].min(), df['Date'].max()))
cats = st.sidebar.multiselect("Categories", df['Category'].unique(), default=df['Category'].unique())
regions = st.sidebar.multiselect("Regions", df['Region'].unique(), default=df['Region'].unique())
segments = st.sidebar.multiselect("Segments", df['Segment'].unique(), default=df['Segment'].unique())

# Apply filters
fdf = df[(df['Date'].dt.date >= date_range[0]) & (df['Date'].dt.date <= date_range[1]) & 
         (df['Category'].isin(cats)) & (df['Region'].isin(regions)) & (df['Segment'].isin(segments))]

# Export
st.sidebar.divider()
st.sidebar.download_button("📥 Export CSV", fdf.to_csv(index=False), "amazon_data.csv", "text/csv")

# KPI Row
st.markdown("### 📊 Key Performance Indicators")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("💰 Revenue", f"${fdf['Revenue'].sum():,.0f}")
c2.metric("📈 Profit", f"${fdf['Profit'].sum():,.0f}")
c3.metric("📦 Orders", f"{len(fdf):,}")
c4.metric("👥 Customers", f"{fdf['Customer_ID'].nunique():,}")
c5.metric("💹 Margin", f"{fdf['Profit'].sum()/fdf['Revenue'].sum()*100:.1f}%")
c6.metric("⭐ Rating", f"{fdf['Rating'].mean():.2f}")

st.divider()

# Tabs - 5 focused sections
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "👥 Customers", "📦 Products", "📈 Trends", "🔍 Insights"])

# =============================================================================
# TAB 1: OVERVIEW (5 charts)
# =============================================================================
with tab1:
    col1, col2 = st.columns(2)
    
    # Chart 1: Weekly Revenue & Profit Trend
    with col1:
        st.markdown("##### 📈 Weekly Revenue & Profit")
        weekly = fdf.groupby('Week').agg({'Revenue': 'sum', 'Profit': 'sum'}).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=weekly['Week'], y=weekly['Revenue'], name='Revenue', 
                                  line=dict(color='#FF9900', width=3), fill='tozeroy'))
        fig.add_trace(go.Scatter(x=weekly['Week'], y=weekly['Profit'], name='Profit', 
                                  line=dict(color='#00D4AA', width=3)))
        fig.update_layout(height=350, template='plotly_dark', hovermode='x unified', margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    
    # Chart 2: Revenue by Category (Donut)
    with col2:
        st.markdown("##### 🎯 Revenue by Category")
        cat_rev = fdf.groupby('Category')['Revenue'].sum().sort_values(ascending=False)
        fig = px.pie(values=cat_rev.values, names=cat_rev.index, hole=0.5,
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=350, template='plotly_dark', margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    # Chart 3: Regional Performance
    with col1:
        st.markdown("##### 🌍 Regional Performance")
        region_data = fdf.groupby('Region').agg({'Revenue': 'sum', 'Profit': 'sum'}).sort_values('Revenue', ascending=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(y=region_data.index, x=region_data['Revenue'], name='Revenue', orientation='h', marker_color='#FF9900'))
        fig.add_trace(go.Bar(y=region_data.index, x=region_data['Profit'], name='Profit', orientation='h', marker_color='#00D4AA'))
        fig.update_layout(height=300, template='plotly_dark', barmode='group', margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    
    # Chart 4: Payment by Day of Week (Heatmap-style bar)
    with col2:
        st.markdown("##### 📅 Revenue by Day of Week")
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow = fdf.groupby('Day')['Revenue'].sum().reindex(day_order)
        fig = px.bar(x=dow.index, y=dow.values, color=dow.values, color_continuous_scale='Oranges')
        fig.update_layout(height=300, template='plotly_dark', showlegend=False, margin=dict(t=10))
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Chart 5: Category vs Region Heatmap
    st.markdown("##### 🔥 Revenue Heatmap: Category × Region")
    heatmap_data = fdf.pivot_table(values='Revenue', index='Region', columns='Category', aggfunc='sum').fillna(0)
    fig = px.imshow(heatmap_data, text_auto='.0f', color_continuous_scale='YlOrRd', aspect='auto')
    fig.update_layout(height=280, template='plotly_dark', margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 2: CUSTOMERS (5 charts)
# =============================================================================
with tab2:
    # RFM Analysis
    today = fdf['Date'].max()
    rfm = fdf.groupby('Customer_ID').agg({
        'Date': lambda x: (today - x.max()).days,
        'Revenue': ['count', 'sum']
    }).reset_index()
    rfm.columns = ['Customer_ID', 'Recency', 'Frequency', 'Monetary']
    rfm['Score'] = pd.qcut(rfm['Monetary'], 4, labels=[1,2,3,4]).astype(int) + \
                   pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1,2,3,4]).astype(int)
    rfm['Segment'] = rfm['Score'].apply(lambda x: 'Champions' if x >= 7 else ('Loyal' if x >= 5 else ('At Risk' if x >= 3 else 'Lost')))
    
    col1, col2 = st.columns(2)
    
    # Chart 6: RFM Segment Distribution
    with col1:
        st.markdown("##### 👥 Customer Segments (RFM)")
        seg_dist = rfm['Segment'].value_counts()
        colors = {'Champions': '#00D4AA', 'Loyal': '#FF9900', 'At Risk': '#FFD93D', 'Lost': '#FF6B6B'}
        fig = px.pie(values=seg_dist.values, names=seg_dist.index, hole=0.5,
                     color=seg_dist.index, color_discrete_map=colors)
        fig.update_layout(height=350, template='plotly_dark', margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    
    # Chart 7: Revenue by Segment
    with col2:
        st.markdown("##### 💰 Revenue by Customer Segment")
        seg_rev = rfm.groupby('Segment')['Monetary'].sum().sort_values(ascending=True)
        fig = px.bar(y=seg_rev.index, x=seg_rev.values, orientation='h',
                     color=seg_rev.index, color_discrete_map=colors)
        fig.update_layout(height=350, template='plotly_dark', showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    # Chart 8: Recency vs Frequency Scatter
    with col1:
        st.markdown("##### 🎯 Recency vs Frequency")
        fig = px.scatter(rfm.sample(min(500, len(rfm))), x='Recency', y='Frequency', 
                        size='Monetary', color='Segment', color_discrete_map=colors, opacity=0.7)
        fig.update_layout(height=350, template='plotly_dark', margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    
    # Chart 9: Customer Rating Distribution
    with col2:
        st.markdown("##### ⭐ Rating Distribution")
        fig = px.histogram(fdf, x='Rating', nbins=20, color_discrete_sequence=['#FF9900'])
        fig.update_layout(height=350, template='plotly_dark', margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    
    # Chart 10: Segment Performance Table
    st.markdown("##### 📊 Segment Performance")
    seg_perf = fdf.groupby('Segment').agg({
        'Revenue': 'sum', 'Profit': 'sum', 'Customer_ID': 'nunique', 'Rating': 'mean'
    }).round(2)
    seg_perf.columns = ['Revenue ($)', 'Profit ($)', 'Customers', 'Avg Rating']
    seg_perf['Revenue ($)'] = seg_perf['Revenue ($)'].apply(lambda x: f"${x:,.0f}")
    seg_perf['Profit ($)'] = seg_perf['Profit ($)'].apply(lambda x: f"${x:,.0f}")
    st.dataframe(seg_perf, use_container_width=True)

# =============================================================================
# TAB 3: PRODUCTS (5 charts)
# =============================================================================
with tab3:
    col1, col2 = st.columns(2)
    
    # Chart 11: Top 10 Products by Revenue
    with col1:
        st.markdown("##### 🏆 Top 10 Products by Revenue")
        top_prod = fdf.groupby('Product')['Revenue'].sum().nlargest(10).sort_values()
        fig = px.bar(y=top_prod.index, x=top_prod.values, orientation='h',
                     color=top_prod.values, color_continuous_scale='Oranges')
        fig.update_layout(height=400, template='plotly_dark', showlegend=False, margin=dict(t=10))
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Chart 12: Category Profit Margin Box Plot
    with col2:
        st.markdown("##### 📦 Profit Margin by Category")
        fig = px.box(fdf, x='Category', y='Margin', color='Category',
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=400, template='plotly_dark', showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    # Chart 13: ABC Analysis
    with col1:
        st.markdown("##### 📊 ABC Product Classification")
        prod_rev = fdf.groupby('Product')['Revenue'].sum().sort_values(ascending=False)
        cumsum_pct = (prod_rev.cumsum() / prod_rev.sum() * 100)
        abc = pd.DataFrame({
            'Product': prod_rev.index[:15],
            'Revenue': prod_rev.values[:15],
            'Class': ['A' if p <= 70 else ('B' if p <= 90 else 'C') for p in cumsum_pct.values[:15]]
        })
        fig = px.bar(abc, y='Product', x='Revenue', color='Class', orientation='h',
                     color_discrete_map={'A': '#00D4AA', 'B': '#FFD93D', 'C': '#FF6B6B'})
        fig.update_layout(height=400, template='plotly_dark', margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    
    # Chart 14: Category Sales Funnel
    with col2:
        st.markdown("##### 🔻 Category Orders Funnel")
        cat_orders = fdf.groupby('Category')['Revenue'].count().sort_values(ascending=False)
        fig = px.funnel(y=cat_orders.index, x=cat_orders.values)
        fig.update_traces(marker_color=['#FF9900', '#FFB347', '#FFD93D', '#00D4AA', '#4ECDC4', '#95E1D3'])
        fig.update_layout(height=400, template='plotly_dark', margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    
    # Chart 15: Return Rate by Category
    st.markdown("##### 🔄 Return Rate by Category")
    return_rate = fdf.groupby('Category')['Returned'].mean() * 100
    fig = px.bar(x=return_rate.index, y=return_rate.values, color=return_rate.values,
                 color_continuous_scale='Reds', text=return_rate.values.round(1))
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(height=280, template='plotly_dark', showlegend=False, margin=dict(t=10))
    fig.update_coloraxes(showscale=False)
    fig.update_yaxes(title='Return Rate (%)')
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 4: TRENDS & FORECASTING (5 charts)
# =============================================================================
with tab4:
    col1, col2 = st.columns(2)
    
    # Chart 16: Monthly Revenue Trend with Moving Average
    with col1:
        st.markdown("##### 📈 Monthly Revenue Trend")
        monthly = fdf.groupby('Month')['Revenue'].sum().reset_index()
        monthly['MA'] = monthly['Revenue'].rolling(3, min_periods=1).mean()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=monthly['Month'], y=monthly['Revenue'], name='Revenue', marker_color='#FF9900', opacity=0.7))
        fig.add_trace(go.Scatter(x=monthly['Month'], y=monthly['MA'], name='3-Month MA', line=dict(color='#00D4AA', width=3)))
        fig.update_layout(height=350, template='plotly_dark', hovermode='x unified', margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    
    # Chart 17: Category Trend (Stacked Area)
    with col2:
        st.markdown("##### 📊 Category Revenue Over Time")
        cat_monthly = fdf.groupby(['Month', 'Category'])['Revenue'].sum().reset_index()
        fig = px.area(cat_monthly, x='Month', y='Revenue', color='Category',
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=350, template='plotly_dark', margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    # Chart 18: Segment Growth Comparison
    with col1:
        st.markdown("##### 👥 Segment Revenue Trend")
        seg_monthly = fdf.groupby(['Month', 'Segment'])['Revenue'].sum().reset_index()
        fig = px.line(seg_monthly, x='Month', y='Revenue', color='Segment', markers=True,
                      color_discrete_map={'Prime': '#FF9900', 'Regular': '#00D4AA', 'New': '#4ECDC4'})
        fig.update_layout(height=350, template='plotly_dark', hovermode='x unified', margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    
    # Chart 19: Profit Trend by Region
    with col2:
        st.markdown("##### 🌍 Regional Profit Trend")
        reg_monthly = fdf.groupby(['Month', 'Region'])['Profit'].sum().reset_index()
        fig = px.line(reg_monthly, x='Month', y='Profit', color='Region', markers=True)
        fig.update_layout(height=350, template='plotly_dark', hovermode='x unified', margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    
    # Chart 20: Revenue Forecast
    st.markdown("##### 🔮 Revenue Forecast (Next 30 Days)")
    daily = fdf.groupby(fdf['Date'].dt.date)['Revenue'].sum().reset_index()
    daily.columns = ['Date', 'Revenue']
    daily['Date'] = pd.to_datetime(daily['Date'])
    
    # Simple trend forecast
    x = np.arange(len(daily))
    z = np.polyfit(x, daily['Revenue'].values, 1)
    p = np.poly1d(z)
    
    future_x = np.arange(len(daily), len(daily) + 30)
    future_dates = [daily['Date'].max() + timedelta(days=i) for i in range(1, 31)]
    forecast = pd.DataFrame({'Date': future_dates, 'Forecast': np.maximum(p(future_x), 0)})
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily['Date'], y=daily['Revenue'], name='Historical', 
                              line=dict(color='#FF9900', width=2), fill='tozeroy', fillcolor='rgba(255,153,0,0.2)'))
    fig.add_trace(go.Scatter(x=forecast['Date'], y=forecast['Forecast'], name='Forecast',
                              line=dict(color='#00D4AA', width=3, dash='dash')))
    fig.update_layout(height=350, template='plotly_dark', hovermode='x unified', margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📊 Avg Daily Revenue", f"${daily['Revenue'].mean():,.0f}")
    c2.metric("🔮 Forecast Avg", f"${forecast['Forecast'].mean():,.0f}")
    c3.metric("📈 Trend", f"{(p[1]/daily['Revenue'].mean()*100):+.1f}%/day")
    c4.metric("🎯 30-Day Projection", f"${forecast['Forecast'].sum():,.0f}")

# =============================================================================
# TAB 5: INSIGHTS (4 charts)
# =============================================================================
with tab5:
    st.markdown("### 💡 Business Insights & Analysis")
    
    # Key Insights Cards
    col1, col2, col3 = st.columns(3)
    
    top_cat = fdf.groupby('Category')['Revenue'].sum().idxmax()
    top_cat_pct = fdf.groupby('Category')['Revenue'].sum().max() / fdf['Revenue'].sum() * 100
    top_region = fdf.groupby('Region')['Revenue'].sum().idxmax()
    return_rate = fdf['Returned'].mean() * 100
    
    with col1:
        st.success(f"🏆 **Top Category**: {top_cat} ({top_cat_pct:.1f}% of revenue)")
    with col2:
        st.info(f"🌍 **Leading Region**: {top_region}")
    with col3:
        st.warning(f"📦 **Return Rate**: {return_rate:.2f}%")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    # Chart: Revenue vs Rating Correlation
    with col1:
        st.markdown("##### ⭐ Revenue vs Customer Rating")
        rating_rev = fdf.groupby(fdf['Rating'].round()).agg({'Revenue': 'sum', 'Profit': 'sum'}).reset_index()
        fig = px.scatter(rating_rev, x='Rating', y='Revenue', size='Profit', 
                        color='Revenue', color_continuous_scale='Oranges')
        fig.update_layout(height=350, template='plotly_dark', margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    
    # Chart: Return Rate by Category
    with col2:
        st.markdown("##### 🔄 Return Analysis by Category")
        returns = fdf.groupby('Category').agg({
            'Returned': lambda x: x.mean() * 100,
            'Revenue': 'sum'
        }).reset_index()
        returns.columns = ['Category', 'Return_Rate', 'Revenue']
        fig = px.bar(returns, x='Category', y='Return_Rate', color='Revenue',
                    color_continuous_scale='RdYlGn_r', text='Return_Rate')
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(height=350, template='plotly_dark', margin=dict(t=10))
        fig.update_yaxes(title='Return Rate (%)')
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    # Chart: Segment Comparison Radar
    with col1:
        st.markdown("##### 📊 Segment Performance Comparison")
        seg_metrics = fdf.groupby('Segment').agg({
            'Revenue': 'sum', 'Profit': 'sum', 'Rating': 'mean', 'Quantity': 'sum'
        }).reset_index()
        # Normalize for radar
        for col in ['Revenue', 'Profit', 'Rating', 'Quantity']:
            seg_metrics[col] = (seg_metrics[col] - seg_metrics[col].min()) / (seg_metrics[col].max() - seg_metrics[col].min() + 0.001) * 100
        
        fig = go.Figure()
        categories_radar = ['Revenue', 'Profit', 'Rating', 'Quantity']
        for _, row in seg_metrics.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row['Revenue'], row['Profit'], row['Rating'], row['Quantity']],
                theta=categories_radar, fill='toself', name=row['Segment']
            ))
        fig.update_layout(height=350, template='plotly_dark', margin=dict(t=30), 
                         polar=dict(radialaxis=dict(visible=True, range=[0, 100])))
        st.plotly_chart(fig, use_container_width=True)
    
    # Chart: Top Products Table
    with col2:
        st.markdown("##### 🏅 Top 10 Products Performance")
        top_prods = fdf.groupby('Product').agg({
            'Revenue': 'sum', 'Profit': 'sum', 'Quantity': 'sum', 'Rating': 'mean'
        }).nlargest(10, 'Revenue').round(2)
        top_prods['Revenue'] = top_prods['Revenue'].apply(lambda x: f"${x:,.0f}")
        top_prods['Profit'] = top_prods['Profit'].apply(lambda x: f"${x:,.0f}")
        top_prods['Rating'] = top_prods['Rating'].apply(lambda x: f"⭐ {x:.1f}")
        st.dataframe(top_prods, use_container_width=True, height=350)

st.divider()
st.caption("📊 Amazon Analytics Dashboard | 20 Interactive Visualizations | Data refreshes on filter change")
