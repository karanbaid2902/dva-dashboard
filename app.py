import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

end_date = datetime.now()
start_date = end_date - timedelta(days=365)
dates = pd.date_range(start=start_date, end=end_date, freq='D')

categories = ['Electronics', 'Books & Media', 'Home & Kitchen', 'Fashion', 'Sports & Outdoors', 'Beauty & Personal Care']

# Real Amazon products with actual average prices
products = {
    'Electronics': {
        'Fire TV Stick': 39.99,
        'Echo Dot': 49.99,
        'Kindle': 99.99,
        'Ring Video Doorbell': 99.99,
        'Echo Show': 149.99
    },
    'Books & Media': {
        'Science Fiction Novel': 14.99,
        'Business Book': 19.99,
        'Self-Help Guide': 16.99,
        'Comic Book': 12.99,
        'Audiobook': 17.99
    },
    'Home & Kitchen': {
        'Air Purifier': 89.99,
        'Coffee Maker': 49.99,
        'Bedding Set': 79.99,
        'Cookware': 99.99,
        'Smart Bulbs': 39.99
    },
    'Fashion': {
        'Running Shoes': 79.99,
        'Denim Jeans': 49.99,
        'Cotton T-Shirt': 19.99,
        'Winter Jacket': 89.99,
        'Backpack': 59.99
    },
    'Sports & Outdoors': {
        'Yoga Mat': 29.99,
        'Dumbbells': 24.99,
        'Tent': 129.99,
        'Fitness Tracker': 149.99,
        'Camping Gear': 89.99
    },
    'Beauty & Personal Care': {
        'Face Cream': 24.99,
        'Shampoo': 12.99,
        'Perfume': 49.99,
        'Lipstick': 9.99,
        'Moisturizer': 19.99
    }
}

regions = ['US East', 'US West', 'Europe', 'Asia-Pacific', 'Latin America']
region_weights = [0.30, 0.20, 0.25, 0.18, 0.07]

payment_methods = ['Amazon Wallet', 'Credit/Debit Card', 'Amazon Pay', 'COD']
customer_segments = ['Prime Member', 'Regular', 'New']

num_transactions = 100000
data = []

print("Generating data with random timestamps... this may take a minute...")

for i in range(num_transactions):
    # Generate random date
    date = pd.Timestamp(np.random.choice(dates))
    
    # Add random time (hour, minute, second)
    hour = np.random.randint(0, 24)
    minute = np.random.randint(0, 60)
    second = np.random.randint(0, 60)
    
    # Combine date and time
    datetime_obj = date.replace(hour=hour, minute=minute, second=second)
    
    category_weights = [0.28, 0.12, 0.18, 0.20, 0.12, 0.10]
    category = np.random.choice(categories, p=category_weights)
    product = np.random.choice(list(products[category].keys()))
    unit_price = products[category][product]
    unit_price = unit_price * np.random.uniform(0.95, 1.05)
    
    region = np.random.choice(regions, p=region_weights)
    payment = np.random.choice(payment_methods)
    customer_segment = np.random.choice(customer_segments, p=[0.45, 0.40, 0.15])
    
    profit_margins_by_category = {
        'Electronics': (0.35, 0.50),
        'Books & Media': (0.40, 0.55),
        'Home & Kitchen': (0.38, 0.52),
        'Fashion': (0.42, 0.58),
        'Sports & Outdoors': (0.40, 0.55),
        'Beauty & Personal Care': (0.38, 0.52)
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
        'DateTime': datetime_obj,  # Combined date and time
        'Date': date.date(),  # Just the date
        'Time': f"{hour:02d}:{minute:02d}:{second:02d}",  # Time in HH:MM:SS format
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
    
    # Progress indicator
    if (i + 1) % 10000 == 0:
        print(f"Generated {i + 1:,} records...")

df = pd.DataFrame(data)
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['Date'] = pd.to_datetime(df['Date'])

# Save to CSV
filename = f"amazon_sales_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
df.to_csv(filename, index=False)

print(f"\n✅ CSV file created successfully: {filename}")
print(f"📊 Total records: {len(df):,}")
print(f"📅 Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"⏰ Time range: 00:00:00 to 23:59:59")
print(f"💾 File size: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
print(f"\n📋 Sample of first 5 rows:")
print(df[['DateTime', 'Date', 'Time', 'Order_ID', 'Product', 'Total_Amount']].head())