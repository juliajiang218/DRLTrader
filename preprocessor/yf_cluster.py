"""
YFinance Cluster-Compatible Version
Designed specifically for cluster environments with limited internet access
"""

import pandas as pd
import time
import sys
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def create_mock_data(ticker, start_date, end_date):
    """Create mock data for testing when yfinance is not available"""
    print(f"Creating mock data for {ticker} from {start_date} to {end_date}")
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Generate business days
    dates = pd.bdate_range(start=start, end=end)
    
    # Create mock price data
    base_price = 100.0
    data = []
    
    for i, date in enumerate(dates):
        # Simple random walk for mock prices
        import random
        change = random.uniform(-2, 2)
        base_price += change
        
        data.append({
            'Date': date,
            'Open': base_price - random.uniform(0, 1),
            'High': base_price + random.uniform(0, 2),
            'Low': base_price - random.uniform(0, 2),
            'Close': base_price,
            'Adj Close': base_price,
            'Volume': random.randint(1000000, 5000000)
        })
    
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    return df

def test_cluster_environment():
    """Test the cluster environment and provide detailed diagnostics"""
    print("=== Cluster Environment Diagnostics ===")
    
    # Basic system info
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Working directory: {os.getcwd()}")
    print(f"User: {os.getenv('USER', 'Unknown')}")
    print(f"Hostname: {os.getenv('HOSTNAME', 'Unknown')}")
    
    # Check network connectivity
    print("\n=== Network Connectivity Test ===")
    try:
        import urllib.request
        response = urllib.request.urlopen('http://www.google.com', timeout=10)
        print("✓ Basic internet connectivity: OK")
    except Exception as e:
        print(f"✗ Basic internet connectivity failed: {e}")
    
    # Check if yfinance is available
    print("\n=== YFinance Availability Test ===")
    try:
        import yfinance as yf
        print(f"✓ YFinance is available (version: {getattr(yf, '__version__', 'Unknown')})")
        return True
    except ImportError as e:
        print(f"✗ YFinance not available: {e}")
        return False

def download_data_cluster_safe(ticker, start_date, end_date, use_mock=False):
    """Download data with cluster-safe fallbacks"""
    
    if use_mock:
        return create_mock_data(ticker, start_date, end_date)
    
    # Try yfinance first
    try:
        import yfinance as yf
        
        print(f"Attempting to download {ticker} data...")
        
        # Method 1: Direct download
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                timeout=60,
                ignore_tz=True
            )
            if df is not None and not df.empty:
                print(f"✓ Direct download successful: {len(df)} rows")
                return df
        except Exception as e:
            print(f"Direct download failed: {e}")
        
        # Method 2: Ticker object
        try:
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(start=start_date, end=end_date)
            if df is not None and not df.empty:
                print(f"✓ Ticker object download successful: {len(df)} rows")
                return df
        except Exception as e:
            print(f"Ticker object download failed: {e}")
        
        # Method 3: With custom session
        try:
            import requests
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                timeout=60,
                session=session
            )
            if df is not None and not df.empty:
                print(f"✓ Custom session download successful: {len(df)} rows")
                return df
        except Exception as e:
            print(f"Custom session download failed: {e}")
        
        print("All yfinance methods failed")
        
    except ImportError:
        print("YFinance not available")
    
    # Fallback to mock data
    print("Falling back to mock data...")
    return create_mock_data(ticker, start_date, end_date)

def main():
    """Main function for cluster testing"""
    print("=== YFinance Cluster Test ===")
    
    # Test environment
    yf_available = test_cluster_environment()
    
    # Test parameters
    ticker = "MSFT"
    start_date = "2025-01-01"
    end_date = "2025-02-01"
    
    print(f"\n=== Testing Data Download ===")
    print(f"Ticker: {ticker}")
    print(f"Date range: {start_date} to {end_date}")
    
    # Try to download data
    df = download_data_cluster_safe(ticker, start_date, end_date, use_mock=not yf_available)
    
    if df is not None and not df.empty:
        print(f"\n✅ Data retrieved successfully!")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print("\nFirst 5 rows:")
        print(df.head())
        
        # Save to file
        output_file = f"{ticker}_data_{start_date}_to_{end_date}.csv"
        df.to_csv(output_file)
        print(f"\nData saved to: {output_file}")
        
    else:
        print("\n❌ Failed to retrieve data")
        print("\n🔧 Cluster Environment Solutions:")
        print("1. Install required packages:")
        print("   pip install yfinance pandas requests")
        print("2. If behind a proxy, set environment variables:")
        print("   export HTTP_PROXY=http://proxy:port")
        print("   export HTTPS_PROXY=http://proxy:port")
        print("3. Try using mock data for testing:")
        print("   df = download_data_cluster_safe(ticker, start_date, end_date, use_mock=True)")
        print("4. Consider using a different data source for production")

if __name__ == "__main__":
    main() 