"""
Download AG News dataset directly (no Kaggle needed)
"""
import pandas as pd
import urllib.request
import os

def download_ag_news():
    """Download AG News dataset from alternative source"""
    print("Downloading AG News dataset...")
    
    # Alternative public URLs for AG News
    urls = {
        'train': 'https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv',
        'test': 'https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv'
    }
    
    os.makedirs('data', exist_ok=True)
    
    for name, url in urls.items():
        filepath = f'data/ag_news_{name}.csv'
        if os.path.exists(filepath):
            print(f"  ✓ {filepath} already exists")
            continue
        
        try:
            print(f"  → Downloading {name} set from {url}")
            urllib.request.urlretrieve(url, filepath)
            print(f"  ✓ Downloaded to {filepath}")
        except Exception as e:
            print(f"  ✗ Failed to download {name}: {e}")
    
    return 'data/ag_news_train.csv', 'data/ag_news_test.csv'

if __name__ == "__main__":
    train_path, test_path = download_ag_news()
    
    # Verify
    try:
        df = pd.read_csv(train_path, header=None)
        print(f"\n✓ Successfully loaded {len(df)} training samples")
        print(f"  Classes: {df[0].value_counts().to_dict()}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
