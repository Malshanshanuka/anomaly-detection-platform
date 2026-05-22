import urllib.request
import pandas as pd
import os

def fetch_nab_data():
    """
    Fetches real-world AWS EC2 CPU utilization data from the 
    Numenta Anomaly Benchmark (NAB) repository.
    """
    # Official URL for the NAB dataset (Real AWS Cloudwatch data)
    url = "https://raw.githubusercontent.com/numenta/NAB/master/data/realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv"
    output_path = "data/real_cpu_data.csv"

    print("Fetching real-world AWS server data...")

    # Ensure the 'data' directory exists
    os.makedirs("data", exist_ok=True)

    # Download the dataset and save it as a CSV file
    urllib.request.urlretrieve(url, output_path)

    # Load and verify the dataset
    df = pd.read_csv(output_path)
    
    print("\nData fetched successfully!")
    print(f"Total records: {len(df)}")
    print("\nFirst 5 rows of the dataset:")
    print(df.head())

if __name__ == "__main__":
    fetch_nab_data()