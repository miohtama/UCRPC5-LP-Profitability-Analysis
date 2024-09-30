import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Step 1: Function to read the necessary CSV files, remove "15/03/2024 00:00" entries, and filter out rows with block_number == 0
def read_csv_files(pool):
    liquidity_csv_path = f'../../../../data/research/task2403-uni-profitability/{pool}/2_position_liquidity.csv'
    address_csv_path = f'../../../../data/research/task2403-uni-profitability/{pool}/3_position_address.csv'
    
    liquidity_df = pd.read_csv(liquidity_csv_path)
    address_df = pd.read_csv(address_csv_path)

    # Convert 'blk_time' to datetime and filter out entries with "15/03/2024 00:00"
    liquidity_df['blk_time'] = pd.to_datetime(liquidity_df['blk_time'], format='%Y-%m-%d %H:%M:%S')
    liquidity_df = liquidity_df[liquidity_df['blk_time'] != pd.to_datetime("2024-03-15 00:00:00")]

    # Additional filter: Remove rows where block_number is 0
    liquidity_df = liquidity_df[liquidity_df['block_number'] != 0]

    return liquidity_df, address_df

# Step 2: Function to merge the liquidity and address DataFrames
def merge_liquidity_and_address(liquidity_df, address_df):
    merged_df = pd.merge(liquidity_df, address_df, how='inner', left_on='id', right_on='position')
    return merged_df

# Step 3: Function to filter the merged DataFrame based on existing position.csv files in the fee directory
def filter_positions_by_fee_files(merged_df, fee_dir):
    position_files = [f.replace('.csv', '') for f in os.listdir(fee_dir) if f.endswith('.csv')]
    filtered_df = merged_df[merged_df['position'].isin(position_files)]
    return filtered_df

# Step 4: Function to calculate the cumulative return (cum_ret) from position.csv files
def calculate_cum_ret(filtered_df, fee_dir):
    cum_ret_values = []
    for position in filtered_df['position']:
        position_csv_path = os.path.join(fee_dir, f'{position}.csv')
        position_df = pd.read_csv(position_csv_path)
        cum_ret = position_df['cumprod_return_rate'].iloc[-1] / position_df['cumprod_return_rate'].iloc[0]
        cum_ret_values.append(cum_ret)
    
    filtered_df['cum_ret'] = cum_ret_values
    return filtered_df

# Step 5: Function to count active days for each user based on mint and burn events
def calculate_active_days(df):
    user_active_days = {}
    cutoff_date = pd.to_datetime("2024-09-15")

    for address, user_df in tqdm(df.groupby('address')):
        user_df = user_df.sort_values(by='blk_time')
        active_days = set()

        for position, position_df in user_df.groupby('position'):
            mint_dates = position_df[position_df['tx_type'] == 'MINT']['blk_time']
            burn_dates = position_df[position_df['tx_type'] == 'BURN']['blk_time']
            
            for mint_date in mint_dates:
                burn_date = burn_dates[burn_dates > mint_date].min() if not burn_dates.empty else cutoff_date
                if pd.isna(burn_date):
                    burn_date = cutoff_date
                
                active_days.update(pd.date_range(mint_date, burn_date).date)

        user_active_days[address] = len(active_days)

    active_days_df = pd.DataFrame(list(user_active_days.items()), columns=['address', 'active_days'])
    return active_days_df

# Step 6: Function to add active days and percentage to the DataFrame
def add_active_days(df, active_days_df):
    df = pd.merge(df, active_days_df, on='address', how='left')
    total_days = 184  # The total number of days from 15 March 2024 to 15 September 2024
    df['active_days_percentage'] = df['active_days'] / total_days
    return df

# Step 7: **Updated Function** to calculate cum_ret_prod for each address
# This version collapses duplicate rows based on 'id' and then calculates cum_ret_prod per user
def calculate_cum_ret_prod(df):
    # Step 7.1: Remove duplicates based on 'id' to ensure one row per unique position
    df_unique_positions = df.drop_duplicates(subset=['id'])

    # Step 7.2: Group by 'address' and calculate the product of 'cum_ret' for all positions of that user
    cum_ret_prod_per_user = df_unique_positions.groupby('address')['cum_ret'].prod().reset_index()

    # Step 7.3: Rename the column for clarity
    cum_ret_prod_per_user.rename(columns={'cum_ret': 'cum_ret_prod'}, inplace=True)

    # Step 7.4: Merge the result back into the original DataFrame
    df = pd.merge(df, cum_ret_prod_per_user, on='address', how='left')

    return df

# Save the DataFrame to a CSV file after step 7
def save_df_to_csv(df, pool):
    output_csv_path = f'./output/{pool}_cum_ret_prod.csv'
    df.to_csv(output_csv_path, index=False)
    print(f"Saved DataFrame with cum_ret_prod to {output_csv_path}")

# Step 8: Function to filter and display top and bottom performers based on cum_ret_prod and active days criteria
# and save the output to a text file instead of printing to the terminal
def display_top_and_bottom_performers_to_txt(df, pool):
    performers = df.groupby('address').agg({
        'cum_ret_prod': 'first',  # Get the first value of cum_ret_prod
        'id': 'nunique',  # Count the unique positions (id) per address
        'active_days': 'first',  # Get active days per address
        'active_days_percentage': 'first'  # Get percentage of active days
    })
    
    performers = performers[(performers['id'] >= 5) & (performers['active_days_percentage'] >= 0.4)]

    top_performers = performers.sort_values(by='cum_ret_prod', ascending=False).head(10)
    bottom_performers = performers.sort_values(by='cum_ret_prod', ascending=True).head(10)
    
    output_txt_path = f'./output/{pool}_top_bottom_performers.txt'
    
    with open(output_txt_path, 'w') as f:
        f.write("Top 10 Performing Users (with >50% active days):\n")
        f.write(top_performers.rename(columns={'id': 'num_positions'}).to_string())
        f.write("\n\nBottom 10 Performing Users (with >50% active days):\n")
        f.write(bottom_performers.rename(columns={'id': 'num_positions'}).to_string())

    print(f"Saved top and bottom performers to {output_txt_path}")

# Step 9: Function to scatter plot upper_tick and lower_tick separately for top and bottom 5 users
def scatter_plot_performers_over_time(df, pool):
    # Convert 'blk_time' to datetime and extract date
    df['date'] = pd.to_datetime(df['blk_time'], format='%Y-%m-%d %H:%M:%S').dt.date

    # Get the top 5 and bottom 5 performing users based on 'cum_ret_prod'
    top_5_performers = df[['address', 'cum_ret_prod']].drop_duplicates().sort_values(by='cum_ret_prod', ascending=False).head(5)
    bottom_5_performers = df[['address', 'cum_ret_prod']].drop_duplicates().sort_values(by='cum_ret_prod', ascending=True).head(5)

    # Define the date range for the x-axis
    date_min = pd.to_datetime("2024-03-15").date()
    date_max = pd.to_datetime("2024-09-15").date()

    # Create directories for saving the plots
    pool_output_dir = f'/output/img/2_2/merged_cum_ret_prod/{pool}/'
    top_output_dir = f'{pool_output_dir}top_performers/'
    bottom_output_dir = f'{pool_output_dir}bottom_performers/'
    os.makedirs(top_output_dir, exist_ok=True)
    os.makedirs(bottom_output_dir, exist_ok=True)

    # Scatter plot for top performers (upper_tick and lower_tick are separated into different plots)
    for i, user in enumerate(top_5_performers['address']):
        user_df = df[df['address'] == user]

        # Scatter plot 'upper_tick' over time with alpha transparency
        plt.figure(figsize=(10, 6))
        plt.scatter(user_df['date'], user_df['upper_tick'], color='green', alpha=0.5, label='Upper Tick')
        plt.title(f'Upper Tick Over Time - Pool: {pool} - User: {user}')
        plt.xlabel('Date')
        plt.ylabel('Upper Tick')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.xlim(date_min, date_max)
        plt.tight_layout()
        plt.savefig(f'{top_output_dir}{pool}_user_{i+1}_upper_tick_scatter.png')
        plt.close()

        # Scatter plot 'lower_tick' over time with alpha transparency
        plt.figure(figsize=(10, 6))
        plt.scatter(user_df['date'], user_df['lower_tick'], color='red', alpha=0.5, label='Lower Tick')
        plt.title(f'Lower Tick Over Time - Pool: {pool} - User: {user}')
        plt.xlabel('Date')
        plt.ylabel('Lower Tick')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.xlim(date_min, date_max)
        plt.tight_layout()
        plt.savefig(f'{top_output_dir}{pool}_user_{i+1}_lower_tick_scatter.png')
        plt.close()

    # Scatter plot for bottom performers (upper_tick and lower_tick are separated into different plots)
    for i, user in enumerate(bottom_5_performers['address']):
        user_df = df[df['address'] == user]

        # Scatter plot 'upper_tick' over time with alpha transparency
        plt.figure(figsize=(10, 6))
        plt.scatter(user_df['date'], user_df['upper_tick'], color='green', alpha=0.5, label='Upper Tick')
        plt.title(f'Upper Tick Over Time - Pool: {pool} - User: {user}')
        plt.xlabel('Date')
        plt.ylabel('Upper Tick')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.xlim(date_min, date_max)
        plt.tight_layout()
        plt.savefig(f'{bottom_output_dir}{pool}_user_{i+1}_upper_tick_scatter.png')
        plt.close()

        # Scatter plot 'lower_tick' over time with alpha transparency
        plt.figure(figsize=(10, 6))
        plt.scatter(user_df['date'], user_df['lower_tick'], color='red', alpha=0.5, label='Lower Tick')
        plt.title(f'Lower Tick Over Time - Pool: {pool} - User: {user}')
        plt.xlabel('Date')
        plt.ylabel('Lower Tick')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.xlim(date_min, date_max)
        plt.tight_layout()
        plt.savefig(f'{bottom_output_dir}{pool}_user_{i+1}_lower_tick_scatter.png')
        plt.close()

    print(f"Saved scatter plots for pool {pool} in {pool_output_dir}")

# Main function to process all pools
def process_pools_and_plot(pools):
    for pool in tqdm(pools, desc="Processing pools"):
        print(f"Processing pool {pool}...")

        fee_dir_base = f'../../../../data/research/task2403-uni-profitability/{pool}/5_position_fee/'

        liquidity_df, address_df = read_csv_files(pool)
        merged_df = merge_liquidity_and_address(liquidity_df, address_df)
        filtered_df = filter_positions_by_fee_files(merged_df, fee_dir_base)
        filtered_df_with_ret = calculate_cum_ret(filtered_df, fee_dir_base)

        # Calculate active days for each user
        active_days_df = calculate_active_days(filtered_df)
        filtered_df_with_active_days = add_active_days(filtered_df_with_ret, active_days_df)

        # Calculate cumulative return for each user (updated version)
        df_with_cum_ret_prod = calculate_cum_ret_prod(filtered_df_with_active_days)

        # Save the DataFrame to a CSV file after step 7
        save_df_to_csv(df_with_cum_ret_prod, pool)

        # Display top 10 and bottom 10 performing users with >50% active days, and save to txt
        display_top_and_bottom_performers_to_txt(df_with_cum_ret_prod, pool)

# List of pools to process
pools = ['usdc-eth-001', 'usdc-eth-03', 'wbtc-eth-03', 'usdc-eth-005']

# Process pools to find top and bottom performers, filter, and create scatter plots
process_pools_and_plot(pools)
