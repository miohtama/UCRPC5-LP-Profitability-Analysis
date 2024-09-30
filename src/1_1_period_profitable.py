import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import seaborn as sns
import numpy as np

pool_list = ["usdc-eth-001","usdc-eth-03","usdc-eth-005","wbtc-eth-03"]
def get_time_period_profitable(pool):

    folder_path = f'../../../../data/research/task2403-uni-profitability/{pool}/5_position_fee'

    result_df = pd.DataFrame(columns=['position_id', 'time_period', 'profitability', 'max_net_value'])
    invalid_count = 0
    empty_files = []

    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)

            df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            invalid_rows = df[df.columns[0]].isna()
            invalid_count += invalid_rows.sum()
            df = df[~invalid_rows]

            if df.empty:
                empty_files.append(filename)
                continue

            df = df[df[df.columns[0]].dt.date != pd.to_datetime('2038-01-01').date()]

            time_period = df[df.columns[0]].iloc[-1] - df[df.columns[0]].iloc[0]
            profitability = df['cumprod_return_rate'].iloc[-1]
            position_id = filename.split('.')[0]
            max_net_value = df['total_net_value'].max()
            result_df = result_df._append({
                'position_id': position_id,
                'time_period': time_period,
                'profitability': profitability,
                'max_net_value': max_net_value
            }, ignore_index=True)
    print(f"Total invalid rows: {invalid_count}")
    print(f"Total invalid csv: {empty_files}")
    
    result_df.to_csv(f'output/csv/1_1/position_period_profitable/{pool}_position_time_period_profitable.csv', index=False)

def get_time_period_profitable_lp(pool):

    folder_path = f'../../../../data/research/task2403-uni-profitability/{pool}/6_address_result'

    result_df = pd.DataFrame(columns=['lp_address', 'time_period', 'profitability', 'max_net_value'])
    invalid_count = 0
    empty_files = []

    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            # Check if the file is empty
            if os.path.getsize(file_path) == 0:
                empty_files.append(filename)
                continue

            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
                continue

            df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            invalid_rows = df[df.columns[0]].isna()
            invalid_count += invalid_rows.sum()
            df = df[~invalid_rows]

            if df.empty:
                empty_files.append(filename)
                continue

            df = df[df[df.columns[0]].dt.date != pd.to_datetime('2038-01-01').date()]

            time_period = df[df.columns[0]].iloc[-1] - df[df.columns[0]].iloc[0]
            profitability = df['cumulate_return_rate'].iloc[-1]
            position_id = filename.split('.')[0]
            max_net_value = df['net_value'].max()
            result_df = result_df._append({
                'lp_address': position_id,
                'time_period': time_period,
                'profitability': profitability,
                'max_net_value': max_net_value
            }, ignore_index=True)
    print(f"Total invalid rows: {invalid_count}")
    print(f"Total invalid csv: {empty_files}")
    result_df.to_csv(f'output/csv/1_1/position_period_profitable_lp/{pool}_position_time_period_profitable_lp.csv', index=False)

# for pool in pool_list:
#     print(f"================== Processing pool: {pool} ==================")
#     get_time_period_profitable_lp(pool)


def price_annualized_profit(pool):
    csv_path = f'../../../../data/research/task2403-uni-profitability/{pool}/4_price.csv'
    df = pd.read_csv(csv_path, low_memory=False)
    df['block_timestamp'] = pd.to_datetime(df['block_timestamp'])
    df['date'] = df['block_timestamp'].dt.date
    daily_prices = df.groupby('date')['price'].last().reset_index()
    daily_prices['daily_return'] = daily_prices['price'].pct_change()

    annualized_return = (1 + daily_prices['daily_return']).prod() ** (365 / len(daily_prices))
    annualized_return = daily_prices['daily_return']
    daily_prices['annualized_return'] = annualized_return+1

    initial_date = daily_prices['date'].iloc[0]
    daily_prices['time_period'] = daily_prices['date'].apply(lambda x: x - initial_date)

    output_csv_path = f'output/csv/1_1/price/{pool}_eth_price_daily_returns_with_annualized.csv'
    daily_prices.to_csv(output_csv_path, index=False)

# for pool in pool_list:
#     print(f"================== Processing price return: {pool} ==================")
#     price_annualized_profit(pool)



def plot_time_period_profitable(pool):
    csv_path = f'output/csv/1_1/position_period_profitable/{pool}_position_time_period_profitable.csv'
    price_path = f'output/csv/1_1/price/{pool}_eth_price_daily_returns_with_annualized.csv'
    
    df = pd.read_csv(csv_path)
    df_price = pd.read_csv(price_path)
    
    initial_rows = len(df)
    df = df[df['max_net_value'] != 0.0]
    removed_rows = initial_rows - len(df)
    print(f"Removed {removed_rows} rows with max_net_value = 0.0 from {csv_path}")
    
    df['time_period'] = pd.to_timedelta(df['time_period'])
    df['time_period_days'] = df['time_period'].dt.days
    df['time_period_months'] = (df['time_period_days'] // 30) + 1
    
    df_price['time_period'] = pd.to_timedelta(df_price['time_period'])
    df_price['time_period_days'] = df_price['time_period'].dt.days

    plt.figure(figsize=(12, 8))
    
    # Create the primary axis for profitability
    ax1 = plt.gca()
    sns.boxplot(x='time_period_days', y='profitability', data=df, hue='time_period_days', palette='Set3', fliersize=2, legend=False, ax=ax1)
    mean_values = df.groupby('time_period_days')['profitability'].mean().reset_index()
    ax1.plot(mean_values['time_period_days'], mean_values['profitability'], linestyle='-', color = 'dodgerblue',label='Mean Profitability of Different Positions')
    ax1.set_xlabel('Position Holding Time Period (days)')
    ax1.set_ylabel('Profitability', color='dodgerblue')
    ax1.tick_params(axis='y', labelcolor='dodgerblue')
    
    # Create the secondary axis for price
    ax2 = ax1.twinx()
    ax2.plot(df_price['time_period_days'], df_price['price'], linestyle='-', color='aquamarine', label='Price of ETH')
    ax2.set_ylabel('Price of ETH', color='aquamarine')
    ax2.tick_params(axis='y', labelcolor='aquamarine')
    
    plt.title(f'Profitability of Positions in pool {pool} with Different Position Time')
    ax1.set_ylim(0, 2)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    max_days = df['time_period_days'].max()
    plt.xticks(range(0, max_days + 1, 30))
    
    output_path = f'output/img/1_1/boxplot_price/{pool}_boxplot_position_time_period_profitable.png'
    plt.savefig(output_path)
    plt.close()

for pool in pool_list:
    print(f"================== Processing img: {pool} ==================")
    plot_time_period_profitable(pool)

def plot_time_period_profitable_lp(pool):
    csv_path = f'output/csv/1_1/position_period_profitable_lp/{pool}_position_time_period_profitable_lp.csv'
    price_path = f'output/csv/1_1/price/{pool}_eth_price_daily_returns_with_annualized.csv'
    
    df = pd.read_csv(csv_path)
    df_price = pd.read_csv(price_path)
    
    initial_rows = len(df)
    df = df[df['max_net_value'] != 0.0]
    removed_rows = initial_rows - len(df)
    print(f"Removed {removed_rows} rows with max_net_value = 0.0 from {csv_path}")
    
    df['time_period'] = pd.to_timedelta(df['time_period'])
    df['time_period_days'] = df['time_period'].dt.days
    df['time_period_months'] = (df['time_period_days'] // 30) + 1
    
    df_price['time_period'] = pd.to_timedelta(df_price['time_period'])
    df_price['time_period_days'] = df_price['time_period'].dt.days
    df_price = df_price.sort_values(by='time_period_days')
    
    plt.figure(figsize=(12, 8))
    
    # Create the primary axis for profitability
    ax1 = plt.gca()
    sns.boxplot(x='time_period_days', y='profitability', data=df, hue='time_period_days', palette='Set3', fliersize=2, legend=False, ax=ax1)
    mean_values = df.groupby('time_period_days')['profitability'].mean().reset_index()
    ax1.plot(mean_values['time_period_days'], mean_values['profitability'], linestyle='-', color='dodgerblue', label='Mean Profitability of Different Positions')
    ax1.set_xlabel('Total Position Holding Time Period (days) of a LP')
    ax1.set_ylabel('Profitability', color='dodgerblue')
    ax1.tick_params(axis='y', labelcolor='dodgerblue')
    
    # Create the secondary axis for price
    ax2 = ax1.twinx()
    ax2.plot(df_price['time_period_days'], df_price['price'], linestyle='-', color='aquamarine', label='Price')
    ax2.set_ylabel('Price of ETH', color='aquamarine')
    ax2.tick_params(axis='y', labelcolor='aquamarine')
    
    plt.title(f'Profitability of being an LP in pool {pool} with Different Total Position Holding Time')
    ax1.set_ylim(0, 2)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    max_days = df['time_period_days'].max()
    plt.xticks(range(0, max_days + 1, 30))
    
    output_path = f'output/img/1_1/boxplot_price_lp/{pool}_boxplot_position_time_period_profitable_lp.png'
    plt.savefig(output_path)
    plt.close()

for pool in pool_list:
    print(f"================== Processing img: {pool} ==================")
    plot_time_period_profitable_lp(pool)
