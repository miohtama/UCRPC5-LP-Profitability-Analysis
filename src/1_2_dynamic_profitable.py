import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import seaborn as sns
import numpy as np

pool_list = ["usdc-eth-001","usdc-eth-03","usdc-eth-005","wbtc-eth-03"]
def get_time_profitable_lp(pool):

    folder_path = f'../../../../data/research/task2403-uni-profitability/{pool}/6_address_result'

    all_data = []
    invalid_count = 0
    empty_files = []
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            try:
                df = pd.read_csv(file_path, engine='python')
            except pd.errors.EmptyDataError:
                print(f"Skipping empty file: {file_path}")
                continue
            except pd.errors.ParserError as e:
                print(f"Error reading file {file_path}: {e}")
                continue

            df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            
            invalid_rows = df[df.columns[0]].isna()
            invalid_count += invalid_rows.sum()
            df = df[~invalid_rows]

            if df.empty:
                empty_files.append(filename)
                continue
            df['date'] = df[df.columns[0]].dt.date
            daily_data = df.groupby('date').apply(lambda x: x.iloc[-1])[['date', 'cumulate_return_rate', 'net_value']]
            daily_data['address'] = filename.split('.')[0]
            all_data.append(daily_data)
    all_data_df = pd.concat(all_data)
    all_data_df = all_data_df.reset_index(drop=True)
    weighted_mean_returns = all_data_df.groupby('date').apply(lambda x: (x['cumulate_return_rate'] * x['net_value']).sum() / x['net_value'].sum() if x['net_value'].sum() != 0 else 1)
    
    print(f"Total invalid rows: {invalid_count}")
    print(f"Total invalid csv: {empty_files}")

    result_df = pd.DataFrame(weighted_mean_returns, columns=['weighted_mean_return'])
    result_df.index.name = 'date'
    result_df = result_df.reset_index()

    output_file_path = f'output/csv/1_2/lp_profit/{pool}_lp_profit.csv'
    result_df.to_csv(output_file_path, index=False)
    
for pool in pool_list:
    print(f"================== Processing pool: {pool} ==================")
    get_time_profitable_lp(pool)

# fee file: f"../../../../data/research/task2403-uni-profitability/{pool}/fee.csv"

def plot_lp_profit_and_fee(pool):
    lp_profit_file_path = f'output/csv/1_2/lp_profit/{pool}_lp_profit.csv'
    lp_profit_df = pd.read_csv(lp_profit_file_path)
    lp_profit_df['date'] = pd.to_datetime(lp_profit_df['date'])
    lp_profit_df = lp_profit_df[lp_profit_df['date'] != '2038-01-01']

    fee_file_path = f"../../../../data/research/task2403-uni-profitability/{pool}/fee.csv"
    fee_df = pd.read_csv(fee_file_path)
    fee_df['date'] = pd.to_datetime(fee_df['date'])
    
    # 计算累积收益率
    fee_df['cumulative_return'] = (1 + fee_df['c']).cumprod()

    csv_path = f'../../../../data/research/task2403-uni-profitability/{pool}/4_price.csv'
    df_price = pd.read_csv(csv_path, low_memory=False)
    df_price['block_timestamp'] = pd.to_datetime(df_price['block_timestamp'])
    df_price['date'] = df_price['block_timestamp'].dt.date
    daily_prices = df_price.groupby('date')['price'].last().reset_index()
    daily_prices = daily_prices.reset_index(drop=True)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative LP Profit', color=color)
    ax1.plot(lp_profit_df['date'], lp_profit_df['weighted_mean_return'], color=color, label='Cumulative LP Profit', linestyle='-')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Cumulative Fee Return', color=color)
    ax2.plot(fee_df['date'], fee_df['cumulative_return'], color=color, label='Cumulative Fee Return', linestyle='-')
    ax2.tick_params(axis='y', labelcolor=color)

    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))  # 将第三个y轴移到右侧
    color = 'tab:green'
    ax3.set_ylabel('ETH Price', color=color)
    ax3.plot(daily_prices['date'], daily_prices['price'], color=color, label='Price Of ETH', linestyle='-')
    ax3.tick_params(axis='y', labelcolor=color)

    plt.title(f'LP Profitable and Fee Earning for {pool}')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax3.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc='upper left')
    plt.savefig(f'output/img/1_2/lp_profit_C/{pool}_lp_profit_C.png')
    plt.close()

for pool in pool_list:
    print(f"================== Processing pool: {pool} ==================")
    plot_lp_profit_and_fee(pool)
