import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import seaborn as sns
import numpy as np
pool_list = ["usdc-eth-001","usdc-eth-03","usdc-eth-005","wbtc-eth-03"]
def plot_lp_profit_and_fee(pool):
    lp_profit_file_path = f'output/csv/1_2/lp_profit/{pool}_lp_profit.csv'
    lp_profit_df = pd.read_csv(lp_profit_file_path)
    lp_profit_df['date'] = pd.to_datetime(lp_profit_df['date'])
    lp_profit_df = lp_profit_df[lp_profit_df['date'] != '2038-01-01']

    fee_file_path = f"../../../../data/research/task2403-uni-profitability/{pool}/fee.csv"
    fee_df = pd.read_csv(fee_file_path)
    fee_df['date'] = pd.to_datetime(fee_df['date'])

    fee_df['cumulative_return'] = (1 + fee_df['c']).cumprod()

    csv_path = f'../../../../data/research/task2403-uni-profitability/{pool}/4_price.csv'
    df_price = pd.read_csv(csv_path, low_memory=False)
    df_price['block_timestamp'] = pd.to_datetime(df_price['block_timestamp'])
    df_price['date'] = df_price['block_timestamp'].dt.date
    daily_prices = df_price.groupby('date')['price'].last().reset_index()
    daily_prices = daily_prices.reset_index(drop=True)

    path = f'../../../../data/research/task2403-uni-profitability/{pool}/2_position_liquidity.csv'
    pos_df = pd.read_csv(path)
    pos_df = pos_df[pos_df['block_number'] != 0]
    pos_df['date'] = pd.to_datetime(pos_df['blk_time'], format='%Y-%m-%d %H:%M:%S').dt.date
    mint_counts = pos_df[pos_df['tx_type'] == 'MINT'].groupby('date').size()
    burn_counts = pos_df[pos_df['tx_type'] == 'BURN'].groupby('date').size()
    date_range = pd.date_range(start=pos_df['date'].min(), end=pos_df['date'].max())
    mint_counts = mint_counts.reindex(date_range, fill_value=0)
    burn_counts = burn_counts.reindex(date_range, fill_value=0)

    fig, ax1 = plt.subplots(figsize=(15, 6))
    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative LP Profit', color=color)
    ax1.plot(lp_profit_df['date'], lp_profit_df['weighted_mean_return'], color=color, label='Cumulative LP Profit', linestyle='-', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Cumulative Fee Return', color=color)
    ax2.plot(fee_df['date'], fee_df['cumulative_return'], color=color, label='Cumulative Fee Return', linestyle='-')
    ax2.tick_params(axis='y', labelcolor=color)

    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 50))  # 将第三个y轴移到右侧
    color = 'tab:green'
    ax3.set_ylabel('ETH Price', color=color)
    ax3.plot(daily_prices['date'], daily_prices['price'], color=color, label='Price Of ETH', linestyle='-')
    ax3.tick_params(axis='y', labelcolor=color)

    ax4 = ax1.twinx()
    ax4.spines['left'].set_position(('outward', 60))  # 将第四个y轴移到左侧
    ax4.spines['left'].set_visible(True)
    ax4.yaxis.set_label_position('left')
    ax4.yaxis.set_ticks_position('left')
    color = 'tab:purple'
    ax4.set_ylabel('Frequency of MINT and BURN', color=color)
    ax4.plot(date_range, mint_counts, label='MINT Frequency', color='hotpink', linestyle='-',alpha = 0.8)
    ax4.plot(date_range, burn_counts, label='BURN Frequency', color=color, linestyle='-',alpha = 0.8)
    ax4.tick_params(axis='y', labelcolor=color)

    plt.title(f'LP Profitable and Fee Earning for {pool}')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()
    ax4.legend(lines + lines2 + lines3 + lines4, labels + labels2 + labels3 + labels4, loc='upper left')
    plt.savefig(f'output/img/1_3/{pool}_position_behaviour.png')
    plt.close()

for pool in pool_list:
    print(f"================== Processing pool: {pool} ==================")
    plot_lp_profit_and_fee(pool)