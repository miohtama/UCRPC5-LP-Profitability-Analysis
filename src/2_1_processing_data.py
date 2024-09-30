import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr
import numpy as np

# List of pools
pools = ['usdc-eth-001', 'usdc-eth-005', 'usdc-eth-03', 'wbtc-eth-03']

# Output directories
csv_output_dir = '/home/zelos/src/UCRPC5/output/csv/2_1/merged_data/'
img_output_dir = '/home/zelos/src/UCRPC5/output/img/2_1/comprehensive_plot/'
corr_output_dir = '/home/zelos/src/UCRPC5/output/csv/2_1/correlation/'

# Ensure the output directories exist
os.makedirs(csv_output_dir, exist_ok=True)
os.makedirs(img_output_dir, exist_ok=True)
os.makedirs(corr_output_dir, exist_ok=True)

# Function to calculate both correlation and p-values
def calculate_corr_and_pvalues(dataframe, columns):
    corr_matrix = dataframe[columns].corr()  # Correlation matrix
    p_value_matrix = pd.DataFrame(np.ones_like(corr_matrix), columns=columns, index=columns)  # P-values matrix
    
    # Calculate correlation coefficients and p-values for each pair of columns
    for col1 in columns:
        for col2 in columns:
            if col1 != col2:
                corr, p_value = pearsonr(dataframe[col1], dataframe[col2])
                corr_matrix.loc[col1, col2] = corr
                p_value_matrix.loc[col1, col2] = p_value
            else:
                p_value_matrix.loc[col1, col2] = np.nan  # P-value of 1 for the diagonal
    return corr_matrix, p_value_matrix

# Loop over each pool
for pool in pools:
    print(f"Processing pool: {pool}")
    
    # Construct file paths based on the pool name
    first_csv_path = f'../../../../data/research/task2403-uni-profitability/{pool}/2_position_liquidity.csv'
    second_csv_path = f'../../../../data/research/task2403-uni-profitability/{pool}/c-lvr.csv'
    sigma_csv_path = f'../../../../data/research/task2403-uni-profitability/{pool}/sigma.csv'
    
    # Read the CSV files
    first_csv = pd.read_csv(first_csv_path)
    second_csv = pd.read_csv(second_csv_path)
    sigma_csv = pd.read_csv(sigma_csv_path)

    # Remove rows where 'block_number' is 0
    first_csv = first_csv[first_csv['block_number'] != 0]

    # Convert the 'blk_time' and 'date' columns to datetime format and extract only the date part
    first_csv['date'] = pd.to_datetime(first_csv['blk_time'], format='%Y-%m-%d %H:%M:%S').dt.date
    second_csv['date'] = pd.to_datetime(second_csv['date']).dt.date
    sigma_csv['date'] = pd.to_datetime(sigma_csv['time']).dt.date

    # Merge the two DataFrames on the 'date' column (many-to-one)
    merged_df = pd.merge(first_csv, second_csv, on='date', how='left')
    merged_df = pd.merge(merged_df, sigma_csv[['date', 'sigma_ln_r', 'sigma_daily_7']], on='date', how='left')

    # Calculate the frequency of "MINT" and "BURN" in the "tx_type" column
    mint_counts = merged_df[merged_df['tx_type'] == 'MINT'].groupby('date').size()
    burn_counts = merged_df[merged_df['tx_type'] == 'BURN'].groupby('date').size()

    # Ensure all dates are included, filling in any missing dates with 0 counts
    date_range = pd.date_range(start=merged_df['date'].min(), end=merged_df['date'].max())
    mint_counts = mint_counts.reindex(date_range, fill_value=0)
    burn_counts = burn_counts.reindex(date_range, fill_value=0)

    # Extract the "0" column (c-lvr), reindex to match the date range
    column_0 = merged_df.groupby('date')['0'].first().reindex(date_range)

    # Extract volatility data, reindex to match the date range
    sigma_ln_r = merged_df.groupby('date')['sigma_ln_r'].first().reindex(date_range).fillna(0)
    sigma_daily_7 = merged_df.groupby('date')['sigma_daily_7'].first().reindex(date_range).fillna(0)

    # Fill NaN values in 'column_0' with 0
    column_0 = column_0.fillna(0)

    # Aggregate liquidity by transaction type
    first_csv['liquidity'] = pd.to_numeric(first_csv['liquidity'], errors='coerce').abs()
    mint_volume_by_date = first_csv[first_csv['tx_type'] == 'MINT'].groupby('date')['liquidity'].sum().reindex(date_range, fill_value=0)
    burn_volume_by_date = first_csv[first_csv['tx_type'] == 'BURN'].groupby('date')['liquidity'].sum().reindex(date_range, fill_value=0)

    # Create DataFrames for mint and burn volume and frequency with a 'date' index
    daily_mint_burn_data = pd.DataFrame({
        'date': date_range,
        'mint_volume': mint_volume_by_date.values,
        'burn_volume': burn_volume_by_date.values,
        'mint_frequency': mint_counts.values,
        'burn_frequency': burn_counts.values
    })

    # Ensure 'date' column is in datetime format for both DataFrames before merging
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    daily_mint_burn_data['date'] = pd.to_datetime(daily_mint_burn_data['date'])

    # Merge the daily mint and burn data back into the original merged_df by 'date'
    merged_df = pd.merge(merged_df, daily_mint_burn_data, on='date', how='left')

    # Save the final DataFrame (including mint and burn volume and frequency) to CSV
    csv_file_path = f'{csv_output_dir}{pool}.csv'
    merged_df.to_csv(csv_file_path, index=False)
    print(f"CSV saved: {csv_file_path}")

    # Create a DataFrame for correlation calculation (collapse daily data)
    collapsed_df = daily_mint_burn_data.copy()
    collapsed_df['column_0'] = column_0.values
    collapsed_df['sigma_ln_r'] = sigma_ln_r.values
    collapsed_df['sigma_daily_7'] = sigma_daily_7.values

    # Sort data by date
    collapsed_df.sort_values(by='date', inplace=True)

    # Columns to use for correlation analysis
    correlation_columns = ['column_0', 'burn_frequency', 'burn_volume', 'mint_frequency', 'mint_volume', 'sigma_ln_r', 'sigma_daily_7']

    # Calculate correlation matrix and p-values
    correlation_matrix, p_value_matrix = calculate_corr_and_pvalues(collapsed_df, correlation_columns)

    # Save the correlation matrix to CSV
    corr_csv_path = f'{corr_output_dir}correlation_{pool}.csv'
    correlation_matrix.to_csv(corr_csv_path)
    print(f"Correlation matrix saved: {corr_csv_path}")

    # Save the p-value matrix to CSV
    pvalue_csv_path = f'{corr_output_dir}p_values_{pool}.csv'
    p_value_matrix.to_csv(pvalue_csv_path)
    print(f"P-value matrix saved: {pvalue_csv_path}")

    # Create the first plot: time series of MINT and BURN frequencies vs c-lvr (column "0"), volatility, and trading volumes
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot MINT and BURN frequencies on the left y-axis (first y-axis)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Frequency of MINT and BURN', color='tab:blue')
    ax1.plot(date_range, mint_counts, label='MINT Frequency', color='tab:green')
    ax1.plot(date_range, burn_counts, label='BURN Frequency', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.legend(loc='upper left')

    # Create a second y-axis for the "0" column (right side)
    ax2 = ax1.twinx()
    ax2.set_ylabel('C-LVR (Column 0)', color='tab:blue')
    ax2.plot(date_range, column_0, label='C-LVR (Column 0)', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # Create a third y-axis for volatility data (right side with offset)
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))  # Move the third y-axis outward
    ax3.set_ylabel('Volatility', color='darkorange')
    ax3.plot(date_range, sigma_ln_r, label='Sigma_ln_r (Volatility)', color='darkorange')
    ax3.plot(date_range, sigma_daily_7, label='Sigma_daily_7 (Volatility)', color='darkgoldenrod')
    ax3.tick_params(axis='y', labelcolor='darkorange')

    # Add two more y-axes for MINT and BURN trading volumes
    ax4 = ax1.twinx()
    ax4.spines['right'].set_position(('outward', 120))
    ax4.set_ylabel('Mint Trading Volume', color='tab:green')
    ax4.plot(date_range, mint_volume_by_date, label='Mint Volume', color='tab:green', linestyle='--')
    ax4.tick_params(axis='y', labelcolor='tab:green')

    ax5 = ax1.twinx()
    ax5.spines['right'].set_position(('outward', 180))
    ax5.set_ylabel('Burn Trading Volume', color='tab:red')
    ax5.plot(date_range, burn_volume_by_date, label='Burn Volume', color='tab:red', linestyle='--')
    ax5.tick_params(axis='y', labelcolor='tab:red')

    # Add legends for each axis
    ax2.legend(loc='upper right')
    ax3.legend(loc='center right')
    ax4.legend(loc='lower right')
    ax5.legend(loc='upper center')

    # Adjust scales
    ax2.set_ylim(column_0.min() * 0.9, column_0.max() * 1.1)  # Adjust scale for Column 0
    ax3.set_ylim(min(sigma_ln_r.min(), sigma_daily_7.min()) * 0.9,
                 max(sigma_ln_r.max(), sigma_daily_7.max()) * 1.1)  # Adjust scale for volatility
    ax4.set_ylim(0, mint_volume_by_date.max() * 1.1)  # Adjust scale for Mint Volume
    ax5.set_ylim(0, burn_volume_by_date.max() * 1.1)  # Adjust scale for Burn Volume

    # Save the plot to a PNG file with the pool name appended, in the img_output_dir
    img_file_path = f'{img_output_dir}{pool}.png'
    plt.savefig(img_file_path, format='png', dpi=300)
    print(f"Plot saved: {img_file_path}")

    # Show the plot
    plt.title(f'MINT and BURN Frequency, Volatility, C-LVR, and Trading Volumes for {pool}')
    plt.show()
