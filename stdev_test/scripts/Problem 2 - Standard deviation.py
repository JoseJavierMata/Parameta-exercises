import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict
import warnings
import os
warnings.filterwarnings('ignore')

class RollingStDev:
    """
    Implementation of a rolling standard deviation calculator using vectorization.
    Generates results for ALL hours in the specified range.
    """
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.price_columns = ['bid', 'mid', 'ask']
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load parquet data."""
        print(f"Loading data from {filepath}...")
        
        # Read with optimized dtypes
        df = pd.read_parquet(filepath)
        
        if not pd.api.types.is_datetime64_any_dtype(df['snap_time']):
            df['snap_time'] = pd.to_datetime(df['snap_time'])
        
        df = df.sort_values(['security_id', 'snap_time'], kind='stable').reset_index(drop=True)
        
        return df
    
    def calculate_rolling_std_batch(self, df: pd.DataFrame,
                                  start_time: str = "2021-11-20 00:00:00", 
                                  end_time: str = "2021-11-23 09:00:00") -> pd.DataFrame:
        """
        Calculate rolling std for ALL hours between start and end time.
        """
        start_dt = pd.to_datetime(start_time)
        end_dt = pd.to_datetime(end_time)
        
        print(f"Calculating rolling std from {start_time} to {end_time}")
        
        # Get unique security IDs
        security_ids = df['security_id'].unique()
        
        # Generate all hours in the range
        all_hours = pd.date_range(start=start_dt, end=end_dt, freq='H')
        print(f"Total hours to process: {len(all_hours)}")
        print(f"Total securities: {len(security_ids)}")
        
        # Create complete grid of all combinations (security_id, snap_time, price_type)
        complete_index = pd.MultiIndex.from_product([
            security_ids,
            all_hours,
            self.price_columns
        ], names=['security_id', 'snap_time', 'price_type'])
        
        # Create empty DataFrame with all combinations
        complete_df = pd.DataFrame(index=complete_index).reset_index()
        complete_df['rolling_std'] = np.nan
        
        # Now calculate rolling stds
        lookback_start = start_dt - timedelta(days=7)
        calc_df = df[df['snap_time'] >= lookback_start].copy()
        
        # Pre-allocate arrays for calculated results
        result_data = []
        
        grouped = calc_df.groupby('security_id', sort=False)
        
        for security_id, group_data in grouped:
            group_size = len(group_data)
            
            # Calculate time differences for continuity check
            time_diffs = group_data['snap_time'].diff().dt.total_seconds() / 3600
            
            # Create continuity mask using numpy
            continuity_array = np.abs(time_diffs.values[1:] - 1.0) < 0.01
            continuity_mask = np.zeros(group_size, dtype=bool)
            
            # Mark positions where we have window_size consecutive valid hours
            for i in range(self.window_size-1, group_size):
                if i < self.window_size:
                    continuity_mask[i] = True
                else:
                    continuity_mask[i] = np.all(continuity_array[i-self.window_size+1:i])
            
            # Process all price columns for this security at once
            for price_col in self.price_columns:
                values = group_data[price_col].values
                
                # Calculate rolling stds using pandas optimized method
                rolling_stds = self._fast_rolling_std(values, self.window_size)
                
                # Apply continuity mask
                valid_stds = np.where(continuity_mask, rolling_stds, np.nan)
                
                # Filter to time range
                time_mask = ((group_data['snap_time'] >= start_dt) & 
                           (group_data['snap_time'] <= end_dt))
                
                # Add valid results to list
                valid_indices = np.where(time_mask & ~np.isnan(valid_stds))[0]
                
                for idx in valid_indices:
                    result_data.append({
                        'security_id': security_id,
                        'snap_time': group_data.iloc[idx]['snap_time'],
                        'price_type': price_col,
                        'rolling_std': valid_stds[idx]
                    })
        
        # Create DataFrame from calculated results
        if result_data:
            calculated_df = pd.DataFrame(result_data)
            
            complete_df = complete_df.merge(
                calculated_df,
                on=['security_id', 'snap_time', 'price_type'],
                how='left',
                suffixes=('_drop', '')
            )
            
            # Use calculated values where available
            complete_df['rolling_std'] = complete_df['rolling_std'].fillna(complete_df['rolling_std_drop'])
            complete_df = complete_df.drop(columns=['rolling_std_drop'])
        
        complete_df = complete_df.sort_values(['security_id', 'snap_time', 'price_type'])
        
        # # Print summary statistics
        # total_rows = len(complete_df)
        # valid_stds = complete_df['rolling_std'].notna().sum()
        # print(f"\nResults summary:")
        # print(f"Total rows generated: {total_rows}")
        # print(f"Rows with valid std: {valid_stds}")
        # print(f"Rows with NaN: {total_rows - valid_stds}")
        # print(f"Coverage: {valid_stds/total_rows*100:.1f}%")
        
        return complete_df
    
    @staticmethod
    def _fast_rolling_std(values: np.ndarray, window_size: int) -> np.ndarray:
        """
        Rolling standard deviation using pandas C-implemented rolling functions.
        """
        n = len(values)
        result = np.full(n, np.nan)
        
        if n < window_size:
            return result
        
        series = pd.Series(values)
        rolling_std = series.rolling(window=window_size, min_periods=window_size).std()
        return rolling_std.values

def main():
    """Main execution function optimized for maximum speed."""
    start_time = datetime.now()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    data_dir = os.path.join(project_root, 'data')
    results_dir = os.path.join(project_root, 'results')
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize calculator
    calculator = RollingStDev(window_size=20)
    
    # Load data
    input_filepath = os.path.join(data_dir, "stdev_price_data.parq.gzip")
    
    try:
        df = calculator.load_data(input_filepath)
    except FileNotFoundError:
        print(f"Error: File not found at '{input_filepath}'")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Expected data directory: {data_dir}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Calculate rolling standard deviations
    calculation_start = datetime.now()
    results = calculator.calculate_rolling_std_batch(
        df, 
        start_time="2021-11-20 00:00:00",
        end_time="2021-11-23 09:00:00"
    )
    calculation_time = (datetime.now() - calculation_start).total_seconds()
    
    # Save results
    if not results.empty:
        output_filepath = os.path.join(results_dir, "rolling_std_results.csv")
        results.to_csv(output_filepath, index=False)
        print(f"\nResults saved to: {output_filepath}")
        
        # Performance summary
        total_time = (datetime.now() - start_time).total_seconds()
        print(f"\nPerformance metrics:")
        print(f"Calculation time: {calculation_time:.3f} seconds")
        print(f"Total execution time: {total_time:.3f} seconds")
        
        # # Final validation
        # n_securities = results['security_id'].nunique()
        # n_hours = 82  # From Nov 20 00:00 to Nov 23 09:00
        # expected_total = n_securities * n_hours * 3  # 3 price types
        # actual_total = len(results)
        
        # print(f"\nFinal validation:")
        # print(f"Expected rows (all hours): {expected_total}")
        # print(f"Actual rows: {actual_total}")
        # print(f"All hours included: {'YES' if expected_total == actual_total else 'NO'}")
        
    else:
        print("No results to save")

if __name__ == "__main__":
    main()