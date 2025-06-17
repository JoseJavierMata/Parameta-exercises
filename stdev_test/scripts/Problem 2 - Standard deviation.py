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
        Calculate rolling std between two dates.
        """
        start_dt = pd.to_datetime(start_time)
        end_dt = pd.to_datetime(end_time)
        
        print(f"Calculating rolling std from {start_time} to {end_time}")
        
        lookback_start = start_dt - timedelta(days=7)
        calc_df = df[df['snap_time'] >= lookback_start].copy()
                
        # Pre-allocate memory
        total_rows = len(calc_df) * len(self.price_columns)
        result_arrays = {
            'security_id': np.empty(total_rows, dtype=object),
            'snap_time': np.empty(total_rows, dtype='datetime64[ns]'),
            'price_type': np.empty(total_rows, dtype=object),
            'rolling_std': np.full(total_rows, np.nan)
        }
        
        row_idx = 0
        
        grouped = calc_df.groupby('security_id', sort=False)
        
        for security_id, group_data in grouped:
            group_size = len(group_data)
            
            time_diffs = group_data['snap_time'].diff().dt.total_seconds() / 3600
            
            # Create continuity mask using numpy
            continuity_array = np.abs(time_diffs.values[1:] - 1.0) < 0.01
            continuity_mask = np.zeros(group_size, dtype=bool)
            
            # Indicate positions where we have window_size consecutive valid hours
            for i in range(self.window_size-1, group_size):
                if i < self.window_size:
                    continuity_mask[i] = True
                else:
                    continuity_mask[i] = np.all(continuity_array[i-self.window_size+1:i])
            
            # Process all price columns for this security at once
            for price_col in self.price_columns:
                values = group_data[price_col].values
                
                rolling_stds = self._fast_rolling_std(values, self.window_size)
                
                # Apply continuity mask
                valid_stds = np.where(continuity_mask, rolling_stds, np.nan)
                
                # Filter to time range and valid values in one operation
                time_mask = ((group_data['snap_time'] >= start_dt) & 
                           (group_data['snap_time'] <= end_dt) & 
                           (~np.isnan(valid_stds)))
                
                if np.any(time_mask):
                    n_valid = np.sum(time_mask)
                    end_idx = row_idx + n_valid
                    
                    result_arrays['security_id'][row_idx:end_idx] = security_id
                    result_arrays['snap_time'][row_idx:end_idx] = group_data.loc[time_mask, 'snap_time'].values
                    result_arrays['price_type'][row_idx:end_idx] = price_col
                    result_arrays['rolling_std'][row_idx:end_idx] = valid_stds[time_mask]
                    
                    row_idx = end_idx
        
        if row_idx > 0:
            final_results = pd.DataFrame({
                k: v[:row_idx] for k, v in result_arrays.items()
            })
            return final_results.sort_values(['security_id', 'snap_time', 'price_type'])
        else:
            print("No results generated in the specified time range")
            return pd.DataFrame()
    
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
    
    # Get the project root directory (parent of scripts directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Set up file paths
    data_dir = os.path.join(project_root, 'data')
    results_dir = os.path.join(project_root, 'results')
    
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize ultra-fast calculator
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
    
    # Calculate rolling standard deviations with ultra-fast method
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
        print(f"Calculation time: {calculation_time:.3f} seconds")
        print(f"Total execution time: {total_time:.3f} seconds")
        
    else:
        print("No results to save")

if __name__ == "__main__":
    main()