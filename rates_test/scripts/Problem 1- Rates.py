import pandas as pd 
import numpy as np
import time 
import pyarrow as pa
import os

# start = time.time()

class Rates:
    def __init__(self):
        self.data_path = '../data/'
        self.results_path = '../results/'
        
        # Create folder if not exists
        os.makedirs(self.results_path, exist_ok=True)
    
    def load_data(self):
        """Load all files"""
        print("Loading data...")
        
        # Load data
        self.df_rates_ccy = pd.read_csv(os.path.join(self.data_path, 'rates_ccy_data.csv'), dtype={'ccy_pair': 'category', 'convert_price': bool})
        self.df_rates_price = pd.read_parquet(os.path.join(self.data_path, 'rates_price_data.parq.gzip'), engine='pyarrow')
        self.df_rates_spot = pd.read_parquet(os.path.join(self.data_path, 'rates_spot_rate_data.parq.gzip'), engine='pyarrow')
        
        # Convert to category
        self.df_rates_ccy['ccy_pair'] = self.df_rates_ccy['ccy_pair'].astype('category')
        self.df_rates_price['ccy_pair'] = self.df_rates_price['ccy_pair'].astype('category')
        self.df_rates_spot['ccy_pair'] = self.df_rates_spot['ccy_pair'].astype('category')
        
        # Convert to timestamp
        self.df_rates_price['timestamp'] = pd.to_datetime(self.df_rates_price['timestamp'])
        self.df_rates_spot['timestamp'] = pd.to_datetime(self.df_rates_spot['timestamp'])
        
        print("Data loaded")
    
    def process_data(self):
        """Procees and join data"""
        print("Processing data...")
        
        # Join price data with pairs data
        df_rates_price_temp = self.df_rates_price.set_index('ccy_pair')
        df_rates_ccy_temp = self.df_rates_ccy.set_index('ccy_pair')
        df_rates_price_ccy = df_rates_price_temp.join(df_rates_ccy_temp)
        df_rates_price_ccy = df_rates_price_ccy.reset_index()
        df_rates_price_ccy['ccy_pair'] = df_rates_price_ccy['ccy_pair'].astype('category')
        
        # Split data that need conversion
        filter_convert = df_rates_price_ccy['convert_price'] == True
        df_convert = df_rates_price_ccy[filter_convert].copy()
        df_no_convert = df_rates_price_ccy[~filter_convert].copy()
        
        #Order by timestamp
        df_convert = df_convert.sort_values('timestamp')
        df_rates_spot_sorted = self.df_rates_spot.sort_values('timestamp')
        

        # merge_asof for data that need conversion
        if len(df_convert) > 0:
            df_converted = pd.merge_asof(
                df_convert,
                df_rates_spot_sorted[['timestamp', 'ccy_pair', 'spot_mid_rate']],
                on='timestamp',
                by='ccy_pair',
                direction='backward',
                tolerance=pd.Timedelta('1h')
            )
        else:
            df_converted = df_convert
        
        # Calculate new prices
        if len(df_converted) > 0:
            df_converted['new_price'] = (
                df_converted['price'] / df_converted['conversion_factor'] + 
                df_converted['spot_mid_rate']
            )
        else:
            df_converted['new_price'] = np.nan
        
        # Conversion diagnose
        df_converted['calculation_status'] = df_converted.apply(self._diagnose_conversion, axis=1)
        df_converted['price_calculated'] = df_converted['calculation_status'].str.startswith('SUCCESS')
        
        # Process data which do not need conversion
        df_no_convert['new_price'] = df_no_convert['price']
        df_no_convert['spot_mid_rate'] = np.nan
        # df_no_convert['calculation_status'] = "SUCCESS: No conversion needed"

        df_no_convert['calculation_status'] = np.where(
            pd.isna(df_no_convert['convert_price']),
            "ERROR: No info about this pair in rates_ccy_data.csv",
            "SUCCESS: No conversion needed"
        )
        df_no_convert['price_calculated'] = True
        
        # Concatenate both dataframes
        self.df_final = pd.concat([df_converted, df_no_convert], ignore_index=True)
        
        print("Process completed")
    
    def _diagnose_conversion(self, row):
        """Control conversion status"""
        if pd.isna(row['price']):
            return "ERROR: Missing original price"
        elif pd.isna(row['conversion_factor']):
            return "ERROR: Missing conversion_factor"
        elif pd.isna(row['spot_mid_rate']):
            return "ERROR: No spot_mid_rate found within 1h window"
        else:
            return "SUCCESS: Price calculated"
    
    def save_results(self, filename='final_prices.csv'):
        """Save data in results folder"""
        output_path = os.path.join(self.results_path, filename)
        self.df_final.to_csv(output_path, index=False)
        print(f"Results saved in: {output_path}")
    
    def run_analysis(self):
        """Execute complete process"""
        self.load_data()
        self.process_data()
        self.save_results()
        return self.df_final


# Ejecutar el análisis
if __name__ == "__main__":
    
    start = time.time()

    rates_processor = Rates()
    df_result = rates_processor.run_analysis()
    
    end = time.time()
    print(f"Execution time: {(end - start) * 1000:.2f} ms")