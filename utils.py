import pandas as pd
import numpy as np
import re

def detect_and_convert_dates(df: pd.DataFrame, return_cols=False) -> list:
    date_cols = []
    date_patterns = [
        r'^^\d{4}-\d{1,2}-\d{1,2}$',  # YYYY-MM-DD
        r'^^\d{1,2}/\d{1,2}/\d{4}$',  # MM/DD/YYYY
        r'^^\d{1,2}-\w{3}-\d{4}$',    # DD-MMM-YYYY
        r'^^\d{1,2}\.\d{1,2}\.\d{4}$' # DD.MM.YYYY
    ]
    
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            print(f'detected date format 👍: {col}')
            date_cols.append(col)
            continue

        if pd.api.types.is_object_dtype(df[col]):
            sample = df[col].dropna().astype(str).head(5)
            if not sample.empty:
                # Check if any sample value matches a date pattern
                is_date_like = any(
                    any(re.match(pattern, x) for pattern in date_patterns)
                    for x in sample
                )
                if is_date_like:
                    converted = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                    if converted.notna().sum() >= len(df[col].dropna()) // 2:
                        print(f'Converted str_date to date format 👍: {col}')
                        df[col] = converted
                        date_cols.append(col)
    
    if return_cols:
        return df, date_cols
    return df

def date_to_numeric(dt, unit='days'):
    """
    Converts datetime or Series of datetime to numeric values (e.g., days since epoch).
    unit: 'days', 'seconds', etc.
    """
    if isinstance(dt, pd.Series) or isinstance(dt, pd.DataFrame):
        dt = pd.to_datetime(dt)
        return (dt - pd.Timestamp("1970-01-01")) // np.timedelta64(1, unit)
    else:
        dt = pd.to_datetime(dt)
        return (dt - pd.Timestamp("1970-01-01")) // np.timedelta64(1, unit)

def numeric_to_date(nums, unit='days'):
    """
    Converts numeric values (e.g., days since epoch) to datetime.
    unit: 'days', 'seconds', etc.
    """
    return pd.Timestamp("1970-01-01") + pd.to_timedelta(nums, unit=unit).