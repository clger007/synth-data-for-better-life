import pandas as pd
import numpy as np

def detect_and_convert_dates(df):
    """
    Detects columns with date information in a DataFrame, including:
    - Columns already of datetime type
    - Columns with string dates (e.g., '2020-10-10', '2020-OCT-10', '2020AUG12')
    Converts string date columns to datetime.
    Returns a new DataFrame with converted columns.
    """
    df = df.copy()
    for col in df.columns:
        # If column is datetime, continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        # If column is string/object, check if it looks like dates
        if pd.api.types.is_object_dtype(df[col]):
            sample = df[col].dropna().astype(str).head(10)
            date_like = sample.apply(
                lambda x: pd.to_datetime(x, errors='coerce', infer_datetime_format=True)
            )
            # If at least half of sample can be parsed as date, convert entire column
            if date_like.notna().sum() >= len(sample) // 2:
                df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
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
    return pd.Timestamp("1970-01-01") + pd.to_timedelta(nums, unit=unit)