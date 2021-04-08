import pandas as pd

def locate_features_with_too_many_missing_values(df, missing_threshold):
    bad_cols = []
    for col in df.columns:
        nan_count = df[col].isna().sum()
        num_rows = df.shape[0]
        nan_perc = nan_count/num_rows
        if nan_perc > 0.1:
            bad_cols.append(col)
    return bad_cols

def clean_up_data(df):
    df['Timestamp'] = pd.to_datetime(df['Date_time'], infer_datetime_format=True, utc=True)
    df.drop_duplicates(subset=['Timestamp'], keep='first', inplace=True)
    
    df = df.set_index(pd.DatetimeIndex(df['Timestamp']))
    df.drop(columns=['Wind_turbine_name','Date_time', 'Timestamp'], inplace=True)
    
    df.dropna(axis=1, how='all', inplace=True)
    df.dropna(axis=0, how='all', inplace=True)
    
    df.sort_index(inplace=True)

    df_r = df.resample('10T').ffill(limit=1)
    
    bad_cols = locate_features_with_too_many_missing_values(df_r, 0.1)
    df_r.drop(columns=bad_cols, inplace=True)
    
    df_r.index = pd.to_datetime(df_r.index).strftime("%Y-%m-%dT%H:%M:%S.%f")
    return df_r

def map_features(df_turbine, df_description):
    feature_map = {}
    for var_name in df_description['Variable_name']:
        cols = [x for x in df_turbine.columns if var_name in x]
        # print(f'{var_name}: {cols}')
        feature_map[var_name] = cols
        
    return feature_map

def order_columns(df_turbine, df_description):
    feature_map = map_features(df_turbine, df_description)
    df_list = []
    for k in feature_map.keys():
        df_list.append(df_turbine[feature_map[k]])
    df_ord = pd.concat(df_list, axis=1)
    return df_ord