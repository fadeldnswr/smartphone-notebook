'''
Utility functions for the smartphone battery notebook project.
Includes functions for data processing, visualization, and model evaluation.
'''

import os
import sys
import pandas as pd
import numpy as np
from psycopg2 import connect

# Define function to create postgres connection
def connect_to_db(host, port, db_name, user, password):
  try:
    # Create connection
    conn = connect(
      host=host,
      port=port,
      dbname=db_name,
      user=user,
      password=password
    )
    # Check if connection created successfully
    if conn:
      print("Connection to database established successfully.")
    else:
      print("Failed to establish connection to database.")
    return conn
  except Exception as e:
    print(f"Error connecting to database: {e}")

# Create function to calculate SoH and battery cycles
def calculate_soh_and_cycles(df: pd.DataFrame, C0: int = 5000, threshold: int = 100, roll_win: int = 3) -> pd.DataFrame:
  try:
    # Prepare dataframe
    df = df.copy() 
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.sort_values("created_at").reset_index(drop=True)
    
    # Time for each sample
    df["delta_t_s"] = df["created_at"].diff().dt.total_seconds().fillna(0)
    df.loc[df["delta_t_s"] < 0, "delta_t_s"] = 0
    df.loc[df["delta_t_s"] > 3600, "delta_t_s"] = 0 
    
    if "charge_counter" in df.columns and df["charge_counter"].notna().any():
      # Asumsi charge_counter dalam µAh -> konversi ke mAh
      df["Q_mAh"] = df["charge_counter"] / 1000.0
      
      # Flag posisi full
      df["is_full"] = (df["battery_level"] >= threshold).astype(int)
      block_id = (df["is_full"].ne(df["is_full"].shift(1))).cumsum()
      df["full_block_id"] = np.where(df["is_full"] == 1, block_id, np.nan)
      
      # Ambil Q_mAh maksimum di tiap blok full sebagai Ct
      ct_map = (
        df.dropna(subset=["full_block_id"])
          .groupby("full_block_id")["Q_mAh"]
          .max()
          .to_dict()
      )
      
      df["Ct_mAh"] = np.nan
      for bid, ct in ct_map.items():
        idx = df.loc[df["full_block_id"] == bid, "Q_mAh"].idxmax()
        df.loc[idx, "Ct_mAh"] = ct
        
      # propagate Ct ke depan
      df["Ct_mAh"] = df["Ct_mAh"].ffill()
      valid_ct = df["Ct_mAh"].dropna()
      
      if not valid_ct.empty:
        C0_ref = valid_ct.quantile(0.95)
      else:
        C0_ref = C0
      
      # State of Health in a scale of 0 to 1
      df["SoH"] = df["Ct_mAh"] / C0_ref
      
      # Simple smoothing
      df["SoH_smooth"] = (
        df["SoH"].rolling(roll_win, min_periods=1, center=True).median()
      )
      
      # EFC from discharge
      df["delta_Q_mAh"] = df["Q_mAh"].diff().fillna(0)
      
      # Discharge 
      df["discharge_mAh"] = df["delta_Q_mAh"].apply(lambda x: -x if x < 0 else 0)
      
      # Cut off extreme discharge values (top 1%)
      q99 = df["discharge_mAh"].quantile(0.99)
      df.loc[df["discharge_mAh"] > q99, "discharge_mAh"] = q99
      
      # EFC dengan kapasitas referensi yang sama (C0_ref)
      df["EFC"] = df["discharge_mAh"].cumsum() / C0_ref
    
    else:
      df["batt_current_a"] = df["current_avg_ua"] / 1_000_000.0
      
      # Integrasi arus → Q_Ah → Q_mAh
      df["delta_Q_Ah"] = df["batt_current_a"] * df["delta_t_s"] / 3600.0
      df["Q_Ah"] = df["delta_Q_Ah"].cumsum()
      
      # Offset supaya mulai dari 0 (biar gak negatif)
      df["Q_Ah"] = df["Q_Ah"] - df["Q_Ah"].min()
      df["Q_mAh"] = df["Q_Ah"] * 1000.0
      
      # Estimasi Ct saat full
      df["is_full"] = (df["battery_level"] >= threshold).astype(int)
      block_id = (df["is_full"].ne(df["is_full"].shift(1))).cumsum()
      df["full_block_id"] = np.where(df["is_full"] == 1, block_id, np.nan)
      
      ct_map = (
        df.dropna(subset=["full_block_id"])
          .groupby("full_block_id")["Q_mAh"]
          .max()
          .to_dict()
      )
      
      df["Ct_mAh"] = np.nan
      for bid, ct in ct_map.items():
        idx = df.loc[df["full_block_id"] == bid, "Q_mAh"].idxmax()
        df.loc[idx, "Ct_mAh"] = ct
      df["Ct_mAh"] = df["Ct_mAh"].ffill()
      
      valid_ct = df["Ct_mAh"].dropna()
      if not valid_ct.empty:
        C0_ref = valid_ct.quantile(0.95)
      else:
        C0_ref = C0
        
      df["SoH"] = df["Ct_mAh"] / C0_ref
      df["SoH_smooth"] = df["SoH"].rolling(roll_win, min_periods=1, center=True).median()
      
      # EFC dari discharge
      df["delta_Q_mAh"] = df["Q_mAh"].diff().fillna(0)
      df["discharge_mAh"] = df["delta_Q_mAh"].apply(lambda x: -x if x < 0 else 0)
      q99 = df["discharge_mAh"].quantile(0.99)
      df.loc[df["discharge_mAh"] > q99, "discharge_mAh"] = q99
      df["EFC"] = df["discharge_mAh"].cumsum() / C0_ref
      df["SoH_pct"] = df["SoH"] * 100
      df["SoH_smooth_pct"] = df["SoH_smooth"] * 100
      
      return df
  except Exception as e:
    print(f"Error calculating SoH and cycles: {e}")