import pandas as pd


def engineer_churn_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Text-to-Voice Engagement Density Ratio
    # Captures shifting communication behaviors (+1 handles zero division safety)
    if "Frequency of SMS" in df.columns and "Seconds of Use" in df.columns:
        df["SMS_to_Voice_Ratio"] = df["Frequency of SMS"] / (df["Seconds of Use"] + 1)

    # 2. Value Trajectory Density per Month
    # Normalizes customer worth relative to their overall contract lifespan
    if "Customer Value" in df.columns and "Subscription  Length" in df.columns:
        df["Value_Density_per_Month"] = df["Customer Value"] / (df["Subscription  Length"] + 1)
    
    return df

