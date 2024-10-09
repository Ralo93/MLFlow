def create_interaction_features(df):
    df['foundation_roof_interaction'] = df['foundation_type'] + '_' + df['roof_type']
    return df


def create_position_stability_feature(df):
    df['is_stable_position'] = df['position'].isin(['j', 'o']).astype(int)
    return df

def create_superstructure_variety(df):
    df['superstructure_variety'] = (
        df['has_superstructure_adobe_mud'] +
        df['has_superstructure_mud_mortar_stone'] +
        df['has_superstructure_cement_mortar_brick'] +
        df['has_superstructure_timber'] +
        df['has_superstructure_rc_non_engineered'] +
        df['has_superstructure_rc_engineered']
    )
    return df


def create_secondary_use_features(df):
    df['has_critical_secondary_use'] = (
            df['has_secondary_use_school'] |
            df['has_secondary_use_health_post'] |
            df['has_secondary_use_gov_office'] |
            df['has_secondary_use_police']
    ).astype(int)

    df['has_secondary_use_combined'] = (
            df['has_secondary_use'] |
            df['has_secondary_use_agriculture'] |
            df['has_secondary_use_hotel'] |
            df['has_secondary_use_rental'] |
            df['has_secondary_use_institution'] |
            df['has_secondary_use_school'] |
            df['has_secondary_use_industry'] |
            df['has_secondary_use_health_post'] |
            df['has_secondary_use_gov_office'] |
            df['has_secondary_use_police'] |
            df['has_secondary_use_other']
    ).astype(int)

    return df


def create_family_density_feature(df):
    df['family_density'] = df['count_families'] / df['area_percentage']
    return df


def create_age_features(df):
    df['is_kinda_old'] = (df['age'] > 50).astype(int)
    df['age_squared'] = df['age'] ** 2
    return df


def create_geographic_risk_features(df):
    # Example: You can adjust this based on actual risk data
    df['geo_combined_id'] = df['geo_level_1_id'].astype(str) + '_' + df['geo_level_2_id'].astype(str) + '_' + df['geo_level_3_id'].astype(str)
    # If you have risk level data by geographic region, you can assign it here
    # df['geo_risk_level'] = some_external_geo_risk_mapping_function(df['geo_combined_id'])
    return df


def create_foundation_roof_features(df):
    df['strong_foundation'] = df['foundation_type'].isin(['h', 'i']).astype(int)
    df['strong_roof'] = df['roof_type'].isin(['n', 'x']).astype(int)
    return df


def create_structural_strength_features(df):
    df['is_engineered_rc'] = df['has_superstructure_rc_engineered']
    df['is_non_engineered'] = (
        df['has_superstructure_adobe_mud'] |
        df['has_superstructure_mud_mortar_stone'] |
        df['has_superstructure_rc_non_engineered']
    ).astype(int)
    return df

def create_density_features(df):
    df['area_height_ratio'] = df['area_percentage'] / df['height_percentage']
    df['avg_floor_height'] = df['height_percentage'] / df['count_floors_pre_eq']
    return df


from scipy.stats import skew


def get_right_skewed_columns(df, skew_threshold=0.5):
    """
    Returns the names of columns that are right-skewed based on the skewness value, excluding binary columns.

    Parameters:
    - df: The input DataFrame (numerical columns only).
    - skew_threshold: The skewness threshold above which a column is considered right-skewed (default is 0.5).

    Returns:
    - List of column names that are right-skewed.
    """
    right_skewed_columns = []

    # Iterate through each column in the dataframe
    for col in df.columns:
        # Check if the column has more than 2 unique values (to avoid binary columns)
        if df[col].nunique() > 2:
            # Calculate skewness for each column
            col_skewness = skew(df[col].dropna())  # Drop NaN values to avoid issues

            # Check if the skewness is above the specified threshold (indicating right-skewness)
            if col_skewness > skew_threshold:
                right_skewed_columns.append(col)

    return right_skewed_columns


import numpy as np


def apply_log_transformation(df, columns):
    """
    Applies log transformation to the specified columns of the DataFrame.

    Parameters:
    - df: The input DataFrame.
    - columns: List of column names to apply the log transformation on.

    Returns:
    - DataFrame with log-transformed columns.
    """
    df_transformed = df.copy()

    # Apply log transformation to each specified column
    for col in columns:
        # Add a small constant to avoid log(0) and handle any zeros or negatives
        df_transformed[col] = np.log1p(df[col])

    return df_transformed
