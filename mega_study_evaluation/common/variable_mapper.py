"""Common variable mapping utilities for mega study evaluation scripts."""

import numpy as np


def create_domain_maps(DV_vars, social, cognitive, known, pref, stim, know, politics):
    """
    Create all standard domain mapping dictionaries.
    
    Args:
        DV_vars: List of dependent variable names
        social: List of 0/1 values for social domain
        cognitive: List of 0/1 values for cognitive domain
        known: List of 0/1 values for known human bias
        pref: List of 0/1 values for preference measure
        stim: List of 0/1 values for stimuli dependent
        know: List of 0/1 values for knowledge question
        politics: List of 0/1 values for political question
    
    Returns:
        Dictionary containing all domain maps
    """
    return {
        'DV_vars_social_map': dict(zip(DV_vars, social)),
        'DV_vars_cognitive_map': dict(zip(DV_vars, cognitive)),
        'DV_vars_known_map': dict(zip(DV_vars, known)),
        'DV_vars_pref_map': dict(zip(DV_vars, pref)),
        'DV_vars_stim_map': dict(zip(DV_vars, stim)),
        'DV_vars_know_map': dict(zip(DV_vars, know)),
        'DV_vars_politics_map': dict(zip(DV_vars, politics))
    }


def add_min_max_columns(df, min_map, max_map):
    """
    Add _min and _max columns to dataframe.
    
    Args:
        df: DataFrame to modify
        min_map: Dictionary of variable to min value
        max_map: Dictionary of variable to max value
    
    Returns:
        Modified DataFrame (modifies in place and returns)
    """
    for var in min_map:
        df[f"{var}_min"] = min_map[var]
    
    for var in max_map:
        df[f"{var}_max"] = max_map[var]
    
    return df


def check_condition_variable(condition_vars):
    """
    Check if condition variable exists and return relevant info.
    
    Args:
        condition_vars: List of condition variable names (often [""] or ["Group"])
    
    Returns:
        Tuple of (cond_exists, cond, cond_h, cond_t)
    """
    if condition_vars and condition_vars[0].strip():
        cond = condition_vars[0]
        cond_h = f"{cond}_human"
        cond_t = f"{cond}_twin"
        cond_exists = True
    else:
        cond = None
        cond_h = None
        cond_t = None
        cond_exists = False
    
    return cond_exists, cond, cond_h, cond_t


def build_min_max_maps(DV_vars, DV_vars_min, DV_vars_max, raw_vars=None, raw_vars_min=None, raw_vars_max=None):
    """
    Build min/max maps from variable lists.
    
    Args:
        DV_vars: List of dependent variables
        DV_vars_min: List of min values for DVs
        DV_vars_max: List of max values for DVs
        raw_vars: Optional list of raw variables
        raw_vars_min: Optional list of min values for raw vars
        raw_vars_max: Optional list of max values for raw vars
    
    Returns:
        Tuple of (min_map, max_map)
    """
    min_map = dict(zip(DV_vars, DV_vars_min))
    max_map = dict(zip(DV_vars, DV_vars_max))
    
    if raw_vars and raw_vars_min and raw_vars_max:
        min_map.update(dict(zip(raw_vars, raw_vars_min)))
        max_map.update(dict(zip(raw_vars, raw_vars_max)))
    
    return min_map, max_map