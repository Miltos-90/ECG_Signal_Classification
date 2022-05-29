from tsfresh.feature_extraction import feature_calculators as fc
from tqdm.notebook import tqdm
import pandas as pd
import pywt 
import numpy as np

def feature_extractor(df, noLevels, waveletName):
    '''
    Preprocesses the entire dataframe (computes DWT + feature extract from each sub-band)
    Inputs:
        df:          2D np.array from which the features will be extracted (each row is one signal to be preprocessed)
        noLevels:    Number of sub-bands to compute
        waveletName: Name of the wavelet
    Outputs:
        Processed dataframe
    '''
    
    # List of dictionaries containing the extracted features
    featDict = []

    for row in tqdm(range(df.shape[0])):

        sample   = df[row].flatten() # Grab sample    
        signals  = _compute_DWT(sample, noLevels, waveletName)
        features = _compute_features(signals)
        featDict.append(features)

    return pd.DataFrame(featDict)

def _make_feat_dict(data, id_name):
    ''' Makes a dictionary out of the specified features
        Inputs: 
            data:    1D np.array containing the signal from which the features will be extracted
            id_name: str Identifier of the signal that gets appended after the name of every feature
        Outputs:
            d: dictionary of extracted features
    '''
    
    d = {"abs_energy"                + id_name: fc.abs_energy(data), 
         "absolute_maximum"          + id_name: fc.absolute_maximum(data),
         "absolute_sum_of_changes"   + id_name: fc.absolute_sum_of_changes(data),
         "benford_correlation"       + id_name: fc.benford_correlation(data),
         "cid_ce"                    + id_name: fc.cid_ce(data, normalize = True),
         "first_location_of_maximum" + id_name: fc.first_location_of_maximum(data),
         "first_location_of_minimum" + id_name: fc.first_location_of_minimum(data),
         "kurtosis"                  + id_name: fc.kurtosis(data),
         "maximum"                   + id_name: fc.maximum(data),
         "mean"                      + id_name: fc.mean(data),
         "mean_abs_change"           + id_name: fc.mean_abs_change(data),
         "median"                    + id_name: fc.median(data),
         "minimum"                   + id_name: fc.minimum(data),
         "root_mean_square"          + id_name: fc.root_mean_square(data),
         "skewness"                  + id_name: fc.skewness(data),
         "standard_deviation"        + id_name: fc.standard_deviation(data),
         "sum_values"                + id_name: fc.sum_values(data),
         "variance"                  + id_name: fc.variance(data)
        }
    
    return d

def _compute_DWT(sample, noLevels, waveletName):
    '''
    Compute the DWT transformation of a signal with a specified number of sub-bands from a given wavelet
        Inputs:
            sample:      1D np.array containing the signal to which the DWT will be applied
            noLevels:    Number of sub-bands to compute
            waveletName: Name of the wavelet
        Outputs: 
            signals: List of extracted signals (np.arrays)
    '''
    
    sample  = np.trim_zeros(sample, trim = 'b') # Trim trailing zeroes
    signals = [] # Empty list to hold the signals from each sub-band
    
    for ii in range(noLevels):

        # Run wavelet transformation
        (sample, coeff_d) = pywt.dwt(sample, waveletName)
        
        # Add signal to list
        signals.append(sample)
        
        # From level 0 also retain detail coefficients
        if ii == 0: 
            signals.append(coeff_d)
    
    return signals

def _compute_features(signals):
    '''
    Extracts features from a list of signals
    Inputs: 
        signals: list of np.arrays
    Outputs:
        features: dictionary containing the extracted features
    '''
    
    # Extract features from each sub-band
    dictList = [] # List of dictionaries to hold the features from each level of the DWT
    
    for sig_id, signal in enumerate(signals):
        
        d = _make_feat_dict(signal, id_name = str(sig_id))
        dictList.append(d)
        
    # Make flat Dict from the DictList
    features = {k:v for element in dictList for k,v in element.items()}
    
    return features
