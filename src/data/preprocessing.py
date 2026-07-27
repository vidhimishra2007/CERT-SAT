"""
Per-channel preprocessing. SMAP/MSL channels have very different scales
(see EDA scale_summary), so each channel is scaled independently rather
than pooling statistics across channels.
"""

from sklearn.preprocessing import StandardScaler


def scale_channel(train_arr, test_arr):
    """
    Fit StandardScaler on train only, apply to both train and test.
    Prevents test-set statistics from leaking into normalization.
    """
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_arr)
    test_scaled = scaler.transform(test_arr)
    return train_scaled, test_scaled

def scale_channel_with_scaler(train_arr, test_arr):
    """
    Fit StandardScaler on train only, apply to both train and test, and
    return the fitted scaler so it can be saved for live inference.
    """
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_arr)
    test_scaled = scaler.transform(test_arr)
    return train_scaled, test_scaled, scaler
