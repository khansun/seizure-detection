def welchBandpower(data, sf, band, display_output=False):
    """
    Calculate the band power of the given data using the Welch method.

    Parameters:
    - data (DataFrame): Input data containing the signals.
    - sf (float): Sampling frequency of the data.
    - band (list or array): Frequency band for which to calculate the power (e.g., [8, 12] for alpha band).
    - display_output (bool): Whether to print the results (default is False).

    Returns:
    - Series: Mean power in the specified frequency band for each channel.
    """
    # Convert all columns to numeric, coercing errors to NaN, and drop NaN values
    data = data.apply(pd.to_numeric, errors='coerce').dropna()

    # Ensure band is an array
    band = np.asarray(band)
    low, high = band

    # Calculate the appropriate nperseg
    nperseg = min(len(data), int((2 / low) * sf))  # Ensure nperseg does not exceed the length of data

    # Compute the periodogram using Welch's method
    freqs, psd = welch(data.values,  # Pass data as numpy array
                       sf,
                       nperseg=nperseg,
                       scaling='density',
                       axis=0)

    # Convert PSD to a DataFrame for easier manipulation
    psd_df = pd.DataFrame(psd, index=freqs, columns=data.columns)

    if display_output:
        print("Welch Output:")
        psd_df.index.name = 'Frequency (Hz)'
        psd_df.columns = ['Power']
        display(psd_df)

    # Find closest indices for the frequency band
    idx_min = np.searchsorted(freqs, low)
    idx_max = np.searchsorted(freqs, high)

    # Ensure the indices are within bounds
    idx_min = max(0, idx_min - 1)
    idx_max = min(len(freqs) - 1, idx_max)

    # Select frequencies of interest and compute mean power
    band_psd = psd_df.iloc[idx_min:idx_max, :].mean()

    if display_output:
        print("\nMean Frequency Band Power:")
        display(band_psd)

    return band_psd


def welchPowerMeasure(data):
    """
    Calculate the mean power across specified frequency bands for given EEG data using the Welch method.

    Parameters:
    - data (DataFrame): A DataFrame containing EEG signal data with columns representing different channels.

    Returns:
    - DataFrame: A DataFrame where each row corresponds to a frequency band and each column corresponds to a channel.
                  The values represent the mean power in the specified frequency bands.

    Frequency Bands:
    - Delta: [0.1, 4] Hz
    - Theta: [4, 8] Hz
    - Alpha: [8, 12] Hz
    - Beta: [12, 30] Hz
    - Gamma: [30, 70] Hz

    The function processes the input data by calculating the power spectral density (PSD) for each frequency band
    using the Welch method. The mean power in each band is computed for all channels in the DataFrame.
    """
    bandpasses = [[[0.1, 4], 'power_delta'],
                  [[4, 8], 'power_theta'],
                  [[8, 12], 'power_alpha'],
                  [[12, 30], 'power_beta'],
                  [[30, 70], 'power_gamma']]

    welch_df = pd.DataFrame()
    for bandpass, freq_name in bandpasses:
        bandpass_data = welchBandpower(data, sample_rate, bandpass)
        bandpass_data = pd.Series(bandpass_data, index=data.columns).rename(freq_name)

        if welch_df.empty:
            welch_df = pd.DataFrame(bandpass_data).T
        else:
            welch_df = pd.concat([welch_df, pd.DataFrame(bandpass_data).T], axis=0)

    return welch_df