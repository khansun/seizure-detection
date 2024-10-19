import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from mne.viz import plot_filter

def sinc_filter(figureDir, sfreq, f_p=40.0):
    """
    Designs and visualizes a finite impulse response (FIR) sinc filter.

    Parameters:
    - figureDir (str): Directory path where the filter plot will be saved.
    - sfreq (float): Sampling frequency in Hz.
    - f_p (float): Pass-band frequency in Hz (default is 40.0).
    
    This function creates a sinc filter based on the specified sampling frequency
    and pass-band frequency, then visualizes the filter response and saves the plot
    to the specified directory.
    """
    nyq = sfreq / 2.  # Nyquist frequency
    freq = [0., f_p, f_p, nyq]
    gain = [1., 1., 0., 0.]

    # Filter configuration
    n = int(round(1 * sfreq))  # Filter length
    n -= n % 2 - 1  # Ensure it's an odd number
    t = np.arange(-(n // 2), n // 2 + 1) / sfreq  # Create time array for sinc function
    h = np.sinc(2 * f_p * t) / (4 * np.pi)  # Sinc filter

    # Frequency limits for plot
    flim = (1., sfreq / 2.)

    # Plot the filter
    fig = plot_filter(h, sfreq, freq, gain, 'No Window', flim=flim, compensate=True)

    # Text box with filter details
    textstr = '\n'.join((
        'Pass-band Frequency: ' + str(f_p),
        'Filter Length: 1s'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # Add text box to the plot
    fig.text(0.7, 0.9, textstr, fontsize=14, bbox=props)

    # Display the plot
    plt.show()
    fig.savefig(figureDir)


def windowed_fir_filter(figureDir, f_p, trans_bandwidth=10, sfreq=5000.0, filter_length_seconds=1):
    """
    Designs and visualizes a windowed finite impulse response (FIR) filter.

    Parameters:
    - figureDir (str): Directory path where the filter plot will be saved.
    - f_p (float): Pass-band frequency in Hz.
    - trans_bandwidth (float): Transition bandwidth in Hz (default is 10).
    - sfreq (float): Sampling frequency in Hz (default is 5000.0).
    - filter_length_seconds (float): Length of the filter in seconds (default is 1).
    
    This function creates a windowed FIR filter based on the specified parameters,
    visualizes the filter response, and saves the plot to the specified directory.
    """
    # Calculate Nyquist frequency
    nyq = sfreq / 2.  # Nyquist frequency

    # Define frequency and gain vectors
    f_s = f_p + trans_bandwidth  # Stop frequency
    freq = [0., f_p, f_s, nyq]
    gain = [1., 1., 0., 0.]

    # Calculate filter coefficients using firwin2
    n = int(round(filter_length_seconds * sfreq))  # Filter length in samples
    n -= n % 2 - 1  # Ensure it's an odd number
    h = signal.firwin2(n, freq, gain, nyq=nyq)

    # Frequency limits for plot
    flim = (1., nyq)

    # Plot the filter response
    fig = plot_filter(h, sfreq, freq, gain, 'Windowed', flim=flim, compensate=True)

    # Create a text box with filter details
    textstr = '\n'.join((
        f'Pass-band Frequency: {f_p} Hz',
        f'Filter Length: {filter_length_seconds} s',
        f'Transition Bandwidth: {trans_bandwidth} Hz'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # Add text box to the plot
    fig.text(0.7, 0.88, textstr, fontsize=14, bbox=props)

    # Display the plot
    plt.show()
    fig.savefig(figureDir)

if __name__ == "__main__":
    design_sinc_filter('sinc_filter.png', sfreq=5000.0)
    design_windowed_fir_filter('windowed_fir_filter.png', f_p=40.0)
