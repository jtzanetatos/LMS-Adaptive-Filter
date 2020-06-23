'''
'''
import numpy as np
import sys
import wx
from scipy.io import wavfile
import os
import matplotlib.pyplot as plt
from scipy.signal import  firwin, convolve

__author__ = "Iason Tzanetatos"
__version__ = "1.0.0"
__status__ = "Prototype" # "Prototype", "Development", "Product"

# TODO: fix convolution function, fix user input default states,
# Fix on main function how each function outputs interact with one antoher.
# Re-implement Kaiser coefficients.

def fir_filter(x, fs):
    '''
    FIR 129th order Kaiser filter. Each element of the signal's array is entered
    into the 'Kaiser_filter_129' function & filtering is performed.
    
    Parameters
    ----------
    x : float 32 array
        Array containing the input (potentially) noised signal.
    fs: int
        The sampling frequency of the input audio signal.
    
    Returns
    -------
    fir_out : float 64 array
        Array containing the filtered signal.
    
    '''
    def fir_func(x, fs):
        '''
        
        
        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        fs : TYPE
            DESCRIPTION.
        
        Returns
        -------
        filt_sig : TYPE
            DESCRIPTION.
        
        '''
        
        # Default filter order
        numtaps = 14
        
        # Sanity check for user input
        while numtaps < 2:
            # Ask user input
            try:
                numtaps = np.uint16(input("Enter filter order (default: 14) >> "))
            except ValueError:
                print("User did not enter number or invalid number entered")
                numtaps = np.uint16(input("Enter filter order (default: 14) >> "))
        # Cutoff frequencies
        cutoff = np.array([60, 7960], dtype=np.uint32) / fs
        
        # Evaluate filter coefficients
        h = firwin(numtaps=numtaps, cutoff=cutoff, pass_zero='bandpass', fs=fs)
        
        # Convolve noised signal with filter coefficients
        filt_sig = convolve(x, h)
        
        # Return filtered signal
        return filt_sig
    
    # Ask user if deterministic filtering is desired
    filt_flg = input("Filter input signal by deterministic means? [Y/n] >> ").lower() or 'y'
    
    # Sanity check of user input
    while filt_flg != 'y' and filt_flg != 'n':
        filt_flg = input("Filter input signal by deterministic means? [Y/n] >> ").lower()
    
    # User opted for no filtering
    if filt_flg == 'n':
        print('User opted for no deterministic filtering, returning input signal.')
        
        # Return input signal
        return x
    # User opted for filtering
    else:
        # Initialize output
        fir_out = np.zeros_like(x, dtype=np.float64)
        
        # Print message to user
        print("Filtering..")
        # Check number of audio channels
        try:
            channels = x.shape[1]
            print("Number of audio channels present: %" %channels)
            # Loop through each audio channel
            for ch in range(channels):
                fir_out[:, ch] = fir_func(x[:, ch], fs)
        except IndexError:
            print("Audio file has one audio channel.")
            fir_out = fir_func(x, fs)
        
        # Print end of filtering process
        print("Done filtering.")
        # Return filtered signal
        return fir_out

def plot_signal(signal, sample_rate):
    '''
    A visualization function that asks the user whether a plot of either the
    input or output (filtered) signal is desired.
    
    Parameters
    ----------
    signal : float 64 array or float 32 array
        Can be either input filter or output (filtered) signal.
    
    Returns
    -------
    None.
    
    '''
    # Default option to visualize signal
    flag = input("Visualize results [Y/n] >> ").lower() or 'y'
    
    # Check if user input is valid
    while flag != 'y' and flag != 'n':
        flag = input("Visualize results [Y/n] >> ").lower()
    
    # User selected visualization
    if flag == 'y' and signal is not None:
        
        # Determine signal's length in seconds
        lenght = signal.shape[0] / sample_rate
        time = np.linspace(0., lenght, signal.shape[0])
        
        # Check number of audio channels
        try:
            channels = signal.shape[1]
            print("Number of audio channels present: %" %channels)
            # Loop & plot each signal's channel
            for i in range(channels):
                plt.plot(time, signal[:, i], label="channel: %d" %channels)
        except IndexError:
            channels = 1
            print("Audio file has one audio channel.")
            plt.plot(time, signal, label="channel: %d" %channels)
        
        plt.legend()
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.show()
    # No visualization asked
    else:
        print("User opted for no visualization or no input provided.")



def audio_read():
    '''
    Reads .wav audio file & returns the sample rate and  audio signals of each
    channel present
    
    Returns
    -------
    sample_rate : int
        Sample rate of entered .wav file
    
    signal : float32
        Array containing each audio channel's values
    
    read_path : string
        Returns the path of the .wav file
    
    '''
    # Dialogue Box to select file containing signal or save filtered signal
    def get_path(wildcard):
        '''
        Function that creates a window to select the appropriate file
        
        Parameters
        ----------
        wildcard : string
            Determines the file extension for the window to look.
        
        Returns
        -------
        path : string
            Path of the selected file (including the file itself).
        
        '''
        wx.App(None)
        style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
        dialog = wx.FileDialog(None, 'Open', wildcard=wildcard, style=style)
        if dialog.ShowModal() == wx.ID_OK:
            path = dialog.GetPath()
        else:
            path = None
        dialog.Destroy()
        return path
    
    # Run OS specific operations
    # if sys.platform.startswith('linux'):
    # Initialize window wildcard
    read_path = get_path('*.wav')
    # Sanity check for path selection
    while read_path is None:
        print("Invalid path. Select path.")
        read_path = get_path('*.wav')
    
    # Read audio file
    Fs, signal = wavfile.read(read_path)
    
    # Process read path to remove filename & avoid potential conflicts
    read_path = os.path.abspath(os.path.join(read_path, os.pardir))
    
    # Return signal & sample rate
    return Fs, signal, read_path

def audio_write(inpt, sample_rate, read_path):
    # TODO: file write sanity checks
    '''
    Function that writes the filtered audio signal(s) into a .wav file.
    If user does not define filename, a default name will be utilized. File
    is outputed in the same directory as the input .wav file.
    
    Parameters
    ----------
    inpt : float 32
        Filtered audio signal array.
    
    sample_rate : int
        Sample rate of input .wav file for consistent results/
    
    read_path : string
        Output path; same as the input .wav file.
    
    Returns
    -------
    None.
    
    '''
    # Select path to save file
    write_path = read_path
    
    # Ask user for predefined output filename or user-defined
    name_flag = input("Use default output name (*_filtered.wav) ? [Y/n] (q to quit) >> ").lower() or 'y'
    
    # Sanity check for user input
    while name_flag != 'y' and name_flag != 'n' and name_flag != 'q':
        name_flag = input("Use default output name (*_filtered.wav) ? [Y/n] (q to quit) >> ").lower()
    
    # Default output name
    if name_flag == 'y':
        filename = os.path.join(write_path, (read_path + "_filtered.wav"))
        
    # User doesn't request output file
    elif name_flag == 'q':
        # Print appropriate message
        print("User opted for no file output.")
        print("Exiting..")
        
    # Ask user for output name
    else:
        filename = input("Enter output filename: ") + ".wav"
        filename = os.path.join(write_path, filename)
        
        # Write audio file to designated directory
        wavfile.write(filename, sample_rate, inpt)
        
        # Print success message
        print("File creationg has been successful.")



def lms(x_sig, d_sig):
    '''
    
    
    Parameters
    ----------
    x_sig : TYPE
        DESCRIPTION.
    d_sig : TYPE
        DESCRIPTION.
    
    Returns
    -------
    filt_out : TYPE
        DESCRIPTION.
    
    '''
    
    def adapt_filt(w, x_sig, filt_ord, sig_len):
        
        # Initialize filtered array
        filt_out = np.zeros_like(x_sig)
        
        # Perform filtering using adaptive coefficients
        for i in range(sig_len):
            # Loop though each coefficient
            for j in range(filt_ord):
                # Set temp element holder for filtered samples
                temp = 0
                
                # Evaulate legality of array position
                pos = i - j
                if pos >= 0:
                    # Convolve
                    temp += w[j] * x_sig[pos]
            # Output filtered sample
            filt_out[i] = temp
        
        # Return filtered signal
        return filt_out
    
    # Default filter order
    d_ord = 64
    
    # Ask user for flter order
    filt_ord = np.uint16(input("Filter order (default: 64) >> ") or d_ord)
    
    # Sanity check for user input
    while filt_ord < 1 and filt_ord%2 != 0:
        print("Value too low or order number is odd.")
        filt_ord = np.uint16(input("Filter order >> "))
    
    # Step size
    mu = 0.0005
    
    # Initialize weights
    w = np.zeros(filt_ord, dtype=np.float32)
    
    # Number of samples of input signal
    sig_len = len(x_sig)
    
    # Initialize estimated filtering
    # est_size = sig_len % filt_ord
    # est = np.zeros(np.uint8(sig_len / filt_ord) + est_size, dtype=np.float32)
    
    # Initialize error rate
    e = np.zeros((sig_len), dtype=np.float64)
    
    
    # LMS algorithm
    for i in range(filt_ord, sig_len):
        # BUG: 1st array element of input singal wil not be processed.
        # Get from last to first value of signal's window
        window_sig = x_sig[i:i-filt_ord:-1]
        
        # Filter with previous coefficients
        est = w * window_sig
        
        # Evaluate error rate
        e[i:i-filt_ord:-1] = d_sig[i:i-filt_ord:-1] - est
        
        # Estimate next coefficients
        w = w * 2 * mu * window_sig * e[i-filt_ord]
        
    # Filter input signal with adaptive coefficients
    # filt_out = adapt_filt(w, x_sig, filt_ord, sig_len)
    filt_out = np.convolve(x_sig, w)
    
    # Ask user for learning rate visualization
    plot_flg = input("Visualize learning rate? [Y/n] >> ").lower() or 'y'
    
    # Sanity check for user input
    if plot_flg != 'y' and plot_flg != 'n':
        plot_flg = input("Visualize learning rate? [Y/n] >> ").lower()
    
    # User opted for visualization
    if plot_flg == 'y':
        # Plot error rate
        plt.plot(e)
        plt.xlabel('Epoch')
        plt.ylabel('Error rate')
        plt.show()
    
    # Return filtered signal
    return filt_out

def awgn_noise(x_sig):
    '''
    
    
    Parameters
    ----------
    x_sig : float32 array
        Input audio signal.
    
    Returns
    -------
    x_noised : float 32 array
        Input audio signal with AWGN.
    
    '''
    # Ask user if noising input signal is desired
    usr_flag = input("Infect input signal with AWGN? [Y/n] >> ").lower() or 'y'
    
    # Sanity check for user input
    while usr_flag != 'y' and usr_flag != 'n':
        usr_flag = input("Infect input signal with AWGN? [Y/n] >> ").lower()
    
    # User opted for no noise infection
    if usr_flag == 'n':
        return None
    # User opted for noise infection
    else:
        upscl_coeff = 400
        # Generate Gaussian noise corresponding to signal's available channels & upscale noise
        awgn = np.random.standard_normal(size=x_sig.shape) * upscl_coeff
        
        # Initialize noised signal 
        x_noised = np.zeros_like(x_sig)
        # Loop through each audio channel & add Gaussian noise
        try:
            # Evaluate number of available channels
            n_ch = x_sig.shape[1]
            for i in range(n_ch):
                x_noised[:, i] = x_sig[:, i] + awgn[:, i]
        # One audio channel present
        except IndexError:
            x_noised = x_sig + awgn
        
        # Return noised signal
        return x_noised

def main():
    '''
    
    
    Returns
    -------
    None.
    
    '''
    # Audio read function, returns signal sample rate & file path
    fs, in_signal, read_path = audio_read()
    
    # Plot input signal function
    plot_signal(in_signal, fs)
    
    # Simulate/generate noised signal
    noised_signal = awgn_noise(in_signal)
    
    # Plot noised signal
    plot_signal(noised_signal, fs)
    
    # Filter input signal
    filtered_signal = fir_filter(in_signal, fs)
    
    # Plot filtered signal
    plot_signal(filtered_signal, fs)
    
    # Adaptive filter via LMS algorithm
    lms_filt = lms(noised_signal, in_signal)
    
    # Plot LMS filtered signal
    # BUG: error rate is filtered signal & filtered signal is wrong
    plot_signal(lms_filt, fs)
    
    # Write filtered signal to file
    audio_write(filtered_signal, fs, read_path)

if __name__ == '__main__':
    main()
