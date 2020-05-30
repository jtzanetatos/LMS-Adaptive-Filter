'''
'''
import numpy as np
import sys
import wx
from scipy.io import wavfile
import os
import matplotlib.pyplot as plt

__author__ = "Iason Tzanetatos"
__version__ = "1.0.0"
__status__ = "Prototype" # "Prototype", "Development", "Product"

# TODO: fix AWGN function, fix convolution function, fix user input default states,
# Fix on main function how each function outputs interact with one antoher.
# Re-implement Kaiser coefficients.

def fir_filter(x):
    '''
    FIR 129th order Kaiser filter. Each element of the signal's array is entered
    into the 'Kaiser_filter_129' function & filtering is performed.
    
    Parameters
    ----------
    x : float 32 array
        Array containing the input (potentially) noised signal.
    
    Returns
    -------
    fir_out : float 64 array
        Array containing the filtered signal.
    
    '''
    def Kaiser_filter_129(x):
        '''
        Coefficients of 129th order Kaiser FIR filter, calculated utilizing MATLAB's
        filter design tool. The input element is multiplied in an iterative manner
        with each of the filter's coefficients.
        
        Parameters
        ----------
        x : float32 array element
            Single element from the input (potentially) noised signal.
        
        Returns
        -------
        y : float 64 array element
            Single filtered element from the input noised signal.
        '''
        # Define precalculated coefficients
        coeff = np.array([-0.0007948268903,-5.593943826e-19,-0.0008397484198,-1.180242081e-18,-0.0009317332297,
  8.140981308e-19, -0.00107358594,6.494930663e-19,-0.001267433865,3.233409206e-19,
  -0.001514673117,-3.787564899e-19,-0.001815926516,-1.332201363e-18,-0.002171012806,
  -3.284744759e-18,-0.002578927903,-6.218275661e-18,-0.003037839429,-1.062049035e-17,
  -0.003545094747,-1.712500773e-17,-0.004097240977,-3.874260872e-18,-0.004690059926,
  -1.036090122e-17,-0.005318613257,-1.895775376e-17,-0.005977303721,-3.117163302e-17,
  -0.006659940816,-4.716072324e-17,-0.007359825075,-6.836691603e-18,-0.008069834672,
  -2.214793461e-17,-0.008782522753,-4.299807909e-17, -0.00949022267,3.498921052e-17,
    -0.0101851495,1.790501423e-17, -0.01085952017,-6.066208845e-18, -0.01150565501,
  -4.168967126e-17, -0.01211609505,-8.695364951e-17, -0.01268371195,-2.599651979e-17,
   -0.01320180576,6.315700001e-17, -0.01366421115,8.923559928e-18,   -0.014065383,
  -7.716252436e-17, -0.01440048218,-6.250780881e-17, -0.01466544718,-3.167477721e-17,
   -0.01485705283,4.404970689e-17, -0.01497296151,-2.602681611e-18,   0.9857720137,
  -2.602681611e-18, -0.01497296151,4.404970689e-17, -0.01485705283,-3.167477721e-17,
   -0.01466544718,-6.250780881e-17, -0.01440048218,-7.716252436e-17,   -0.014065383,
  8.923559928e-18, -0.01366421115,6.315700001e-17, -0.01320180576,-2.599651979e-17,
   -0.01268371195,-8.695364951e-17, -0.01211609505,-4.168967126e-17, -0.01150565501,
  -6.066208845e-18, -0.01085952017,1.790501423e-17,  -0.0101851495,3.498921052e-17,
   -0.00949022267,-4.299807909e-17,-0.008782522753,-2.214793461e-17,-0.008069834672,
  -6.836691603e-18,-0.007359825075,-4.716072324e-17,-0.006659940816,-3.117163302e-17,
  -0.005977303721,-1.895775376e-17,-0.005318613257,-1.036090122e-17,-0.004690059926,
  -3.874260872e-18,-0.004097240977,-1.712500773e-17,-0.003545094747,-1.062049035e-17,
  -0.003037839429,-6.218275661e-18,-0.002578927903,-3.284744759e-18,-0.002171012806,
  -1.332201363e-18,-0.001815926516,-3.787564899e-19,-0.001514673117,3.233409206e-19,
  -0.001267433865,6.494930663e-19, -0.00107358594,8.140981308e-19,-0.0009317332297,
  -1.180242081e-18,-0.0008397484198,-5.593943826e-19,-0.0007948268903], dtype=np.float64)
        
        # Coefficient length constant
        coeff_len = 129
        
        # Initialize output
        y = np.float64(0)
        
        # Filter input sample
        for i in range(0, coeff_len, 2):
            y += x * coeff[i]
        
        # Return result of filtering
        return y
    # Ask user if deterministic filtering is desired
    filt_flg = 'y'
    
    filt_flg = input("Filter input signal by deterministic means? [Y/n] >> ").lower()
    
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
                # Filter each signal's samples
                for i in range(len(x)):
                    fir_out[i, ch] = Kaiser_filter_129(x[i, ch])
        except IndexError:
            print("Audio file has one audio channel.")
            # Filter each signal's samples
            for i in range(len(x)):
                fir_out[i] = Kaiser_filter_129(x[i])
        
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
    flag = 'y'
    flag = input("Visualize results [Y/n] >> ").lower()
    
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
    sample_rate, signal = wavfile.read(read_path)
    
    # Process read path to remove filename & avoid potential conflicts
    read_path = os.path.abspath(os.path.join(read_path, os.pardir))
    
    # Return signal & sample rate
    return sample_rate, signal, read_path

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
    name_flag = 'y'
    name_flag = input("Use default output name (*_filtered.wav) ? [Y/n] (q to quit) >> ").lower()
    
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
    filt_ord = 64
    
    # Ask user for flter order
    filt_ord = np.uint16(input("Filter order (default: 64) >> "))
    
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
    plot_flg = 'y'
    plot_flg = input("Visualize learning rate? [Y/n] >> ").lower()
    
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
    usr_flag = 'y'
    usr_flag = input("Infect input signal with AWGN? [Y/n] >> ").lower()
    
    # Sanity check for user input
    while usr_flag != 'y' and usr_flag != 'n':
        usr_flag = input("Infect input signal with AWGN? [Y/n] >> ").lower()
    
    # User opted for no noise infection
    if usr_flag == 'n':
        return None
    # User opted for noise infection
    else:
        # Generate Gaussian noise corresponding to signal's available channels
        awgn = np.random.normal(0, 1, size=(x_sig.shape))
        
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
    sample_rate, in_signal, read_path = audio_read()
    
    # Plot input signal function
    plot_signal(in_signal, sample_rate)
    
    # Simulate/generate noised signal
    noised_signal = awgn_noise(in_signal)
    
    # Plot noised signal
    plot_signal(noised_signal, sample_rate)
    
    # Filter input signal
    filtered_signal = fir_filter(in_signal)
    
    # Plot filtered signal
    plot_signal(filtered_signal, sample_rate)
    
    # Adaptive filter via LMS algorithm
    lms_filt = lms(noised_signal, in_signal)
    
    # Plot LMS filtered signal
    plot_signal(lms_filt, sample_rate)
    
    # Write filtered signal to file
    audio_write(filtered_signal, sample_rate, read_path)

if __name__ == '__main__':
    main()
