# !/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.io import wavfile
import wx
import os
import sys

__author__ = "Iason Tzanetatos"
__version__ = "1.0.0"
__status__ = "Prototype" # "Prototype", "Development", "Product"


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
    if read_path is None:
        sys.exit("User did not select a file. Exiting..")
    
    # Read audio file
    Fs, signal = wavfile.read(read_path)
    
    # Process read path to remove filename & avoid potential conflicts
    read_path = os.path.abspath(os.path.join(read_path, os.pardir))
    
    # Return signal & sample rate
    return Fs, signal, read_path

def plotSignal(signal, sample_rate):
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
            print("Number of audio channels present: %d" % channels)
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

def plotLR(history):
    '''
    
    
    Parameters
    ----------
    history : TYPE
        DESCRIPTION.
        
    Returns
    -------
    None.
    
    '''
    # Ask user to visualize Learning Rate
    usr_flag = input("Visualize Learning Rate? [Y/n] >> ").lower() or 'y'
    
    while usr_flag != 'y' and usr_flag != 'n':
        usr_flag = input("Infect input signal with AWGN? [Y/n] >> ").lower()
    
    # User opted for no visualization
    if usr_flag == 'n':
        return
    # Visualize LR
    else:
        # Get Loss of model
        loss=history.history['loss']
        
        # Evaluate the number of epochs
        epochs=range(len(loss))
        
        # Plot Learning Rate
        plt.plot(epochs, loss)
        plt.title('Training loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(['Loss'])
        
        # Show plot
        plt.show()

def NetPredict(model, noised_signal, window_size, batch_size):
    '''
    
    
    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    noised_signal : TYPE
        DESCRIPTION.
    window_size : TYPE
        DESCRIPTION.
    batch_size : TYPE
        DESCRIPTION.
    
    Returns
    -------
    filtered_signal : TYPE
        DESCRIPTION.
    
    '''
    
    noised_signal = tf.expand_dims(noised_signal, axis=1)
    
    b_pred = tf.data.Dataset.from_tensor_slices(noised_signal)
    b_pred = b_pred.window(window_size + 1, shift=1, drop_remainder=True)
    b_pred = b_pred.flat_map(lambda w: w.batch(window_size + 1))
    b_pred = b_pred.batch(batch_size).prefetch(1)
    
    filtered_signal = model.predict(b_pred)
    
    return filtered_signal

def NeuralNetTrain(in_signal):
    '''
    
    
    Parameters
    ----------
    in_signal : TYPE
        DESCRIPTION.
    
    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    
    '''
    
    # Assert input signal's dimensions (mono channel/multichannel.)
    try:
        nChnls = in_signal.shape[1]
        
        # Normalize input signal
        norm_sig = np.zeros_like(in_signal, dtype=np.float32)
        
        # Loop thourgh each available channel
        for i in range(nChnls):
            norm_sig[:, i] = (in_signal[:,i] - np.min(in_signal[:,i])) / \
                             (np.max(in_signal[:, i]) - np.min(in_signal[:, i]))
    # Mono channel case
    except IndexError:
        nChnls = 1
        
        # Normalize input signal
        norm_sig = (in_signal - np.min(in_signal)) / \
            (np.max(in_signal) - np.min(in_signal))
        
    
    # Ask user to provide batch size & window size for input signal's segmentation
    batch_size = int(input("Enter batch size (default [64]) >> "))
    window_size = int(input("Enter window size (default [32]) >> "))
    
    # Verify validity of user inputs
    while True:
        if batch_size <= 0:
            print("Invalid batch size.")
            batch_size = int(input("Enter batch size (default [64]) >> "))
        elif window_size <= 0:
            print("Invalid window size.")
            window_size = int(input("Enter window size (default [32]) >> "))
        elif window_size <= 0 and batch_size <= 0:
            print("Invalid batch & window size.")
            batch_size = int(input("Enter batch size (default [64]) >> "))
            window_size = int(input("Enter window size (default [32]) >> "))
        else:
            break
    
    def batchSignal(in_singal, batch_size=64, window_size=32):
        
        # Experimental - Expand dimensions of input signal
        b_signal = tf.expand_dims(in_signal, axis=1)
        b_signal = tf.data.Dataset.from_tensor_slices(b_signal)
        b_signal = b_signal.window(window_size +1, shift=1, drop_remainder=True)
        b_signal = b_signal.flat_map(lambda w: w.batch(window_size+1))
        b_signal = b_signal.map(lambda w: (w[:-1], w[1:]))
        
        return b_signal.batch(batch_size).prefetch(1)
    
    # Batch input signal
    b_signal = batchSignal(norm_sig, batch_size, window_size)
    
    # TODO: Current model has extremely poor performance; time consuming to 
    # randomly add/remove layers. Further studying required.
    # Define a Neural Network model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                                strides=1, padding='causal',
                                activation='relu',
                                input_shape=[None, nChnls]),
        # tf.keras.layers.Conv1D(filters=32, kernel_size=5,
        #                        strides=1, padding='causal',
        #                        activation='relu'),
        # tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
        ])
    
    # Utilize Stochastic Gradient Descent optimizer
    optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
    
    # Compile mode
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=[tf.keras.metrics.MeanSquaredError()])
    
    # Train model on the input signal
    history = model.fit(b_signal, epochs=500)
    
    return model, history, window_size, batch_size

def main():
    # Audio read function, returns signal sample rate & file path
    fs, in_signal, read_path = audio_read()
    
    # Plot input signal
    plotSignal(in_signal, fs)
    
    # Simulate/generate noised signal
    noised_signal = awgn_noise(in_signal)
    
    # Plot noised signal
    plotSignal(noised_signal, fs)
    
    # Adaptive filter via LMS algorithm
    model, history, window_size, batch_size = NeuralNetTrain(in_signal)
    
    # Plot Neural Network's Learning Rate
    plotLR(history)
    
    # Filter Noised Signal
    filtered_signal = NetPredict(model, noised_signal, window_size, batch_size)
    
    # Plot filtered signal
    plotSignal(filtered_signal, fs)
    
    # Write filtered signal to file
    audio_write(filtered_signal, fs, read_path)

if __name__ == '__main__':
    # Must call this to make cuddn work with convolution layers
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
    main()