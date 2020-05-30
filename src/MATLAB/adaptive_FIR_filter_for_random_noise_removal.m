%% Adaptive FIR filter for random noise removal from recorded audio signal
clear variables
close all
clc
%% User Input parameters

% Import the desired audio file
[filename,pathname] = uigetfile('OSR_us_000_0017_8k.wav','Select the audio file (Default:OSR_us_000_0017_8k)');

% Dialog for used defined inputs such as sample rate
prompt = {'Enter first sample to read (sec):',...
    'Final Audio sample to read: (Enter "inf" to select all the available samples)',...
    'Enter the name of the output audio file: (entering zero will result in error)',...
    'Visualizesignal (1 = yes / 0 = no)',...
    'Choose between utilizing fir1 or fir2 (Enter 1 for fir1, 2 for fir2)',...
    'Enter the order of the FIR filter (default 20th order)',...
    'Enter the lowest frequency for the rectangular window (in Hz)',...
    'Enter the highest frequency for the rectangular window (in Hz)',...
    'Enter window size for Auto Correlation Function (ACF) (Default 100)'};
dlg_title = 'Input';
num_lines = 1;
defaultans = {'1', 'inf', '_out.flac','1','1', '14', '60', '7940', '100'};
answer = inputdlg(prompt,dlg_title,num_lines,defaultans);

% Useful to implement user defined sample rate as variable for later
% processing, the duration of the time space, and the N point/length of the FFT
audio_sample_start = str2num(answer{1});
audio_sample_end = str2num(answer{2});
filename_out = answer{3};
visualize_flag = str2num(answer{4});
fir_flag = str2num(answer{5});
fir_order = str2num(answer{6});
f_low = str2num(answer{7}); 
f_high = str2num(answer{8});
window_size = str2num(answer{9});

%% Task 1 
[s, y, yff, freq_y, yff_h, freq_y_h, sff, freq_s, sff_h, freq_s_h,  Fs, samples, signal_duration] = task_1(audio_sample_start, audio_sample_end, filename, pathname);

% Write signal with added noise to an audio file for human interpretation
audiowrite('audio_with_noise.wav', s, Fs);

%% Task 2
[fir_out, fir1_w, f_ff_len, f_ff, freq_f, f_ff_h, freq_f_h] = task_2(Fs, s, fir_order, f_low, f_high);

% Write filtered signal to an audio file for human interpretation
audiowrite('filtered_audio.wav', fir_out, Fs);

%% Task 3
[acf_r, lags, a_ff, freq_a] = task_3(fir_out, window_size, Fs);

%% Task 4
[e, wn] = task_4(y, fir_out, fir_order);
% Filter noised signal
lms_out = filter(wn,1,fir_out);
% Write filtered signal to an audio file for human interpretation
audiowrite('lms_filtered_audio.wav', lms_out, Fs);

%% Plot results of Task 1
% Time domain plotting for Original Signal //FIGURE 1
figure('Name', 'Original Signal')
plot(signal_duration, y)
title('Waveform of input signal');ylabel('Amplitude');xlabel('Time');grid on;grid minor;

% Frequency Spectrum for Original Signal
figure('Name', 'Frequency Spectrum of signal')
plot(freq_y, yff);
title('FFT of input signal');ylabel('Magnitude (dB)');xlabel('Frequency Response(Hz)');grid on;grid minor;

% Plot first half of the frequency spectrum present in the original signal
figure('Name', 'First Half of the frequencies present in the original signal')
plot(freq_y_h, yff_h);ylabel('Magnitude (dB)');xlabel('Frequency Response(Hz)');grid on;grid minor;

% Time domain plotting for Signal with noise //FIGURE 2
figure('Name','Original signal overlayed with the noised signal')
plot(signal_duration, s)
ylabel('Amplitude');xlabel('Time');grid on;grid minor;title('Original signal overlayed with the noised signal');
hold on
plot(signal_duration, y)
hold off % Unnecessary  since hold on executes hold off automatically but for demo reasons is present

% Frequency Spectrum for Signal with Noise
figure('Name', 'Noised signal in time domain')
plot(signal_duration, s);
ylabel('Amplitude');xlabel('Time');grid on;grid minor;title('Noised signal in time domain');

figure('Name', 'Noised signal in frequency domain')
plot(freq_s, sff) % Normalized Magnitude values
title('FFT of signal with noise');ylabel('Magnitude (dB)');xlabel('Frequency Response (Hz)');grid on;grid minor;title('Noised signal in frequency domain');

% Plot first half of the frequency spectrum present in the noised signal
figure('Name', 'First Half of the frequencies present in the noised signal')
plot(freq_s_h, sff_h);ylabel('Magnitude (dB)');xlabel('Frequency Response(Hz)');grid on;grid minor;

%% Plot results of task 2
% Plot Filter Response //FIGURE 4
% In order to plot the correct filter the if loop bellow is utilized
figure('Name', 'FIR Filter Response')
freqz(fir1_w, 1, f_ff_len)
ylabel('Amplitude'); title('Filter Response'); xlabel('Time'); grid on;grid minor;

% Plot signal with AWGN combined with Filtered signal, & the PSD of
% filtered signal // FIGURE 5
figure('Name', 'Signal with AWGN & Filtered signal')
plot(signal_duration, s) % Plotted signal with noise and filtered signal
title('Filtered signal (RED) & signal with noise (Blue)');ylabel('Amplitude');xlabel('Time');grid on;grid minor;
hold on
plot(signal_duration, fir_out)
hold off % As above, unnecessary

% Plot frequency domain of filtered signal
figure('Name', 'Filtered signal in frequency domain')
plot(freq_f, f_ff);title('Frequency domain of filtered signal');ylabel('Magnitude');xlabel('Frequency (Hz)');grid on;grid minor;

figure('Name', 'First half of frequency domain');
plot(freq_f_h, f_ff_h);title('First half of frequency domain of filtered signal');ylabel('Magnitude');xlabel('Frequency (Hz)');grid on;grid minor;

%% Plot Task 3
figure('Name', 'Response of ACF') % // FIGURE 6
% Based on the results it seems like ARMA
stem(acf_r);grid on;grid minor;

% Plot PSD of autocorrelated signal // FIGURE 7
figure('Name', 'FFT plot of autocorrelated signal')
plot(freq_a, a_ff);title('PSD of autocorrelated signal');ylabel('Magnitude');xlabel('Frequency (Hz)');grid on;grid minor;   

%% Plot Task 4
% Plot learning rate // FIGURE 8
figure('Name', 'Learning Rate')
plot(e)
title('Learning rate of LMS algorithm');ylabel('Error rate');xlabel('Iteration');grid on;grid minor;
    
% Plot filtered signal from LMS algorithm // FIGURE 9
figure('Name', 'Filtered signal using LMS Algorithm')
plot(signal_duration, lms_out)
title('Filtered signal using LMS Algorithm');ylabel('Amplitude');xlabel('Time (in sec)');grid on;grid minor;

figure('Name', 'Desired Signal overlayed by LMS output')
plot(signal_duration, y);title('Desired Signal overlayed by LMS output');ylabel('Amplitude');xlabel('Time (in sec)');grid on;grid minor;
hold on
plot(signal_duration, lms_out);
hold off