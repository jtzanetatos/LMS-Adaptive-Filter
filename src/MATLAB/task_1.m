function [s, y, yff, freq_y, yff_h, freq_y_h, sff, freq_s, sff_h, freq_s_h,  Fs, samples, signal_duration] = task_1(audio_sample_start, audio_sample_end, filename, pathname)
%Task 1, produce single audio signal infected with random noise
%   Output Arguments:
%       signal: signal with AWGN
%       y: signal without noise       
%       Fn: 
%   Produce an audio signal, add to the input .wav file a random white
%   Gaussian noise. Firstly the input file is
%   read and then seperated by utilizing the first column of the y variable
%   as the left channel and the second column as the right channel. In
%   order to verify that the correct sample rate is implemented, the FFT
%   calculation is utilized. Afterwards, the time and freqency variables
%   are defined in order to plot the results of the calculations mentioned
%   above.

% Define Number of Samples to read
samples = [audio_sample_start, audio_sample_end];
[y, Fs] = audioread(fullfile(pathname, filename),samples); % Read Audio file and sampling rate

% Gather and define the duration of the audio signal in seconds
audio_info = audioinfo(fullfile(pathname, filename));
signal_duration = 0:seconds(1/Fs):seconds(audio_info.Duration);
signal_duration = signal_duration(1:end-1);

% Perform FFT on the original Signal, Normalize FFT values, Remove Mirrored
% Frequencies & Define Proper Frequency Values
yff = fft(y); % Perform FFT
yff = abs(yff).^2; % Normalize FFT values
yff_len = length(yff); % Number of frequencies present
freq_y = (0:yff_len-1)*Fs/yff_len; % Normalize frequency values
% Implement for the first half of the frequency spectrum
yff_h = yff(1:(yff_len/2)); % Show half of the FFT spectrum
freq_y_h =freq_y(1:(length(freq_y)/2)); % Show half of the Normalized frequency values

% Generate Random Gaussian White noise
r_noise = randn(size(y));

% Normalize and Add the generated Random Gaussian White noise to the signal
s = y + (r_noise./100);

% Perform FFT on the signal with the added Random Gaussian White noise,
% Normalize FFT values, Show Half of the present frequencies & Define Proper
% Frequency values
sff = fft(s); % Perform FFT
sff = abs(sff).^2; % Normalize FFT values
sff_len = length(sff); % Number of frequencies present
freq_s = (0:sff_len-1)*Fs/sff_len; % Normalize frequency values
% Implement for the first half of the frequency spectrum
sff_h = sff(1:(sff_len/2)); % Show half of the FFT spectrum
freq_s_h = freq_s (1:(length(freq_s)/2)); % Show half of the Normalized frequency values
end