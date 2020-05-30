function [fir_out, fir1_w, f_ff_len, f_ff, freq_f, f_ff_h, freq_f_h] = task_2(Fs, s, fir_order, f_low, f_high)
%Task 2. Implement FIR bandpass filter

% Define bandpass window
Wn = [(f_low./Fs), (f_high./Fs)];

% Create a Nth-order FIR bandpass filter. The bandpass argument not
% necessairy but utilized to highlight the type of filter
fir1_w = fir1(fir_order, Wn, 'bandpass');

% Apply the filter to the signal with AWGN
fir_out = filter(fir1_w,1,s);

% Calculate the fft of the filtered signal in order to verify the filter
% response in the frequency domain
f_ff = fft(fir_out);

% Full Frequency Spectrum
f_ff = abs(f_ff).^2;
f_ff_len = length(f_ff);
freq_f = (0:f_ff_len-1)*Fs/f_ff_len;

% Half Frequency Spectrum
f_ff_h = f_ff(1:(f_ff_len/2));
freq_f_h = freq_f(1:(length(freq_f)/2));
end