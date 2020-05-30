function [acf_r, lags, a_ff, freq_a] = task_3(fir_out, window_size, Fs)
%Task 3 Perform Autocorrelation on filtered signal and calculate its PSD

[acf_r, lags] = xcorr(fir_out,fir_out, window_size);

% Perform FFT on the autocorrelated signal

a_ff = fft(acf_r); % Perform FFT
a_ff = abs(a_ff).^2; % Normalize FFT values
a_ff_len = length(a_ff);
freq_a = (0:a_ff_len-1)*Fs/a_ff_len;

end