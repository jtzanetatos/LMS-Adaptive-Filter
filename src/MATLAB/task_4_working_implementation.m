%% LMS implementation
clear all
close all
clc
%% LMS algorithm
% Can be implemented as a function in the general form of function
% [e,w]=lms(mu,M,u,d);
%       Input arguments:
%       mu      = step size, dim 1x1
%       M       = filter length, dim 1x1
%       u       = input signal, dim Nx1
%       d       = desired signal, dim Nx1
%
%       Output arguments:
%       e       = estimation error, dim Nx1
%       w       = final filter coefficients, dim Mx1

%% Import audio file & infect it with AWGN
[d, Fs] = audioread('OSR_us_000_0017_8k.wav');
r = randn(size(d));
u = d + (r./100);
%audiowrite('noise_out.wav', u, Fs)

%% LMS Implementation
% Assuming M = 30
M = 64;

% Assuming step size 
% mu = 0.00292213201441088;
mu = 0.005;

% Initialize values: 0
w = zeros(M,1);

% Number of samples of the input signal
N = length(d);

% Verify that u and d are column vectors
u=u(:);
d=d(:);

% LMS algorithm
for n = M:N
    uvec=u(n:-1:n-M+1); % Segment signal
    est(n) = w' * uvec;  % Multiply with previous coefficients
    e(n) = d(n) - est(n); % Calculate error
    w = w + 2*mu*uvec*e(n); % Estimate next coefficients
end
e = e(:); % Verify that is a column vector
e = e.^2; % Verify that values are possitive
de = filter(w,1,u); % Filter noised signal with adaptive coefficients
%% Plot results
% Plot learning rate
figure('Name', 'Learning Rate'); 
plot(e);title('Learning Rate');ylabel('Error rate');xlabel('Iterations');grid on; grid minor;

% Plot filtered signal on top of noised signal
% figure('Name', 'Noised signal overlayed by filtered signal')
% plot(u);title('Noised signal overlayed by filtered signal');ylabel('Amplitude');xlabel('Time');grid on; grid minor;
% hold on
% plot(de);

% Plot filtered signal on top of desired signal
% figure('Name', 'Desired signal overlayed by fitlered signal');
% plot(d);title('Desired signal overlayed by filtered signal');ylabel('Amplitude');xlabel('Time');grid on; grid minor;
% hold on
% plot(de)
%audiowrite('lms_out.wav', de, Fs) % Write results into an audio file
% 
% % Calculate FFT of Filtered signal
% ff = fft(de);
% xm = abs(ff).^2;
% ff_len = length(ff);
% freqHz = (0:1:ff_len-1)*Fs/ff_len;
% 
% % Plot Results
% figure('Name', 'Frequency Spectrum of filtered signal by LMS')
% plot(freqHz, xm);title('Frequency Spectrum of filtered signal by LMS');ylabel('Magnitude');xlabel('Frequency');grid on; grid minor;
% 
% % Calculate first half of FFT
% xm_h=xm(1:(ff_len/2));
% freqhz_h = freqHz(1:(length(freqHz)/2));
% 
% % Plot results
% figure('Name', 'First half of frequency Spectrum of filtered signal by LMS');
% plot(freqhz_h, xm_h);title('First half of frequency Spectrum of filtered signal by LMS');ylabel('Magnitude');xlabel('Frequency');grid on; grid minor;
% 
