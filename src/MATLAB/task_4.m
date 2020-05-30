function [e, wn] = task_4(y, fir_out, fir_order)
%Task 4 Adaptive filtering using LMS

% Get input of desired signal, signal with noise & initial FIR
% coefficients
d = y(:);
x = fir_out(:);
w = zeros(fir_order,1);
% Define error rate variable
e = zeros(1, length(d));
% Pre-allocate adaptive filter coefficients
wn = zeros(fir_order, 1);
est = zeros(fir_order, 1);
% Assuming step size: 0.0004
mu = 0.0005;

% Determine the value of present samples
N = length(d);
M = fir_order;
% Loop through every sample of the signal
for n = M:N
    xN = x(n:-1:n-M+1); % Implement window
    est(n)= w'*xN; % Calculate estimated filter coefficients
    e(n) = d(n) - est(n); % Error rate
    wn = w + 2*mu*xN*e(n); % Caclulate filter coefficient for next sample
end
e = e(:); % Verify that variable is column vector
e = e.^2; % Verify that only positive values are present in order to plot learnig rate
end