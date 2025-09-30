function f = fftfreq(n, d)
    % FFTFREQ Computes the discrete Fourier Transform sample frequencies.
    %   f = FFTFREQ(n, d) returns the Discrete Fourier Transform sample
    %   frequencies for a signal of length n with sample spacing d.
    %
    %   Parameters:
    %     n: Integer, the length of the signal.
    %     d: Float, the sample spacing (inverse of the sampling rate).
    %
    %   Returns:
    %     f: Array of length n containing the sample frequencies.
    if nargin<2
        d = 1;
    end
    % Calculate the frequency bins
    if mod(n, 2) == 0
        % If n is even
        f = (-n/2:n/2-1) / (d*n);
    else
        % If n is odd
        f = (-(n-1)/2:(n-1)/2) / (d*n);
    end
    % Rearrange frequencies to match the output of numpy.fft.fftfreq
    f = ifftshift(f);
end