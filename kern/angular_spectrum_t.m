function wf_x_z_f = angular_spectrum_t(x, z, c_x_z, f, tx_x_f, aawin)
arguments
    x 
    z 
    c_x_z 
    f (1,:)
    tx_x_f 
    aawin (:,1)
end

% get data size
nS = size(tx_x_f,3);

% Spatial Grid
dx = mean(diff(x)); nx = numel(x); 
x = dx*((-(nx-1)/2):((nx-1)/2)); 
dz = mean(diff(z)); 

% FFT Axis for Lateral Spatial Frequency
kx = mod(fftshift((0:nx-1)'/(dx*nx))+1/(2*dx), 1/dx)-1/(2*dx);

% Convert to Slowness [s/m]
s_x_z = 1./c_x_z.'; % Slowness = 1/(Speed of Sound)
s_z = mean(s_x_z, 1); % Mean Slowness vs Depth (z)
ds_x_z = s_x_z - s_z; % Slowness Deviation

% cast to same data type
[aawin, ds_x_z, f, kx] = dealfun(@(x) cast(x, 'like', tx_x_f), aawin, ds_x_z, f, kx);

% Forward and Inverse Fourier Transforms with Anti-Aliasing Windows
fftdim = 1; 
ft = @(sig) fftshift(fft(aawin.*sig, [], fftdim), fftdim);
ift = @(sig) aawin.*ifft(ifftshift(sig, fftdim), [], fftdim);

% Generate Wavefield at Each Frequency
wf_x_z_f = zeros([numel(x), numel(f), nS, numel(z)], 'like', gather(tx_x_f(1)));
wf_x_z_f(:,:,:,1) = tx_x_f; % Injection Surface (z = 0) % X x F x S x Z

% Continuous Wave Response By Downward Angular Spectrum
for z_idx = 1:numel(z)-1
        % Create Propagation Filter for this Depth
        kz = csqrt((f.*s_z(z_idx)).^2 - kx.^2); % Axial Spatial Frequency
        H = exp(1i*2*pi*kz*dz); % Propagation Filter in Spatial Frequency Domain
        H((f .* s_z(z_idx)).^2 - kx.^2 <= 0) = 0; % Remove Evanescent Components
        % Create Phase-Shift Correction in Spatial Domain
        dH = exp(1i*2*pi*f.*ds_x_z(:, z_idx)*dz); 
        % Downward Continuation with Split-Stepping
        wf_x_z_f(:, :, :, z_idx+1) = conj(dH).*ift(conj(H).*ft(wf_x_z_f(:, :, :, z_idx))); 
end

% set dimensions to Z x X x F x S
wf_x_z_f = permute(wf_x_z_f, [4,1,2,3]);