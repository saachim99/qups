function [img] = shot_gather_mig_t(x, z, c_x_z, f, rxdata_f, txdata_f, aawin, kwargs)
arguments
    x
    z
    c_x_z
    f
    rxdata_f
    txdata_f
    aawin
    kwargs.version = 2
    kwargs.plot (1,1) logical = true
end

% Verify the Number of Common Shot Gathers
ns =  size(txdata_f, 3); 
%assert(size(rxdata_f, 3) == ns, ...
%    'Number of sources must equal to number of common-source gathers');
AAwin = cast(aawin(:), 'like', rxdata_f);

% Forward and Inverse Fourier Transforms with Anti-Aliasing Windows
ft = @(sig) fftshift(fft(AAwin.*sig, [], 1), 1);
ift = @(sig) AAwin.*ifft(ifftshift(sig, 1), [], 1);

% Spatial Grid
dx = mean(diff(x)); nx = numel(x); 
x = dx*((-(nx-1)/2):((nx-1)/2)); 
dz = mean(diff(z)); 

% FFT Axis for Lateral Spatial Frequency
kx = mod(fftshift((0:nx-1)/(dx*nx))+1/(2*dx), 1/dx)-1/(2*dx);

% Convert to Slowness [s/m]
s_x_z = 1./c_x_z; % Slowness = 1/(Speed of Sound)
s_z = mean(s_x_z, 2); % Mean Slowness vs Depth (z)
ds_x_z = s_x_z - s_z; % Slowness Deviation (residual)

% Construct Image by Looping Over Frequency
img = zeros(numel(z), numel(x));

% initialize Ultrasound Image
if kwargs.plot
hfig = figure;
hax = gca;
him = imagesc(hax, 1000*x, 1000*z, db(abs(img)/max(abs(img(:)))), [-80, 0]);
xlabel(hax, 'x Azimuthal Distance (mm)');
ylabel(hax, 'z Axial Distance (mm)');
title(hax, ['Image Reconstruction up to ', num2str(0/(1e6)), ' MHz']);
zoom on;
axis equal; axis xy; axis image;
colormap gray; colorbar();
xlim(hax, [-19.1,19.1]);
set(hax, 'YDir', 'reverse');
end

% case 1 is for reference, case 2 is accelerated
switch kwargs.version
    case 1
        % M = moviein(numel(f));

        for f_idx = 1:numel(f)
            % Setup Downward Continuation at this Frequency
            rx_surface = squeeze(permute(rxdata_f(:,f_idx,:), [2,1,3]));
            tx_surface = squeeze(permute(txdata_f(:,f_idx,:), [2,1,3]));
            rx_singleFreq_x_z = zeros(numel(z), numel(x), ns);
            tx_singleFreq_x_z = zeros(numel(z), numel(x), ns);
            rx_singleFreq_x_z(1, :, :) = rx_surface;
            tx_singleFreq_x_z(1, :, :) = tx_surface;

            % Continuous Wave Response By Downward Angular Spectrum
            for z_idx = 1:numel(z)-1
                % Create Propagation Filter for this Depth
                kz = csqrt((f(f_idx) .* s_z(z_idx)).^2 - kx.^2); % Axial Spatial Frequency
                H = vec(exp(1i*2*pi*kz.*dz)); % Propagation Filter in Spatial Frequency Domain
                H(kz.^2 <= 0) = 0; % Remove Evanescent Components
                % H = repmat(H(:), [1, ns]); % Replicate Across Shots

                % Create Phase-Shift Correction in Spatial Domain
                dH = vec(exp(1i*2*pi*f(f_idx)*ds_x_z(z_idx,:)*dz));
                % dH = repmat(dH(:), [1, ns]); % Replicate Across Shots

                % Downward Continuation with Split-Stepping
                rx_singleFreq_x_z(z_idx+1, :, :) = dH .* ...
                    ift(     H  .* ft(squeeze(rx_singleFreq_x_z(z_idx,:,:))));
                tx_singleFreq_x_z(z_idx+1, :, :) = conj(dH) .* ...
                    ift(conj(H) .* ft(squeeze(tx_singleFreq_x_z(z_idx,:,:))));
            end
            % Accumulate Image Frequency-by-Frequency
            img = img + sum(tx_singleFreq_x_z .* conj(rx_singleFreq_x_z), 3);

            if kwargs.plot, try %#ok<TRYNC,ALIGN> 
            % update and record image
            him.CData = db(abs(img)/max(abs(img(:))));
            caxis(hax, [-80, 0] + max(him.CData(:)))
            hax.Title.String = ['Image Reconstruction up to ', num2str(f(f_idx)/(1e6)), ' MHz'];
            drawnow limitrate;
            end, end
             M(f_idx) = getframe(hfig);
        end
        % Save Accumulation of Image in Frequency Domain
        %movie2gif(M, 'FreqDomain.gif');
    case 2
        

        % reshape as X x F x S x Z
        AAwin = aawin(:); % X x 1 x 1
        f = shiftdim(f(:),-1); % 1 x F
        kx = kx(:); % X x 1 x 1
        ds_x_z = permute(ds_x_z, [2, 1]); % X x Z x 1

        [AAwin, f, kx, ds_x_z] = dealfun(@(x) cast(x, 'like', rxdata_f), AAwin, f, kx, ds_x_z);

        % Forward and Inverse Fourier Transforms with Anti-Aliasing Windows
        fftdim = 1;
        ft = @(sig) fftshift(fft(AAwin.*sig, [], fftdim), fftdim);
        ift = @(sig) AAwin.*ifft(ifftshift(sig, fftdim), [], fftdim);

        % data at the surface % X x F x S
        p_rx_curr = rxdata_f;
        p_tx_curr = txdata_f;

        % init
        img = zeros(numel(z), numel(x));
        img(1,:) = sum(p_tx_curr .* conj(p_rx_curr), [2 3]);

        % Continuous Wave Response By Downward Angular Spectrum
        for z_idx = 1:numel(z)-1
            % Create Propagation Filter for this Depth
            kz = csqrt((f .* s_z(z_idx)).^2 - kx.^2); % Axial Spatial Frequency
            H = (exp(1i*2*pi*kz.*dz)); % Propagation Filter in Spatial Frequency Domain
            H(kz.^2 <= 0) = 0; % Remove Evanescent Components
            % H = repmat(H(:), [1, ns]); % Replicate Across Shots

            % Create Phase-Shift Correction in Spatial Domain
            dH = (exp(1i*2*pi*f.*ds_x_z(:,z_idx)*dz));
            % dH = repmat(dH(:), [1, ns]); % Replicate Across Shots

            % Downward Continuation with Split-Stepping 
            p_rx_curr =      dH  .* ift(     H  .* ft(p_rx_curr)); % X x F x S
            p_tx_curr = conj(dH) .* ift(conj(H) .* ft(p_tx_curr)); % X x F x S

            % accumulate the correlation over each frequency, shot
            img(z_idx+1,:) = sum(p_tx_curr .* conj(p_rx_curr), [2 3]);

            % update plot?
            if kwargs.plot
              %  disp('we are in plot')
            him.CData(z_idx+1,:) = db(abs(img(z_idx+1,:)));
            caxis(hax, [-80, 0] + max(him.CData(:)));
            hax.Title.String = string(['Image Reconstruction up to ', num2str(z(z_idx+1)*1e3), ' mm']);
            drawnow limitrate;
            %imagesc(hfig);
            
            M{z_idx} = getframe(hfig);
            N{z_idx} = M{z_idx}.cdata;
            end

        end
        % Accumulate Image Frequency-by-Frequency
        % img = img + sum(tx_singleFreq_x_z .* conj(rx_singleFreq_x_z), 3);

        % Save Accumulation of Image in Frequency Domain
        movie2gif(M, 'FreqDomain.gif');
       % movie2gif(N, 'FreqDomain_test.gif');
        %movie(M,1,'limitrate');
        %saveas(M);

end

if kwargs.plot, try close(hfig); end, end %#ok<TRYNC> 
