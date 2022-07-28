function [img,tx_perFreq_x_z] = wfc_cluster(f,delay, apod,xpos,rxdata_h,x,z,c_x_z, version, plot_updates)

            P_Rx_f = fftshift(fft(rxdata_h, nt, 1), 1);
            [~, F, ~] = meshgrid(1:size(rxdata_h,2), f, 1:size(rxdata_h,3)); 
            P_Rx_f = P_Rx_f .* exp(-1i*2*pi*F*time(1));
            disp(length(xpos));
            disp(size((P_Rx_f)));
            rxdata_f = interp1(xpos, permute(P_Rx_f, [2,1,3]), x, 'nearest', 0);

 % Only Keep Positive Frequencies within Transducer Passband
            passband_f_idx = ((f > 3e6) & (f < 12e6));
            rxdata_f = rxdata_f(:,passband_f_idx,:);
            P_Tx_f = ones(size(f)); f = f(passband_f_idx); 
            P_Tx_f = P_Tx_f(passband_f_idx); % Assume Flat Passband

            %{
% get the transmit delays and apodization
            % (M is the transmit aperture, V is each transmit)
            delay  = self.sequence.delays(self.tx); % M x V
            apod = self.sequence.apodization(self.tx); % M x V
            %disp(size(apod))
            %disp(size(xpos))
            %delay= self.sequence.dealays(self.tx);

            M = size(delay,1);
            V = size(delay,2);
            delay = reshape(delay,M,V,1);
            delay = permute(delay, [1,3,2]);
            
            MA = size(apod,1);
            VA = size(apod,2);
            apod = reshape(apod, MA,VA,1);
            apod = permute(apod, [1,3,2]);
            %}
            
            %delayre = reshape(delay, )
            %apod = permute(eye(no_rx_elements), [1,3,2]); % N x 1 x N
            %delay = permute(zeros(no_rx_elements), [1,3,2]); % N x 1 x N
            apod_x     = interp1(xpos, apod (:,:,tx_elmts), vec(x), 'nearest', 0); % X x 1 x M
            delayIdeal = interp1(xpos, delay(:,:,tx_elmts), vec(x), 'nearest', 0); % X x 1 x M
            txdata_f = (apod_x.*P_Tx_f).*exp(-1i*2*pi*delayIdeal.*f); % X x F x M
            txdata_f = cast(txdata_f, 'like', rxdata_f); % match the data type

            % shot gather mig
             disp('Migrating ...'); tic;
             
             img = shot_gather_mig_t(x, z, c_x_z, f, rxdata_f, txdata_f, aawin, version, plot_updates);
             
             
             %job = createJob(clu);
             %task = createTask(job, @shot_gather_mig_t,1,{x, z, c_x_z, f, rxdata_f, txdata_f, aawin, 'version', kwargs.version, 'plot', kwargs.plot_updates});
             %submit(job);
             
             disp('Done!_shotgather'); toc;
          
            % Generate Time-Gain Compensation From Center Frequency
            f_ctr_idx = round(numel(f)/2);
            
            % get the propagated energy into the medium (at the center frequency)
            disp('Migrating angular spectrum...'); tic;
            tx_perFreq_x_z = angular_spectrum_t(...
                x, z, c_x_z, f(f_ctr_idx), txdata_f(:,f_ctr_idx,:), aawin...
                ); % Z x X x 1 x S
            disp('Done! angular spectrum'); toc;

end