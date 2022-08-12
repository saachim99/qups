% select some data to load in

cd /data/tbrevett/Projects/DopplerSandbox/

% all data
% listing = dir(fullfile(pwd, './data/kidney/P*/Acq*/RF/*_CFPD.mat'));
% select some data
% ind = 5; 

% get data and background data directories
switch hostname()
    case "diligent-doggo", 
        dlist = dir('./local-data/test/P*/Acq*/RF/*_CFPD.mat'); 
        bglist = dir('./local-data/test/noise/RF/*_CFPD.mat'); 
        ind = 1;
    otherwise
        dlist = dir('./data/test/P*/Acq*/RF/*_CFPD.mat');
        bglist = dir('./data/test/noise/RF/*_CFPD.mat');
        ind = 1;
end

% choose data files
dlist = dlist(ind);


% filter for data that is L12-3v
bgdat  = arrayfun(@(p) load(fullfile(p.folder, p.name), 'P'), bglist);
bgfilt = arrayfun(@(p) contains(p.P.configName, 'L12-3v'), bgdat);
bglist = bglist(bgfilt);
bgdat = bgdat(bgfilt);

% make a filter to match to the proper background file
Pflds = @(P) rmfield(P, setdiff(fieldnames(P), {'f0', 'na', 'totalAngle', 'rxStartMm', 'rxDepthMm'}));
bgP = cellfun(Pflds, {bgdat.P}); % fields that must match
bgmatch = @(P) arrayfun(@(bgP) isequal(Pflds(P), bgP), bgP); % check which background matches

% find matches for all files
dPlist = arrayfun(@(p) load(fullfile(p.folder, p.name), 'P'), dlist, 'UniformOutput',false);
bgind = cellfun(@(dPl) find(bgmatch(dPl.P)), dPlist);
blist = bglist(bgind);

%% get the QUPS objects
j = 1;
[chdb,   ~,   ~,    ~, db, memb] = load2QUPS(blist(j), 'ensembles', 1:256, 'frames', 1);
[chd0, xdc, seq, scan, d , mem ] = load2QUPS(dlist(j), 'ensembles', 1:256, 'frames', 1);
us = UltrasoundSystem('xdc', xdc, 'sequence', seq, 'scan', scan);

%% Setup / params
% params
c0 = d.P.speedOfSound; % sound speed (m/s)
prf = d.P.DopplerPRF; % PRF - pulse repitition frequency (Hz)

% choose apodization
fnum = 2;
apod = apertureGrowthApodization(us.scan, us.sequence, us.xdc, fnum);

%% beamform as many ensembles at a time as fits on the system
% define the beamformer
opts = {'interp', 'cubic', 'apod', apod};
preproc = @(chd) hilbert(singleT(gpuArray(chd)));
bmfrm = @(chd) {gather(DAS(us, preproc(chd), c0, opts{:}))}; 

% beamform data per ensemble
b0  = cell2mat(arrayfun(bmfrm, splice(chd0, 4, 64)));

% beamform background per ensemble
bg = cell2mat(arrayfun(bmfrm, splice(chdb, 4, 64)));

%% normalize the image by dividing by the TGC image (Code from Leo)

% TODO: we could try this in the channel dimension rather than in the image
% space
bgnoise = mean(abs(sum(bg, 3)), 4);
bgnoise = medfilt2(bgnoise, [5, 5]);
bgnoise = imgaussfilt(bgnoise, [3, 3]);
b = b0 ./ bgnoise;

%% display the image
b_im = mod2db(b); % convert to power in dB
% b_im = rad2deg(angle(b)); % get the phase
figure; him = imagesc(us.scan, b_im(:,:,1)); % display with 80dB dynamic range
colormap gray; colorbar; caxis([-80, 0] + max(b_im(:))); % power
% colormap hsv; colorbar; caxis('auto'); % phase

%% animate (over ensembles, and frames)
while(1)
for f = 1:size(b_im,5)
for e = 1:size(b_im,4)
if isvalid(him)
    him.CData(:) = b_im(:,:,1,e,f);
    title("Ensemble #" + e + ", Frame #" + f);
    drawnow limitrate;
    pause(1/prf); % real-time speed
end
end
end
end

%% SVD filter
% reduce the resolution
ds = 1; % downsample ratio
svdscan = copy(scan); 
svdscan.x = svdscan.x(1:ds:end);
svdscan.z = svdscan.z(1:ds:end);
bs = sub(b, {1:ds:size(b,1),1:ds:size(b,2)}, [1,2]);

% compute the SVD
bs = reshape(double(bs), [prod(svdscan.size), size(bs,4:ndims(bs))]); % move to I x E x F
tic; [u,s,v] = pagesvd(bs, "econ", "vector"); toc;
% TODO: 

% reconstruct like so:
% isalmostn((u .* s') * v', bs)
% isalmostn(u * (s  .* v'), bs)
% isalmostn(u * (s' .* v)', bs)

%% Define the eigen shift estimator
% kasai_scale = prf/pi*c0/xdc.fc/4; % phase -> speed (m/s)
kasai_scale = prf*c0/xdc.fc/(2*pi); % phase -> speed (m/s)
pdiff = @(v, k, dim) (sub(v, 1:size(v,dim)-k, dim) .* conj(sub(v, (1+k):size(v,dim), dim))); % phase diff function

%% Recreate the data, and estimate the velocity
w = 35 <= i & i <= 100; % binary selection weights
bse = u * (s .* w .* v'); % filtered ensemble data
bse = reshape(bse, [svdscan.size, size(b,4:ndims(b))]); % send back to original dimensions
% figure; imagesc(us.scan, mod2db(sum(bse,4))); colormap gray; colorbar; 
vel = kasai_scale * angle(sum(pdiff(bse, 1, 4), 4)); % get velocity image

%% Overlay on the b-mode image
figure; 
%%Create two axes
ax1 = axes;
him(1) = imagesc(scan, mod2db(b(:,:,1)), ax1);
ax2 = axes;
him(2) = imagesc(scan, 1e2*vel, ax2);

% set the alpha properties
bse_im = mod2db(sum(bse,4));
bse_filt = (bse_im > max(bse_im(:)) - 10);
him(2).AlphaData = 0.3 .* (abs(vel) > 3e-3) .* bse_filt;

%%Link them together
linkaxes([ax1,ax2])

%%Hide the top axes
ax2.Visible = 'off';
ax2.XTick = [];
ax2.YTick = [];

%%Give each one its own colormap
colormap(ax1,'gray')
colormap(ax2,'jet')

%%Then add colorbars and get everything lined up
% set([ax1,ax2],'Position',[.17 .11 .685 .815]);
cb1 = colorbar(ax1,'eastoutside');
cb2 = colorbar(ax2,'westoutside');

return;

%% Option2: estimate the velocity of the eigen-shifts
vel = kasai_scale .* sum(angle(pdiff(v, 1, 1)), 1); % get velocity in slow-time
% TODO: use the higher order differences by computing across multiple 
% frames, unwrapping the phases, and then getting the slope of the
% progression?

%% Sort by the eigen-shift velocities instead of eigen-values
[~, i] = sort(vel, 'ascend'); 
[up,sp,vp,velp] = deal(u(:,i), s(i), v(:,i), vel(i));

%% Create an image of the weighted average velocity (not how color flow works)
i = (1:numel(s))';
w = 9 <= i & i <= 100; % binary selection weights
wp = w & vel' > 0; % positive velocities
wm = w & vel' < 0; % negative velocities
bvel  = (reshape((abs(u) .* (s.*w )') * vel', svdscan.size));
bvelp = (reshape((abs(u) .* (s.*wp)') * vel', svdscan.size)) ./ sum(wp); 
bvelm =-(reshape((abs(u) .* (s.*wm)') * vel', svdscan.size)) ./ sum(wm); 
figure; imagesc(svdscan, bvelp); colormap jet; colorbar;

%% Display across the images
% move back to image dimensions
ui = reshape(u, [svdscan.size, size(b,4:ndims(b))]);

figure;
him = imagesc(svdscan, sub(mod2db(ui), {1,1}, [4,5])); 
colormap jet; colorbar;
caxis(max(mod2db(ui(:))) + [-40 0]);
for i = 1:size(ui(:,:,:),3), 
    if isvalid(him)
    him.CData(:) = mod2db(ui(:,:,i));
    title("Value #" + i); 
    pause(1/5); 
    else, break;
    end
end


%% Compute the phase shift across the ensembles


%% Compare to Leo's code as a reference
[bsvd, svdpar] = svd_filtering(b, 'P', d.P, 'TW', d.TW(1), 'method', 2, 'flagDisplay', true);



