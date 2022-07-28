function [bimage] = wfc(us, chd, targ, tscan)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
bimage = bfWavefieldCorrelation(us, chd, targ, tscan, 'interp', 'cubic'); % use   

end