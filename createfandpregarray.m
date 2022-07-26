%arrayoffs
function [preg_layers]=createfandpregarray(smoothedarray,impsarray)

num = length(smoothedarray);
fns = {};
preg_layers = {};

for i =1:num-1
    %fns{i} = @(p) smoothedarray(i,2) <= sub(p,3,1) & sub(p,3,1)<=smoothedarray(i+1);
    temparray = [smoothedarray(i+1,1),round(impsarray(i+1,1)/smoothedarray(i+1,1))];
    preg_layers{i} = {@(p) smoothedarray(i,2) <= sub(p,3,1) & sub(p,3,1)<=smoothedarray(i+1),temparray};
end

end