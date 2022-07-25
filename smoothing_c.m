function [bigarray] = smoothing_c(arraycz,n,w)
%SMOOTHING FUNCTION Summary of this function goes here
%   Detailed explanation goes here
numc = length(arraycz);
disp(numc);
disp(n);
alength = ((numc-1)*n)+numc;
disp(alength);
bigarray = zeros(alength,2);
index = 1;
for i = 1:numc-1
    c1 = arraycz(i,1);
    z1 = arraycz(i,2);
    c2 = arraycz(i+1,1);
    z2 = arraycz(i+1,2);
    cs = linspace(c1,c2,n+2);
    %disp(cs)
    %disp(z1);
    %zs = linspace(z1,z2,n);
    assert((z1 - ceil(n/2)*w) > 0,'whoops');
    assert((z2 - floor(n/2)*w) > 0,'oops');
    z1 = z1 - ceil(n/2)*w;
    %disp(z1);
    z2 = z2 + floor(n/2)*w;
       % disp(z2);

    zn = z1;
    
    bigarray(index,:) = [c1,z1];
    index = index+1;
    %for j = n:-1:1
    for k = 1:n
        %if (j+k==n+1)
        zn = zn+w;
        %disp(zn);
        %disp(index);
        bigarray(index,:)=[cs(k+1), zn];
        index = index+1;
    end
end
bigarray(alength,:) = [c2,z2];
zn=0;

end








%{
        steps"
        take in c and z array, and number of steps for interpolation
        divide num steps by 2, take floor and ceiling for even and odd
        change the distances of the original array behind and forward by
        amount
        insert those
%}

