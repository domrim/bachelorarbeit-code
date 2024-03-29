% This m-file is designed as a test of the splitstep routine.
% When it is working correctly, this should reproduce the results
% of Agrawal, Fig. 4.14.  Specifically, this routine simulates
% the case of pulse propagation at the zero-dispersion point, in
% the presence of nonlinearity.  This test case was chosen
% because both the output waveform and the output spectrum are
% asymmetric, which allows one to identify sign errors in the
% algorithms.

clear all;
addpath(genpath("/home/dominik/Downloads/ssprop-3.0.1-linux-i586/ssprop-3.0.1"));
load("vars.mat");

nt = length(u0);
T = nt*dt;
t = ((1:nt)'-(nt+1)/2)*dt;   % time vector
u0 = u0';
w = wspace(T,nt);            % angular frequency vector
vs = fftshift(w/(2*pi));     % frequency (shifted for plotting)
alphalin = alpha/(10/log(10));

tic;
u = ssprop(u0,dt,dz,nz,alphalin,betap,gamma, 1, inf);   % propagate signal
duration = toc;

save("output.mat");
