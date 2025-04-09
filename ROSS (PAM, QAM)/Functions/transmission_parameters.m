function [p] = transmission_parameters(L,D,lamda,a)

p.Spans=1; %that means that we take into account Spans*L total transmission length
p.h=6.626070040e-34;
p.c=299792458;
p.l=1550e-9;
p.fo=p.c/p.l;
p.L=L; %από 10 έως 80 km
p.D=D; %2 (o-band), 16 (c-band)
p.Dc=0.03*1e3; %%ps/nm^2/km
p.lamda=lamda;
p.lamda2=p.lamda+0.002e-6;
p.b2=-p.lamda^2/2/pi/p.c*p.D;
p.b3=p.Dc*p.lamda2^3*(3*p.lamda2-2*p.lamda)/((2*pi*p.c)^2);
p.b4=-3*p.Dc*p.lamda2^5/((2*pi*p.c)^3)-3*p.Dc*p.lamda2^4*(3*p.lamda2-2*p.lamda)/((2*pi*p.c)^3);
%b2=-18e-27; %%%%%%%%%%% second order dispersion in s^2/m 
p.g=1.3e-3; %%%%%%%%%%%%%%%% nonlinear parameter in 1/W/m, you can put it to zero in the beginning
p.a=a/4.343/1000; %%%%%%%%%%%% losses in m^-1 0.34 dB for O-band 
p.dz=p.L/40; %%%%%%%%%%%% simulation step for the NLSE method
p.Nz=p.L/p.dz; %%%%%%%%%%% number of steps


end

