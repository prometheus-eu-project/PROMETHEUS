function [B,H]=MZM(A, samples,pulse_duration,N,fc,f0, Tdelay_in, Tdelay2, losa, losb,phase,L);

dt=pulse_duration/samples;			
j=-(samples/2):1:(samples/2)-1;
f=j*(1/samples);
f=f/dt;

H=0.5*(1+exp(-i*2*pi.*(f-f0)./(fc)));
% H=Hpar.^(-1);

A=fft(A);
A=fftshift(A);

%%Recurrent Node Equation
% A=H.*A*(losa)./(1+(losb*losa)*H.*exp(-i*(2*pi*f*Tdelay2+phase))).*exp(-i*2*pi*f*Tdelay_in);
A=H.*A*sqrt(1-losa.^2).*sqrt(1-losb.^2)./(1+(losa*losb*sqrt(L)).*H.*exp(-i*(2*pi*f*Tdelay2+phase))).*exp(-i*2*pi*f*(Tdelay_in));

A=ifftshift(A);
A=ifft(A);

B=A;