function [out,noise]=photodetector(out,dt,fo,length,t_total,bitrate,cancel_noise)
    rng('shuffle')
    kbol=1.38e-23; % Boltzmann constant
    TinK=300; % temperature in Kelvin
    Rload=50; % resistance in Ohm
    h=6.626070040e-34; % h Planck
    nQN=0.5*h*fo/dt; %%%%%%%%%% power spectral density of quantum noise - shot noise
    q=1.6e-19;
out = out + sqrt(nQN / 2) * (randn(size(out)) + 1i * randn(size(out)));

%% Non linearity and square law

R=0.9;
out=R*abs(out).^2;

%% thermal noise

nTH=cancel_noise*4*kbol*TinK/Rload/(dt);

out=out+sqrt(nTH)*randn(size(out));

out=butterworth(out, length, t_total, 4,  bitrate, 0); 
out=real(out);  %%DD output
noise=[sqrt(nTH) sqrt(nQN / 2)];