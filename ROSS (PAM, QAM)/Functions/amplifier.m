function A=amplifier(A, gain, fo,dt,add_noise )
 rng('shuffle')
    nsp=1.5; %%spontaneous emmission
    kbol=1.38e-23; % Boltzmann constant
    TinK=300; % temperature in Kelvin
    Rload=50; % resistance in Ohm
    h=6.626070040e-34; % h Planck
    nQN=1*0.5*h*fo/dt; %%%%%%%%%% power spectral density of quantum noise - shot noise
    
A=A*sqrt(gain);
% Ay=Ay*sqrt(G);
 
n=add_noise*h*fo*nsp*(gain-1)/dt; %%%%%%%%%% power spectral density of ASE from EDFA

A = A + sqrt(n / 2) * (randn(size(A)) + 1i * randn(size(A)));
% outy = outy + sqrt(n / 2) * (randn(size(outy)) + 1i * randn(size(outy)));
