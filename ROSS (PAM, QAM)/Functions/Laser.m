function [Einj_i,Linewidth]=Laser(Nt,dt,aj,bb)
%%%%%%%%%%%%%% laser 

gl=1.2e4;
s=5e-7;
tph=2e-12;
tn=2e-9;
N0=1.5e8;    
q=1.6e-19;
NN=zeros(1, Nt);
E=zeros(1, Nt);
rng('shuffle')


%% rate equation modelling
Ib=50e-3; %% Bias Current

for nt=1:Nt-1
    GG=gl*(NN(nt)-N0)/(1+s*abs(E(nt)).^2);
    ae=(1+i*aj)/2.0*(GG-1/tph);
    %be=add_noise(sqrt(2*bb*NN(nt)/dt), 0)+i*add_noise(sqrt(2*bb*NN(nt)/dt), 0);
    be=sqrt(4*NN(nt)*bb/dt)*(randn(1)+1i*randn(1));%*kinjM';
    an=-1/tn-gl*abs(E(nt)).^2/(1+s*abs(E(nt)).^2);
    bn=Ib/q+gl*N0*abs(E(nt)).^2/(1+s*abs(E(nt)).^2);
    
    E(nt+1)=E(nt)*exp(ae*dt)+be/ae*(exp(ae*dt)-1);
    NN(nt+1)=NN(nt)*exp(an*dt)+bn/an*(exp(an*dt)-1);
end

Linewidth=4*bb*mean(NN)*(1+aj^2)/4/pi/mean(abs(E).^2);
clear NN;
%%%%%%%%%%%%% transmission data%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mean(abs(E).^2);

% E=E.*exp(-i*angle(E));
phi_n=angle(E);
% plot(angle(E));
Einj_i=E;
