function [Axin,PdBm,mod_data,Diff]=PAM_transmitter(M,symbols,Einj_i,PowmW,samples_symbol,dt,bandwidth)


rng('shuffle');
data=randi([0,M-1],1,symbols); 



mod_data=pammod(data,M)+1*M; %modulated data
meanP=mean(abs(mod_data).^2);

% Initially A = sqrt(Pow)
A=Einj_i*sqrt(PowmW)/sqrt(mean(abs(Einj_i.^2)));

%oversampling of modulation to analog sampling
for kk=1:length(mod_data)
    Resampled(samples_symbol*(kk-1)+1:samples_symbol*kk)=mod_data(kk);
end

Resampled=real(butterworth(real(Resampled),length(Resampled),length(Resampled).*dt,4,bandwidth,0));

Diff=length(A)-length(Resampled);

A(Diff/2:length(Resampled)+Diff/2-1)=1*A(Diff/2:length(Resampled)+Diff/2-1).*sqrt(Resampled);

Power=mean(abs(A(Diff/2:length(Resampled)+Diff/2-1)).^2); % print the actual mean power after modulation, this is the power you should include in your x axis. 
PdBm=10*log10(Power/1e-3)


Axin=A;