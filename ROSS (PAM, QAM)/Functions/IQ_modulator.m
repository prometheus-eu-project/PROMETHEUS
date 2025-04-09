function [Axin,signal_power,Diff]=IQ_modulator(bias,PP,Einj_i,PowmW,samples_symbol,dt,data,bandwidth)


mod_data=data; %modulated data

% Initially A = sqrt(Pow)
A=Einj_i*sqrt(PowmW)/sqrt(mean(abs(Einj_i.^2)));

%oversampling of modulation to analog sampling
for kk=1:length(mod_data)
    Resampled(samples_symbol*(kk-1)+1:samples_symbol*kk)=mod_data(kk);
end

%filtering of analogue signal
Resampled_x=real(butterworth(real(Resampled),length(Resampled),length(Resampled).*dt,4,bandwidth,0));
Resampled_y=real(butterworth(imag(Resampled),length(Resampled),length(Resampled).*dt,4,bandwidth,0));
Resampled=Resampled_x+1i.*Resampled_y;


Diff=length(A)-length(Resampled);


dataI=(Resampled_x)/max(Resampled_x)+1;
dataQ=(Resampled_y)/max(Resampled_y)+1;

% Vpi=6;
% Swing=0.08*Vpi;          %% driving voltage <0.5*Vpi
% I=Swing*(dataI);
% Q=Swing*(dataQ);
% bias=pi+pi/4;            %% bias in the quad point
% outI=1+cos(bias+I*pi/Vpi);
% outQ=1+cos(bias+Q*pi/Vpi);

Vpi=6;
Swing=PP*Vpi;          %% driving voltage <0.3*Vpi
I=Swing*(dataI);
Q=Swing*(dataQ);
outI=1+cos(bias+I*pi/(2*Vpi));
outQ=1+cos(bias+Q*pi/(2*Vpi));

Out=outI+outQ*exp(i*(pi/2));

% creating the field
A(Diff/2:length(Out)+Diff/2-1)=A(Diff/2:length(Out)+Diff/2-1).*Out;

% scatterplot(Out);
signal_power=10*log10(mean(abs(A).^2)*1000);

Axin=A;