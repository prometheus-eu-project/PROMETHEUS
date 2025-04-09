close all;
clear all;

%% Simulation Setup
for uu=1:1:1
M=16;
PowdBm=10;
PowmW=1e-3*10.^(PowdBm/10)/M;

%%% sampling
Nt=2^21; %%%%%%%%%% number of samples (keep it over 2^21)
Symbol_rate=32e9; %%% symbol rate
samples_symbol=2^5; %%%% samples per symbol
symbols=ceil(Nt/samples_symbol*2/3); %%% number of symbols
dt=1/Symbol_rate/samples_symbol; %%%%% dt: sampling time
Tw=dt*Nt;  %%%%%%%%%% time window
t=dt:dt:Tw; %%%%%%%%%%% time matrix

%Fourier properties
dw=1/dt/Nt*2*pi;
wmax=1/dt*pi-dw;
wmin=-1/dt*pi;
w=wmin:dw:wmax;
f=w./2/pi;

%%% Laser Output (Nt,dt,aj, bb)
[Einj_i,linewidth]=Laser(Nt,dt,1.0,200);

%%% Transmission Setup (L,D,lamda,losses)
p=transmission_parameters(0e3,6e-6,1310e-9,0.3);
Linear=exp(-p.a/2*p.dz+1i*(w.^2*p.b2*p.dz/2.0+w.^3*p.b3*p.dz/6.0+w.^4*p.b4/24.0));

%%% Transmitter Output (M,symbols,E,P,samples,dt,csper,transmitter_bw)
rng('shuffle');
data=randi([0,M-1],1,symbols); 
mod_data=qammod(data,M);

[A,signal_power,Diff]=IQ_modulator(-1*pi/2,0.2,Einj_i,PowmW,samples_symbol,dt,mod_data,80e9);
Ax=A;
%%% Transmission and Amplification
Ax= span_single_polar(Ax,p.Nz,p.g,p.dz,Linear);
power_before=10*log10(mean(abs(Ax).^2)*1000);
G=(15e-3)/mean(abs(Ax).^2);
Ax=amplifier(Ax,G,p.fo,dt,1);
Ax=butterworth(Ax, Nt, Tw, 4, 1.1*Symbol_rate, 0);

OSNR=10*log10(mean((abs(Ax).^2))/(6.626070040e-34*p.fo*1.5*(G-1)*Symbol_rate));


%% Receiver with filters

order=1.0;
offset=0;
L=1;

for rr=1:1
band=15e9;%rr*0.1*Symbol_rate; %%% 0.6 * symbol rate
Td1=3e-12;%0.3/Symbol_rate; %%% 3.5/ symbol rate
Td_in=6.3e-11;%6.3/Symbol_rate;  %%% 6.3/ symbol rate
losa=sqrt(0.5);
losb=sqrt(0.5);
losb1=sqrt(0.5);
BWel=0.5*Symbol_rate;

for det=1:1:20

 det1=det*1;


Axout_1=MZDI(Ax.*sqrt(0.3), Nt, Tw, order, band, (det1-1)*1e9+1e9, 1*Td_in, Td1, losa, losb, pi,L);
[Axout1,noise]=photodetector(Axout_1.*(losb1),dt,p.fo,length(Axout_1),length(Axout_1).*dt,BWel,0);
for kk=1:length(mod_data)
out_data1(kk)=abs(Axout1(Diff/2+samples_symbol*(kk-1)+samples_symbol/2+offset));
end

Axout_2=MZDI(Ax.*sqrt(0.3), Nt, Tw, order, band, -1.0*(det1-1)*1e9+0e9, 2*Td_in, 1*Td1, losa, losb, pi,L);
[Axout2,noise]=photodetector(Axout_2.*(losb1),dt,p.fo,length(Axout_1),length(Axout_1).*dt,BWel,0);
for kk=1:length(mod_data)
out_data2(kk)=abs(Axout2(Diff/2+samples_symbol*(kk-1)+samples_symbol/2+offset));
end

Axout_3=MZDI(Ax.*sqrt(0.3), Nt, Tw, order, band, 0*1e9, 2*Td_in, 2*Td1, losa, losb, pi,L);
[Axout3,noise]=photodetector(Axout_3.*(losb1),dt,p.fo,length(Axout_1),length(Axout_1).*dt,BWel,1);
for kk=1:length(mod_data)
out_data3(kk)=abs(Axout3(Diff/2+samples_symbol*(kk-1)+samples_symbol/2+offset));
end

Axout_4=MZDI(Ax.*sqrt(0.3), Nt, Tw, order, band, -2*det*1e9, 2*Td_in, 1*Td1, losa, losb, pi,L);
[Axout4,noise]=photodetector(Axout_4.*(losb1),dt,p.fo,length(Axout_1),length(Axout_1).*dt,BWel,1);
for kk=1:length(mod_data)
out_data4(kk)=abs(Axout4(Diff/2+samples_symbol*(kk-1)+samples_symbol/2+offset));
end

Axout_5=MZDI(Ax.*sqrt(0.3), Nt, Tw, order, band, 2*det*1e9, 2*Td_in, 1*Td1, losa, losb, pi,L);
[Axout5,noise]=photodetector(Axout_5.*(losb1),dt,p.fo,length(Axout_1),length(Axout_1).*dt,BWel,1);
for kk=1:length(mod_data)
out_data5(kk)=abs(Axout5(Diff/2+samples_symbol*(kk-1)+samples_symbol/2+offset));
end

dl=Symbol_rate/50e9*0.4e-9;
Mem=floor(p.L*p.D*dl*Symbol_rate);

ms=2;
word=fix(Mem/2)*2+51;
start=(word+1)/2;
points=length(out_data1);
out_matrix=zeros(length(out_data1), ms*word);
for k=1:points-word
    
    out_matrix(k,:)= [ out_data1(k:word+k-1)  out_data2(k:word+k-1)];% out_data3(k:word+k-1)];% out_data4(k:word+k-1)   out_data5(k:word+k-1)];
end


ofA=M;
endP=15000;
bR=ridge((real(mod_data(1:endP))+ofA)', out_matrix(1:endP, :), 0.01, 0);


xout=bR(1)+ out_matrix*bR(2:end);

bI=ridge((imag(mod_data(1:endP))+ofA)', out_matrix(1:endP, :), 0.01, 0);

yout=bI(1)+ out_matrix*bI(2:end);

out=xout-ofA-i*(yout-ofA);

data=qamdemod(mod_data, M);
Rec_data_x=qamdemod((out)', M);
start=1;
endPP=floor(length(data)*5/5);

[NNx, RRx]=biterr(data(endP+start+400:endPP-1000), Rec_data_x(endP+401:endPP-start+1-1000));
BER_out(det,rr)=RRx; % teliko bit error

figure
histogram2(repmat(real(out(100:end-100)),1),repmat(imag(out(100:end-100)),1),[100 100],'FaceColor','flat')
set(gcf,'color','w')


ROPw(det)=10*log10(mean(Axout1(end/2:end/2+10000))*1000);
ROPc(det)=10*log10(mean(Axout2(end/2:end/2+10000))*1000);
%     end
end
end
%end
% ber(uu)=BER_out(det);


end
%% Figures

% fasma=abs(fftshift(fft(Ax))).^2;
% n =3000;
% bbb = arrayfun(@(i) mean(fasma(i:i+n-1)),1:n:length(fasma)-n+1)';
% % plot(f(n+end/2-300*n:n:end/2+n*300),10*log10(bbb(1+end/2-300:end/2+300)))
% % hold on
% fasma2=abs(fftshift(fft(Axout_2))).^2;
% fasma3=abs(fftshift(fft(Axout_1))).^2;
%  n = 500; % average every n values
%  aa = reshape(cumsum(ones(n,10),2),[],1); % arbitrary data
%  bbb1 = arrayfun(@(i) mean(fasma2(i:i+n-1)),1:n:length(fasma2)-n+1)'; % the averaged vector
%  bbb3 = arrayfun(@(i) mean(fasma3(i:i+n-1)),1:n:length(fasma3)-n+1)'; % the averaged vector

%     plot(f(n+end/2-2000*n:n:end/2+n*2000),10*log10(bbb1(1+end/2-2000:end/2+2000)))
%     hold on
%     plot(f(n+end/2-2000*n:n:end/2+n*2000),10*log10(bbb3(1+end/2-2000:end/2+2000)))

% 
% figure
% prin=(fftshift(fft(Ax)));
%  meta=(fftshift(fft(Axout_1)));
%  sun=meta./prin;
% plot(f,10*log10(abs(sun).^2))
% hold on
% prin=(fftshift(fft(Ax)));
%  meta=(fftshift(fft(Axout_2)));
%  sun2=meta./prin;
% plot(f,10*log10(abs(sun2).^2))
% figure
% % constdiag = comm.ConstellationDiagram
% %  constdiag(out)
% histogram2(repmat(real(out(100:end-100)),1),repmat(imag(out(100:end-100)),1),[100 100],'FaceColor','flat')

