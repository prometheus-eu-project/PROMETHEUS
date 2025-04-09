close all;
clear all;

tic
%% Power

PowdBm=5;
mW1=1e-3;
PowmW=mW1*10.^(PowdBm/10)/4;  %% 8 stands for 2*(+1*M)


%% Parameters

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
L=50e3;

%%%%%%%%%%%%%% laser 
Einj_i=zeros(1, Nt)*100;

rng('shuffle')

%%% Laser Output (Nt,dt,aj, bb)
[Einj_i,linewidth]=Laser(Nt,dt,2,100);

Spans=1;
%%% Transmission Setup (L,D,lamda,losses)
p=transmission_parameters(L,15e-6,1550e-9,0.3);
Linear=exp(-p.a/2*p.dz+1i*(w.^2*p.b2*p.dz/2.0+w.^3*p.b3*p.dz/6.0+w.^4*p.b4/24.0));

%%% Transmitter Output (M,symbols,E,P,samples,dt,transmitter_bw)
M=4;
[A,Pdbm,mod_data,Diff]=PAM_transmitter(M,symbols,Einj_i,PowmW,samples_symbol,dt,Symbol_rate*0.7);
Axin=A;


%% Transmission --- 

Ax= span_single_polar(Axin,p.Nz,p.g,p.dz,Linear);

Ax=Ax.*sqrt(0.5); %%% insertion losses about 4 db

G=(2e-3)/mean(abs(Ax).^2);
% Ax=amplifier(Ax,G,p.fo,dt,1);
% Ax=butterworth(Ax,length(Ax),length(Ax).*dt,1,80e9,0);

sps=1;
Sampling_Rate=samples_symbol/sps;

OSNR=10*log10(mean((abs(Ax).^2))/(6.626070040e-34*p.fo*1.5*(G-1)*Symbol_rate));


%% Filter loops
for ii=1:2:1      %%%  detuning  
    for jj=130:10:130      %%%  bandwidth (FSR for MZDI)
        for ll=11:2:11    %%%  feedback strength
            for rr=6:2:6   %%%  delay loop
 
a = 20;   % linear Losses  (m-1)
K1=0.1;
K2=0.0;
R=jj*10e-6;

det=(ii-1);
order=1.0;
offset=samples_symbol/2;
band=jj*1e9;
Td1=(rr-1)*0.1/Symbol_rate;
Td_in=0;
phase=pi;
losa=sqrt(0.5);
losb=sqrt(0.5);
F=(ll-1)*0.1;
%%

PD_BW=Symbol_rate/2;
res=samples_symbol;

 [Axout_1,Hf1]=MZDI(Ax.*sqrt(1), Nt, Tw, order, 1*band, +det*1e9, Td_in, Td1, losa, losb,phase,F);
% [Axout_1, Hth, Hdr]=MRR(Ax, Nt,Tw,order,band,+det*1e9, Td_in, Td1, losa, losb, a, K1, K2, R); 
[Axout1,noise]=photodetector(Axout_1,dt,p.fo,length(Axout_1),length(Axout_1).*dt,PD_BW,1);



 [Axout_2,Hf2]=MZDI(Ax.*sqrt(1), Nt, Tw, order, 1*band, -det*1e9, Td_in, Td1, losa, losb,phase,F);
% [Axout_2, Hth, Hdr]=MRR(Ax, Nt,Tw,order,band,-det*1e9, Td_in, Td1, losa, losb, a, K1, K2, R);
[Axout2,noise]=photodetector(Axout_2,dt,p.fo,length(Axout_2),length(Axout_2).*dt,PD_BW,1);

Axout3=(Axout1 + Axout2);

out_data_temp3=downsample(Axout3(Diff/2+samples_symbol/2+offset:end-Diff/2+offset), res);
out_data3=resample(out_data_temp3,res,samples_symbol);

out_data_temp1=downsample(Axout1(Diff/2+samples_symbol/2+offset:end-Diff/2+offset), res);
out_data1=resample(out_data_temp1,res,samples_symbol);

out_data_temp2=downsample(Axout2(Diff/2+samples_symbol/2+offset:end-Diff/2+offset), res);
out_data2=resample(out_data_temp2,res,samples_symbol);



dl=Symbol_rate/50e9*0.4e-9;
Mem=floor(L*p.D*dl*Symbol_rate);

ms=1;
word=fix(Mem/2)*2+35;
start=(word+1)/2;
points=length(out_data1);
out_matrix=zeros(length(out_data1), ms*word);

for k=1:points-word
  out_matrix(k,:)= [out_data1(k:word+k-1)];% out_data2(k:word+k-1)];% out_data3(k:word+k-1)  out_data4(k:word+k-1)];
end

b=ridge(resample((mod_data(start:end/2+start-1))',sps,1), out_matrix(1:end/2, 1:word*ms), 0.00001, 0);

 xout=b(1)+ out_matrix*b(2:end);

data=pamdemod(mod_data-1*M, M);
Rec_data_x=pamdemod((xout(1:sps:end)-1*M)', M);

[NNx_tr, RRx_tr]=biterr(data(1+start+400:end/4-100), Rec_data_x(1+401:end/4-start+1-100));
BER_tr(ii,jj,ll,rr)=RRx_tr; % teliko bit error training

[NNx, RRx]=biterr(data(end/4+start+400:end-100), Rec_data_x(end/4+401:end-start+1-100));
BER_out(ii,jj,ll,rr)=RRx; % teliko bit error test

ROP(ii,jj)=(10*log10(mean(Axout2(end/2-10000:end/2+10000))/1e-3)); %%% in dBm

            end
        end
    end
end

toc

ber(1,:)=squeeze(BER_out(ii,jj,ll,rr));
%% Others
% snr=mean(Axout2)/noise(1,1);
% % 
%   ber(:,:)=squeeze(BER_out(:,:,ll,rr));
 
%  BER=(1/(sqrt(2*pi)*Q)*exp((-Q^2)/2));
Q=sqrt(2)*erfcinv(2*ber);



% % % % arxiko=fftshift(fft(Axin));
% % % % kanali=fftshift(fft(Ax));
% % % % filtro1=fftshift(fft(Axout_1));
% % % % filtro2=fftshift(fft(Axout_2));
% % % % CD=kanali./arxiko;
% % % % CD_F1=filtro1./arxiko;
% % % % CD_F2=filtro2./arxiko;
% % % % 
% % % % figure
% % % % plot(f,real(CD))
% % % % hold on
% % % % plot(f,real(CD_F1))
% % % % hold on
% % % % plot(f,real(CD_F2))
% % % %  
% % % % figure
% % % % plot(f,imag(CD))
% % % % hold on
% % % % plot(f,imag(CD_F1))
% % % % hold on
% % % % plot(f,imag(CD_F2))
% prin=(fftshift(fft(Ax)));
%  meta=(fftshift(fft(Axout_1)));
%  sun=meta./prin;
%  prin2=(fftshift(fft(Ax)));
%  meta2=(fftshift(fft(Axout_2)));
%  sun2=meta2./prin2;
%  plot(f/1e9,10*log10(abs(sun).^2),'Linewidth',1)
% figure
%  origin=10*log10(abs(sun).^2);
%  plot(f,10*log10(abs(sun).^2))
% figure 
% plot(f,angle(sun))
% xlabel('dΦ/dt','Fontsize',18);
% ylabel('Amplitude(dB)','Fontsize',18);
%% averaging
fasma2=abs(fftshift(fft(Axout2))).^2;
fasma1=abs(fftshift(fft(Axout1))).^2;
fasma3=(fasma1+fasma2)/2;
fasma_pf=abs(fftshift(fft((Ax)))).^2;
% % % % fasma_xout=abs(fftshift(fft(abs(xout).^2)));

fasma_1=abs(fftshift(fft(Axout_1))).^2;
fasma_2=abs(fftshift(fft(Axout_2))).^2;
 n = 300; % average every n values
 
% % % % bbb_xout=arrayfun(@(i) mean(fasma_xout(i:i+n-1)),1:n:length(fasma_xout)-n+1)';
bbb_pf=arrayfun(@(i) mean(fasma_pf(i:i+n-1)),1:n:length(fasma_pf)-n+1)';
 bbb1 = arrayfun(@(i) mean(fasma2(i:i+n-1)),1:n:length(fasma2)-n+1)'; % the averaged vector
  bbb2 = arrayfun(@(i) mean(fasma1(i:i+n-1)),1:n:length(fasma1)-n+1)'; % the averaged vector
bbb3 = arrayfun(@(i) mean(fasma3(i:i+n-1)),1:n:length(fasma3)-n+1)'; % the averaged vector
 bbb_1 = arrayfun(@(i) mean(fasma_1(i:i+n-1)),1:n:length(fasma_1)-n+1)'; % the averaged vector
 bbb_2 = arrayfun(@(i) mean(fasma_2(i:i+n-1)),1:n:length(fasma_2)-n+1)'; % the averaged vector
 
% % % %  figure
% % % %   plot(10*log10(bbb_xout),'Linewidth',1.5)

 figure
 plot(f(n:n:end)/1e9,10*log10(bbb3(1:end)),'Linewidth',1.5)
 hold on
  plot(f(n:n:end)/1e9,10*log10(bbb1(1:end)),'Linewidth',1.5)
  hold on
  plot(f(n:n:end)/1e9,10*log10(bbb2(1:end)),'Linewidth',1.5)
xlabel('Frequency (GHz)','Fontsize',20);
ylabel('Amplitude(dB)','Fontsize',20);
set(gca,'FontSize',20)
set(gcf,'color','w')
legend('Without ROSS','With ROSS1','With ROSS2','FontSize',20);
% lgd.FontSize = 58;
