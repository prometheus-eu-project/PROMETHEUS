function B=butterworth(A, samples,pulse_duration,N,fc,f0);

dt=pulse_duration/samples;			
j=-(samples/2):1:(samples/2)-1;
f=j*(1/samples);
f=f/dt;
Hpar=1+i*(((f-f0)/fc).^N);
H=Hpar.^(-1);
A=fft(A);
A=fftshift(A);
A=H.*A;
A=ifftshift(A);
A=ifft(A);
B=A;

