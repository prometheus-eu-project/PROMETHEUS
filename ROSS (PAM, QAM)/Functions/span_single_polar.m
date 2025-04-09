function Ax= span(Ax,Nz,g,dz,Linear)
% coder.gpu.kernelfun;
for k=1:Nz

     Ax=Ax.*exp(1i*g*abs(Ax).^2*dz);
     Axf=fft(Ax);
     Axf=fftshift(Axf);
     Axf=Axf.*Linear;
     Axf=ifftshift(Axf);
     Ax=ifft(Axf);
                  
end