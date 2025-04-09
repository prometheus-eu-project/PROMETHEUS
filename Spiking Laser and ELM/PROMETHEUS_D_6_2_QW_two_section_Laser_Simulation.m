clear all
close all

%% Scanning parameters
I_bias_start = 0e-3;  I_bias_step = 1e-3;    I_bias_final = 100e-3;     %Scanning range of the bias
Vrev_start = 0;       Vrev_step = 2;         Vrev_final = 4;            %Scanning range of the reverse voltage
L_start = 500e-6;     L_step = 500e-6;       L_final = 1000e-3;         %Scanning range of the cavity length
Rabs_start = 0.03;    Rabs_step = 0.03;      Rabs_final = 0.25;         %Scanning range of the absorber's ratio

Peak_Power = zeros(length(Vrev_start:Vrev_step:Vrev_final),length(I_bias_start:I_bias_step:I_bias_final));
ISI = zeros(length(Vrev_start:Vrev_step:Vrev_final),length(I_bias_start:I_bias_step:I_bias_final));
Pulse_duration = zeros(length(Vrev_start:Vrev_step:Vrev_final),length(I_bias_start:I_bias_step:I_bias_final));

cc_L = 1;
for L = L_start : L_step : L_final
    cc_Rabs = 1;
    for Rabs = Rabs_start : Rabs_step : Rabs_final
        cc_Vrev = 1;
        for Vrev = Vrev_start : Vrev_step : Vrev_final
        cc_I = 1;
            for I_bias = I_bias_start : I_bias_step : I_bias_final

                p = constants(L , Rabs , Vrev);           
                [ Ng , Nabs , Nph ] = INITIALIZATION( p );
                for cc_t = 1 : p.time - 1
                    % comput derivatives of the gain section carriers   
                    Ag = - p.Gammag*p.gg*(Nph(cc_t)/p.Vg)*1/(1 + p.s*Nph(cc_t)) - 1/p.tg;
                    Bg = p.Gammag*p.gg*p.Ng0*(Nph(cc_t)/p.Vg)*1/(1 + p.s*Nph(cc_t)) + I_bias/(p.Vg*p.q); 
                    
                    % compute derivatives of the absorber section carriers
                    Aa = - p.Gammaa*p.ga*(Nph(cc_t)/p.Vabs)*1/(1 + p.s*Nph(cc_t)) - 1/p.tabs;
                    Ba = p.Gammaa*p.ga*(p.Nabs0*Nph(cc_t)/p.Vabs)*1/(1 + p.s*Nph(cc_t));   
                    % compute derivatives of photons
                    As = p.Gammag*p.gg*(Ng(cc_t) - p.Ng0)*p.Vg/p.V + p.Gammaa*p.ga*(Nabs(cc_t) - p.Nabs0)*p.Vabs/p.V - 1/p.tph;
                    Bs = p.bsp*p.Br*p.Vg*Ng(cc_t).^2; 
                
                    % Compute new values according to the exponential integrator method  
                    Ng( cc_t + 1 ) = Ng( cc_t ) * exp( Ag * p.dt ) + Bg / Ag * ( exp( Ag * p.dt ) - 1 );
                    Nabs( cc_t + 1 ) = Nabs( cc_t ) * exp( Aa * p.dt ) + Ba / Aa * ( exp( Aa * p.dt ) - 1 );
                    Nph( cc_t + 1 ) = Nph( cc_t ) * exp( As * p.dt ) + Bs / As * ( exp( As * p.dt ) - 1 );
                end   

                % compute output power
                Pph = p.eta_c * p.Gammag * p.h * p.c * Nph / ( p.tph * p.lambda ); 
                
                % compute spikes peak power, pulse repetion frequency and
                % Full-Width at Half Maximum
                [val,loc,width] = findpeaks(Pph(p.stab+1:p.time),'MinPeakHeight',3e-4,'MinPeakWidth',20);
                
                % Distinguish spiking from non-spiking outputs
                if isempty(val) || (std(Pph(p.stab+1:p.time)) < 0.1e-6)
                    Peak_Power(cc_Vrev,cc_I) = 0;
                    ISI(cc_Vrev,cc_I) = 0;
                    Pulse_duration(cc_Vrev,cc_I) = 0;
                else
                    Peak_Power(cc_Vrev,cc_I) = mean(val*1e3);
                    ISI(cc_Vrev,cc_I) = 1e-9/((loc(2)-loc(1))*p.dt);
                    Pulse_duration(cc_Vrev,cc_I) = mean(width)*p.dt*1e12;
                end     
                clear val loc width 
                cc_I = cc_I + 1;
            end

            % Plot data
            figure((cc_L-1)*length(Vrev_start:Vrev_step:Vrev_final)+cc_Rabs)
            subplot(1,3,1)
            hold on
            plot(Peak_Power(cc_Vrev,:),'LineWidth',3)
            xlabel('I(mA)','FontSize',20)
            ylabel('Peak Power (mW)','FontSize',20)
            title(sprintf('L=%d Rabs=%1.2f',L*1e6,Rabs))
            subplot(1,3,2)
            hold on
            plot(ISI(cc_Vrev,:),'LineWidth',3)
            xlabel('I(mA)','FontSize',20)
            ylabel('Firing Rate (GHz)','FontSize',20)
            title(sprintf('L=%d Rabs=%1.2f',L*1e6,Rabs))
            subplot(1,3,3)
            hold on
            plot(Pulse_duration(cc_Vrev,:),'LineWidth',3)
            xlabel('I(mA)','FontSize',20)
            ylabel('Pulse duration (ps)','FontSize',20)
            title(sprintf('L=%d Rabs=%1.2f',L*1e6,Rabs))
            cc_Vrev = cc_Vrev + 1;
        end  
        
        cc_Rabs = cc_Rabs + 1;
    end
    cc_L = cc_L + 1;
end


function  [ Ng , Nabs , Nph ] = INITIALIZATION( p )
    Ng = 1e1 * ones( 1 , p.time );
    Nabs = 1e1 * ones( 1 , p.time );
    Nph = 1e-10 * ones( 1 , p.time );
end


function p = constants(L, R_abs, Vrev)
    %% global constants
    p.h = 6.626070e-34;                        % Planck's constant (J/s)
    p.c = 299792458;                           % speed of light (m/s)
    p.q = 1.6021764*1e-19;                     % electron charge
    p.T = 300;                                 % temperature 
    p.kb = 1.380649e-23;                       % Boltzman constant (J/K)
    
    %% material constants
    p.ng = 3.43;                               % refractive index 
    p.prop_losses = 1000;                      % propagation losses
    p.vg = p.c / p.ng;                         % group velocity
    p.s = 5e-23;                               % saturation coefficient
    p.lambda = 1550e-9;                        % lasing wavelength

    %% Reflectivity of mirrors
    p.R1 = 1;    p.R2 = 0.6;
    
    %% dimensions
    p.L = L;                                   % Cavity length
    p.Labs = R_abs*p.L;                        % Absorber's length
    p.Lg = (1 - R_abs)*p.L;                    % Gain Length
    p.W = 4e-6;     p.H = 98e-9;               % width and height
    p.Vg = p.Lg*p.H*p.W;   p.Vabs = p.Labs*p.H*p.W;  p.V = p.Vg + p.Vabs;  % gain, absorber and total cavity volume

    %% Losses
    p.am = log(1/(p.R1*p.R2))/(2*L);           % mirror losses
    p.a = 0.6*(p.am + p.prop_losses);          % fitting parameters for the carrier lifetime in the absorber section
    p.b = (p.L/p.Labs)*p.a;                    % fitting parameters for the carrier lifetime in the absorber section
    p.V_built = 0.6;                           % built in bias value
    p.ag = p.prop_losses;                      % losses in gain section
    p.a_sa = p.prop_losses + p.b*(p.V_built + Vrev);    % losses in absorber section
    p.total_losses = 2*p.ag*p.Lg + 2*p.a_sa*p.Labs + log(1/(p.R1*p.R2));      % total losses
    p.Gammag = 0.1;                            % confinement factor in gain section
    p.Gammaa = 0.1;                            % confinement factor in absorber section
    p.tg = 1e-9;                               % carrier lifetime in gain section
    recombination_rate = 1/p.tg;               % carrier recombination rate
    G0 = 1/100e-12 - 1/p.tg;                   
    thermal_escape_rate = G0*exp(p.H*p.q*abs(Vrev)/(2*p.W*p.kb*p.T));   % thermal escape rate
    p.tabs = 1/(recombination_rate + thermal_escape_rate);              % carrier lifetime in absorber section
    p.tph = 1/(p.vg*p.total_losses/(2*p.L));                            % photon lifetime
    p.gg = 0.3e-11;                            % differential gain in gain section
    p.ga = 3.34e-11;                           % differetial gain in absorber section
    p.Ng0 = 1.1e+24;                           % carrier density at transparecy for the gain section
    p.Nabs0 = 0.89e+24;                        % carrier density at transparecy for the absorer section
    p.Br = 10e-16;                             % Bimolecular recombination rate
    p.bsp = 1e-4;                              % spontaneous emission coefficient
    p.eta_c = 0.4;                             % output power coupling coefficient

    %% time constants 
    p.dt = 1e-12;                              % simulation time step
    p.stab = round(20e-9 / p.dt);              % stabilization time of the laser
    p.time = round( 50e-9 / p.dt );            % simulation time
end