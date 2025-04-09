% clear all
% close all

%% Define constants
t_node_min = 750e-12;      step_tnode = 250e-12;    t_node_max = 1000e-12;   % node seperation limits
t_node = t_node_min : step_tnode : t_node_max;
mod_ampl_min = 5e-3;            mod_ampl_step = 0.5e-3;     mod_ampl_max = 10e-3;    % modulaiton amplitude limits
mod_ampl = mod_ampl_min : mod_ampl_step : mod_ampl_max;
row_ker = [4];              % rows per kernel

%% Load Spheres images
load('C:\Users\Menelaos Skontranis\Desktop\Postdoc\Databases\Cytometry\Spheres_original.mat')
load('C:\Users\Menelaos Skontranis\Desktop\Postdoc\Databases\Cytometry\Target_Spheres.mat')
Initial_Images = Whole_images;

%% Stabilize laser
I_bias = 28e-3;
p = constants(1,1,1);
[ Ng , Nabs , Nph ] = INITIALIZATION( p );
for cc_t = 1 : p.stab - 1
    % compute gain section derivatives     
    Ag = - p.Gammag*p.gg*(Nph(cc_t)/p.Vg)*1/(1 + p.s*Nph(cc_t)) - 1/p.tg;
    Bg = p.Gammag*p.gg*p.Ng0*(Nph(cc_t)/p.Vg)*1/(1 + p.s*Nph(cc_t)) + I_bias/(p.Vg*p.q); 
    
    % compute absorber section derivatives 
    Aa = - p.Gammaa*p.ga*(Nph(cc_t)/p.Vabs)*1/(1 + p.s*Nph(cc_t)) - 1/p.tabs;
    Ba = p.Gammaa*p.ga*(p.Nabs0*Nph(cc_t)/p.Vabs)*1/(1 + p.s*Nph(cc_t));   
    
    % compute photon derivatives 
    As = p.Gammag*p.gg*(Ng(cc_t) - p.Ng0)*p.Vg/p.V + p.Gammaa*p.ga*(Nabs(cc_t) - p.Nabs0)*p.Vabs/p.V - 1/p.tph;
    Bs = p.bsp*p.Br*p.Vg*Ng(cc_t).^2; 

    % Compute new values    
    Ng( cc_t + 1 ) = Ng( cc_t ) * exp( Ag * p.dt ) + Bg / Ag * ( exp( Ag * p.dt ) - 1 );
    Nabs( cc_t + 1 ) = Nabs( cc_t ) * exp( Aa * p.dt ) + Ba / Aa * ( exp( Aa * p.dt ) - 1 );
    Nph( cc_t + 1 ) = Nph( cc_t ) * exp( As * p.dt ) + Bs / As * ( exp( As * p.dt ) - 1 );
end            
Pph = p.eta_c * p.Gammag * p.h * p.c * Nph / ( p.tph * p.lambda ); 

%% Process Sphere Images
for cc_ker = 1 : length( row_ker )
    for nr_lin_comb = 1 : 3
        % constants
        p = constants( row_ker(cc_ker) , nr_lin_comb , length(Target) );

        rng(123)
        mask = rand( p.nr_lin_comb , p.kernel_col * p.kernel_row );
        Msk_data = zeros( length(Target) , p.nr_lin_comb*p.nr_CW );
        Images = zeros( length(Target) , p.Nr_Scanning_Directions*p.row_img*p.col_img );
        
        % Create image
        Output = zeros( p.tot_syms , p.nr_lin_comb*p.nr_CW );
              
        for cc_img = 1 : length( Target )
            Images(cc_img , 1:p.row_img*p.col_img ) = Hor_Snake_Scan_Convert_2D_to_1D_with_stride(Initial_Images(cc_img,:,:), p.kernel_row , p.kernel_col , p.row_stride , p.col_stride );
            Images(cc_img , p.row_img*p.col_img + 1 : 2*p.row_img*p.col_img) = Ver_Snake_Scan_Convert_2D_to_1D_with_stride(Initial_Images(cc_img,:,:), p.kernel_row , p.kernel_col , p.row_stride , p.col_stride );                
        end
        
        % Masking
        rng(123)
        for cc_img = 1 : length(Target)
            for cc_CW = 1 : p.nr_CW
                for cc_lin_comb = 1 : p.nr_lin_comb
                    Msk_data( cc_img , ( cc_CW - 1 )*p.nr_lin_comb + cc_lin_comb ) = Images( cc_img , ( cc_CW - 1 )*p.kernel_row*p.kernel_col + 1 : cc_CW*p.kernel_row*p.kernel_col ) * mask( cc_lin_comb , : )'; 
                end
            end
        end
        Msk_data = Msk_data ./ max(max(Msk_data));
        
        for cc_t_node =  1 : length( t_node )
            p.tnode = round(t_node(cc_t_node)/p.dt);
            % Injection
            for cc_inj = 1 : length( mod_ampl )            
                % Initialize Laser
                Ng_sym = 1e-10 * ones( 1 , p.nr_samples * p.tnode ); 
                Nabs_sym = 1e-10 * ones( 1 , p.nr_samples * p.tnode ); 
                Nph_sym = 1e-10 * ones( 1 , p.nr_samples * p.tnode ); 
        
                Ng_sym( 1 ) = Ng( end );
                Nabs_sym( 1 ) = Nabs( end );
                Nph_sym( 1 ) = Nph( end );
                
        
                for cc_img = 1 : p.tot_syms
                    % Insert Data       
                    Data = mod_ampl(cc_inj) * Msk_data(cc_img,:);
                    Data = kron( Data , ones( 1 , p.tnode ) );
        
                    %% Simulation of symbol is performed in samples as simulating the entire symbol consumed the memory of our computers
                    for cc_n = 1 : p.nr_samples
                        for cc_t = 1 : p.tnode
                            t = ( cc_n - 1 ) * p.tnode + cc_t;
        
                            % compute gain section derivatives    
                            Ag = - p.Gammag*p.gg*(Nph_sym(t)/p.Vg)*1/(1 + p.s*Nph_sym(t)) - 1/p.tg;
                            Bg = p.Gammag*p.gg*p.Ng0*(Nph_sym(t)/p.Vg)*1/(1 + p.s*Nph_sym(t)) + (I_bias + Data(t))/(p.Vg*p.q); 
                            
                            % compute absorber section derivatives 
                            Aa = - p.Gammaa*p.ga*(Nph_sym(t)/p.Vabs)*1/(1 + p.s*Nph_sym(t)) - 1/p.tabs;
                            Ba = p.Gammaa*p.ga*(p.Nabs0*Nph_sym(t)/p.Vabs)*1/(1 + p.s*Nph_sym(t));   
                            
                            % compute photon derivatives
                            As = p.Gammag*p.gg*(Ng_sym(t) - p.Ng0)*p.Vg/p.V + p.Gammaa*p.ga*(Nabs_sym(t) - p.Nabs0)*p.Vabs/p.V - 1/p.tph;
                            Bs = p.bsp*p.Br*p.Vg*Ng_sym(t).^2; 
                                                            
                            % Compute new values according to the exponential integrator method 
                            Ng_sym( t + 1 ) = Ng_sym( t ) * exp( Ag * p.dt ) + Bg / Ag * ( exp( Ag * p.dt ) - 1 );
                            Nabs_sym( t + 1 ) = Nabs_sym( t ) * exp( Aa * p.dt ) + Ba / Aa * ( exp( Aa * p.dt ) - 1 );
                            Nph_sym( t + 1 ) = Nph_sym( t ) * exp( As * p.dt ) + Bs / As * ( exp( As * p.dt ) - 1 );  
                        end
                    end
                    
                    %compute output power
                    P_sym = p.eta_c * p.Gammag * p.h * p.c * Nph_sym / ( p.tph * p.lambda );
        
                    %% Convert Spikes to Binary Representation
                    [val,loc] = findpeaks(P_sym,'MinPeakHeight',3e-4);
                    loc = ceil(loc/p.tnode);
                    for cc_sp = 1 : length(loc)
                        Output(cc_img,loc(cc_sp)) = Output(cc_img,loc(cc_sp)) + 1;
                    end

                    Ng_sym( 1 ) = Ng_sym( end );
                    Nabs_sym( 1 ) = Nabs_sym( end );
                    Nph_sym( 1 ) = Nph_sym( end );
                end
                % save generated binary output
                save( [ 'C:\Users\Menelaos Skontranis\Desktop\Postdoc\Time_Delay_Photonic_SNN\Cytometry_with_a_single_VCSEL\Results\TDLSM_Cytometry_nodes=' num2str(p.nr_samples) '_tnode=' num2str(p.tnode*1e12*p.dt) '_Ib=' num2str(I_bias*(1e3)) 'mA_mod_ampl=' num2str(mod_ampl(cc_inj)*1e3) 'mA_Nrlincomb=' num2str(p.nr_lin_comb) '_kernel=' num2str( p.kernel_row ) 'x' num2str(p.kernel_col) '_stride=' num2str( p.row_stride ) '_Whole_images.mat'] , 'Output')
            end
        end
    end
end

function  [ Ng , Nabs , Nph ] = INITIALIZATION( p )
    Ng = 1e1 * ones( 1 , p.stab );
    Nabs = 1e1 * ones( 1 , p.stab );
    Nph = 1e-10 * ones( 1 , p.stab );
end


function p = constants( row_ker , nr_lin_comb , tot_syms)
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
    R_abs = 0.05;                              % absorber ratio
    p.L = 200e-6;                              % Cavity length
    p.Labs = R_abs*p.L;                        % Absorber's length
    p.Lg = (1 - R_abs)*p.L;                    % Gain Length
    p.W = 4e-6;     p.H = 98e-9;               % width and height
    p.Vg = p.Lg*p.H*p.W;   p.Vabs = p.Labs*p.H*p.W;  p.V = p.Vg + p.Vabs;  % gain, absorber and total cavity volume

    %% Losses
    Vrev = 0;                                  % reverse bias voltage
    p.am = log(1/(p.R1*p.R2))/(2*p.L);           % mirror losses
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
    p.tot_syms = tot_syms;

    %% TDRC parameters
    p.kernel_col = row_ker;                    % rows per kernel
    p.kernel_row = row_ker;                    % columns per kernel
    p.row_stride = p.kernel_row;               % row stride    
    p.col_stride = p.kernel_col;               % columns stride
    p.row_img = 100;                           % rows per image
    p.col_img = 100;                           % columns per image
    p.inputs_of_CW = p.kernel_col * p.kernel_row;   % inputs per convolutional window
    p.nr_lin_comb = nr_lin_comb;               % number of linear combinations per kernel
    p.Nr_Scanning_Directions = 2;              % number of scanning directions 
    p.nr_CW = p.Nr_Scanning_Directions*length(1:p.row_stride:p.row_img-1) * length(1:p.col_stride:p.col_img-1);   % number of convolutional windows
    p.nr_samples = p.nr_lin_comb * p.nr_CW;    % number of inputs at the laser
end