% LIP simulation
% 2025/12/30

clear
close all
clc

%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Declare variables %%%
%%%%%%%%%%%%%%%%%%%%%%%%%

% Model variable (Cassie)
m  = 31;        % kg
z0 = 0.5;       % m (COM height)
g  = 9.81;

% Controller variable 
Ts = 0.35;          % step time (s)
xf_i = -0.05;        % initial CoM to foot position (final, foot at 0.2)
xd = 0.1;           % desired step length
vd = 1;          % desired forward speed (m/s)
a  = 1;             % position weight
b  = 1;             % velocity weight

% Initial calculations:
Tc = sqrt(z0/g);
CT = cosh(Ts/Tc);
ST = sinh(Ts/Tc);

%%%%%%%%%%%%%%%%%%%
%%% Simulation  %%%
%%%%%%%%%%%%%%%%%%%

out = sim('SIMrun_LIP.slx','StopTime','1.5'); % run simulation

x_pos = squeeze(out.CoM.Data);
foot_pos = squeeze(out.footout.Data);

% Fix data point:
x_pos = [0.2 + xf_i; x_pos];
x_pos(end) = [];


%%%%%%%%%%%%%%%%%
%%% Graphing  %%%
%%%%%%%%%%%%%%%%%

animate_xz_com_foot(x_pos, foot_pos, z0);

