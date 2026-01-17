% H-LIP simulation
% 2026/1/10

clear
close all
clc

%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Declear variables %%%
%%%%%%%%%%%%%%%%%%%%%%%%%

% Model variable (Cassie)
m  = 31;            % kg
z0 = 0.5;           % m (COM height)
g  = 9.81; 

% Controller variable 
Ts = 0.2;          % SSP
Td = 0.1;           % DSP
xi = -0.05;         % initial CoM to foot position (final, foot at 0.2)
vi = 0.5;           % initial velo
xd = [-0.1;1];      % desired state


% Initial calculations:
ld = sqrt(g/z0);
K  = [1, Td + (1/ld)*coth(ld*Ts)];
ud = xd(2)*(Ts + Td);                     % desired step length in P1 orbit

%%%%%%%%%%%%%%%%%%%
%%% Simulation  %%%
%%%%%%%%%%%%%%%%%%%
 
out = sim('SIMrun.slx','StopTime','2.1'); % run simulation

x_pos = squeeze(out.CoM.Data);
foot_pos = squeeze(out.footout.Data);

% Fix data point:
%x_pos = [xi; x_pos];
%x_pos(end) = [];
%foot_pos = [0; foot_pos];
%foot_pos(end) = [];

%%%%%%%%%%%%%%%%%
%%% Graphing  %%%
%%%%%%%%%%%%%%%%%

animate_xz_com_foot(x_pos, foot_pos, z0);

