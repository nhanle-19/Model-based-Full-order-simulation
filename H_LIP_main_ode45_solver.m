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
Ts = 0.2;           % SSP
Td = 0.1;           % DSP
xi = [-0.05;0.5];   % initial state
xd = [-0.1;1];      % desired state
stoptime = 2.1;     % desired stop time
i = 0;              % initial start time
u = 0;              % initial cartesian step

% Initial calculations:
ld = sqrt(g/z0);
K  = [1, Td + (1/ld)*coth(ld*Ts)];
ud = xd(2)*(Ts + Td);  % desired step length in P1 orbit


%%%%%%%%%%%%%%%%%%%%%%
%%% Running solver %%%
%%%%%%%%%%%%%%%%%%%%%%

f = @(t, x) [x(2); ld^2 * x(1)];             % SSP dynamic function
xf = xi;                                     % set initial cond 
while i <= stoptime
    % integrate within-step dynamics
    [t, x] = ode45(f, [0 Ts], xf(:,end));    % run solver
    x = x(end,:)';
   
    % compute foot placement input for THIS step transition
    uk = ud + K * (xf(:,end) - xd);
    u = [u; u(end) + uk];
    
    % state reset
    xf = [xf, [x(1) + x(2) * Td - uk; x(2)]];
    i = i + Ts + Td;
end

x_Cart = xf(1,:)' + u;


animate_xz_com_foot(x_Cart, u, z0);