%============================================================================
% Compute Transitional Dynamics in Baseline Model with Fixed Wages and Wedges
%
%
% Loads inc_process.mat, par.mat, grid.mat, steady.mat
%
% Date : 01/23/2018
%============================================================================

%% Housekeeping
close all;
clear;
clc;

%=========================================================================
%%                          1. Set parameters
%=========================================================================
% Convenient to have parameters from labor supply equation as globals
global theta z fac gameta

% Load calibrated parameters, steady states and asset grid
load par;
load steady;
load grid;

% Numerical parameters
T       = 100;   % horizon, high enough for convergence to be approximately complete
maxit   = 400;   % maximum number of iterations
tol_mkt = 1e-5;  % tolerance for market clearing

% Updating weights, same works for interest rate ands wedges (could be improved)
decay   = 0.2;
weight  = exp(-decay*(0:T-1));
weight  = weight / sum(weight);

% Credit crunch
t_shock = 6;                            % quarters to get to new constraint
dphi    = (phi1 - phi2) / t_shock;      % step size
phi_t   = max(phi2, phi1 - dphi*(0:T)); % full path

% Interest rate path
load trans         % mon. pol. replicates flexible-price response
r_t = max(r_t, 0); % apply ZLB

% Initial guess for labor wedge
omega_t = zeros(1, T);

% Preallocation
ib_pol_t = zeros(S, I, T); % sequence of policy functions
wei_t    = zeros(S, I, T); % sequence of weights on adjecent grid points
Bdem_t   = zeros(1, T);
C_t      = zeros(1, T);
Y_t      = zeros(1, T);
D_t      = zeros(1, T);
D_4Y_t   = zeros(1, T);

% Set rerun to 1 to speed things up
rerun = 1;

%=========================================================================
%%                 2. Iterate on interest rate sequence
%=========================================================================
if rerun == 1
  load trans_fix;
end

for it = 1:maxit
  % A) Iterate HH problem backwards
  %------------------------------------------
  c_pol = c_pol2; % start from terminal ss
  for t = T:-1:1
    % current values
    r   = r_t(t);
    phi = phi_t(t+1);
    fac = (pssi ./  (1-(omega_t(t))) ./ theta).^ (1/eta); % wedge shifts labor supply equation

    % update budget constraint
    tau = (pr(1)*nu + r/(1+r)*B) / (1 - pr(1)); % labor tax
    z   = [nu, -tau*ones(1, S-1)];              % full transfer scheme (tau tilde in paper)

    % find consumption at the lower bound of the state space
    for s = 1:S
      cl(s) = fzero('find_cl',...                  % objective
                    [cmin, 100],...                % interval for bisection
                    [],...                         % options
                    s, -phi_t(t), -phi_t(t+1), r); % state, assets, interest rate
    end

    % expected marginal utility tomorrow
    ui = Pr * c_pol.^(-gam);
    ui = ui(:, b_grid >= -phi);

    % previous consumption and savings policy
    for s = 1:S
      % unconstrained
      c = ((1+r) * bet * ui(s, :)) .^ (-1/gam);                   % Euler
      n = max(0, 1 - fac(s)*c.^gameta);                           % labor supply
      b = b_grid(b_grid >= -phi) / (1+r) + c - theta(s)*n - z(s); % budget

      % constrained
      if b(1) > -phi_t(t)
          c_c = linspace(cl(s), c(1), Ic);
          n_c = max(0, 1 - fac(s)*c_c.^gameta);         % labor supply
          b_c = -phi/(1+r) + c_c - theta(s)*n_c - z(s); % budget
          b   = [b_c(1:Ic-1), b];
          c   = [c_c(1:Ic-1), c];
      end

      % update policies
      c_pol(s, :) = interp1(b, c, b_grid, 'linear', 'extrap');
      c_pol(s, :) = max(c_pol(s, :), cmin);
      n_pol(s, :) = max(0, 1 - fac(s)*c_pol(s, :).^gameta);
      y_pol(s, :) = theta(s) * n_pol(s, :);
      b_pol(s, :) = max((1+r) * (b_grid + y_pol(s, :) - c_pol(s,:) + z(s)), -phi);

      % save policies
      c_pol_t(s, :, t) = c_pol(s, :);
      n_pol_t(s, :, t) = n_pol(s, :);
      y_pol_t(s, :, t) = y_pol(s, :);
    end

    % Save weights needed to iterate distribution
    [~, ~, ib_pol]    = histcounts(b_pol, b_grid);
    wei               = (b_pol - b_grid(ib_pol)) ./ (b_grid(ib_pol+1) - b_grid(ib_pol));
    ib_pol_t(:, :, t) = ib_pol;
    wei_t(:, :, t)    = wei;
  end

  % B) Iterate distribution forwards
  %------------------------------------------
  pd = pd1; % start from initial ss
  for t = 1:T
    %  Bond demand
    Bdem_t(t) = sum(sum(pd) .* b_grid);

    % Goods market clearing
    C_t(t)    = sum(sum(pd  .* c_pol_t(:, :, t)));
    Y_t(t)    = sum(sum(pd  .* y_pol_t(:, :, t)));

    % Calculate other aggregates
    D_t(t)    = -sum(sum(pd) .* min(b_grid, 0)); % debt
    D_4Y_t(t) =  D_t(t) / Y_t(t) / 4;            % debt to annual GDP

    pdi = zeros(S, I);
    for s = 1:S
      for i = 1:I
        for si = 1:S
          pdi(si, ib_pol_t(s, i, t))     = (1 - wei_t(s, i, t)) * Pr(s, si) * pd(s, i) + pdi(si, ib_pol_t(s, i, t));
          pdi(si, ib_pol_t(s, i, t) + 1) = wei_t(s, i, t)       * Pr(s, si) * pd(s, i) + pdi(si, ib_pol_t(s, i, t) + 1);
        end
      end
    end

    % make sure that distribution integrates to 1
    pd = pdi / sum(sum(pdi));
  end

  % C) Check convergence
  %------------------------------------------
  % Market clearing
  BM_t = Bdem_t - B;
  GM_t = Y_t - C_t;

  % Metric of deviation
  res_BM = sqrt(BM_t * BM_t' / T);
  res_GM = sqrt(GM_t * GM_t' / T);

  % Report progress
  disp(['Iteration ', num2str(it)]);
  disp(['Bond mkt clearing: ' , num2str(res_BM)]);
  disp(['Goods mkt clearing: ', num2str(res_GM)]);
  disp('-----------------------------------------------')

  % Uncomment if want to see progression
  % figure(1);
  % subplot(2, 1, 1); plot(BM_t); title('bond market');
  % subplot(2, 1, 2); plot(GM_t); title('goods market');
  % pause(0.01);

  % Check convergence
  if max(res_BM, res_GM) < tol_mkt
    break;
  end

  % Update interest rate based on bond market
  r_t = r_t - (weight .* BM_t)';

  % Update labor wedges based on goods market
  omega_t = 1 - (1-omega_t).*exp(-weight .* GM_t)';

  % Monetary policy
  omega_t(r_t > 0) = 0;
  r_t(r_t < 0)     = 0;

end

% Save interest rate path
save trans_fix r_t omega_t

%=========================================================================
%%                              3. Figures
%=========================================================================
Tp = 24; % horizon to plot

figure(2);

% annualized interest rate
subplot(1, 2, 1);
plot(0:Tp, [r1*400; r_t(1:Tp)*400], 'LineWidth', 1.3)
title('interest rate')
box on; grid on;

% output % deviaton from steady state
subplot(1, 2, 2);
plot(0:Tp, [0, 100*(Y_t(1:Tp)/Y1-1)], 'LineWidth', 1.3);
title('output');
box on; grid on;

set(gcf,'Position',  [440   378   800   300]);
set(gcf,'PaperPosition', [0 0 16 6]);
set(gcf,'PaperSize', [16 6]);
saveas(gcf,'fig8-sticky.pdf')