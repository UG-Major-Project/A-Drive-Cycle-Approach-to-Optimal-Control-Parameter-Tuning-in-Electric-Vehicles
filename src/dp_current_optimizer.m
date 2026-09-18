%% dp_current_optimizer.m
% Before/after comparison: your Simulink baseline current draw & SoC vs a
% DP-optimal current trajectory that minimizes I^2*R heating + jerk for
% the SAME required power profile, subject to the same SoC/current limits.
%
% RUN THIS AFTER results_analysis.m — it reuses t, net_power,
% SoC_corrected (and optionally V_batt / have_Vbatt) from that script's
% workspace.
%
% This is a MATLAB port of current_optimization.ipynb's Layer-2 DP,
% adapted to consume your Simulink battery power trace instead of its
% own synthetic drive-cycle physics model.

if ~exist('results/dp', 'dir'); mkdir('results/dp'); end

%% ---------------------------------------------------------------
%  1. Required current, resampled to a coarser dt (DP cost scales
%     with N * n_soc * n_I, so we downsample from 0.1 s to 1 s just
%     like the notebook does).
%  ---------------------------------------------------------------
dt_dp = 1.0;                          % s, DP time step
t_dp  = (t(1):dt_dp:t(end))';
P_req = interp1(t, net_power, t_dp, 'linear', 'extrap');   % W, +ve = draw

V_nom      = 360.0;                   % matches your battery pack
Q_batt_Ah  = 50.0;                    % matches your battery pack
I_req      = P_req / V_nom;           % A

%% ---------------------------------------------------------------
%  2. Battery / DP parameters
%  ---------------------------------------------------------------
SOC_min = 0.10; SOC_max = 1.00;   % must cover SOC0 — grid can't represent
                                    % anything above SOC_max, so if SOC0 sits
                                    % above it, the forward rollout snaps down
                                    % to SOC_max on the very first step
SOC0    = 1.0;                                    % you start full

% Target the SAME endpoint your baseline sim actually reached, so the
% comparison isolates "smoother/lower-loss path to the same place"
% rather than "ended up with more energy left."
SOC_target = SoC_corrected(end);
SOC_band   = 0.01;

% Matches battery_soc_upgrade.m's Thevenin model, not the notebook's
% placeholder SoC-dependent curve. R0 is the series resistance; R1 is the
% RC-branch resistance, which at DC/steady-state (the assumption the DP's
% quasi-static I^2R integral makes) carries current in series with R0 too
% (the capacitor no longer blocks it), so the effective resistance the
% pack presents is R0+R1. It's constant here because your model's R0/R1
% don't vary with SoC — if you later make them SoC-dependent in the
% battery block, mirror that dependence here too.
R0_batt = 0.05;   % ohm, matches battery_soc_upgrade.m
R1_batt = 0.03;   % ohm, matches battery_soc_upgrade.m
R_int   = @(soc) (R0_batt + R1_batt) * ones(size(soc));
alpha   = 0.01;                                    % jerk penalty weight (was 0.002)

% Tightened from +/-25A: letting the DP drift this far from what's
% actually demanded means it's no longer delivering the same tractive
% power the drive cycle requires at that instant -- not physically
% meaningful. Narrower band forces it to actually smooth the required
% trajectory rather than substitute a different one.
dI_tol         = 12.0;     % A, allowed deviation from I_req per step
I_hard_min     = -30.0;    % A -- data-driven range instead of a huge
I_hard_max     =  90.0;    % A    generic window, so the grid resolution
                             %      below actually lands on useful currents

% Grid resolution (coarsen these first if it's too slow: 41x41 instead
% of 81x81 cuts runtime roughly 4x with modest optimality loss).
% NOTE: since I_hard_min/max above are now a tight, realistic range
% (not -120..220), the same n_I gives ~5x finer current resolution
% than before -- this is what was producing the staircase artifact.
n_soc = 61; n_I = 61;
soc_grid = linspace(SOC_min, SOC_max, n_soc);
I_grid   = linspace(I_hard_min, I_hard_max, n_I);
N        = length(t_dp);
BIG_M    = 1e6;
Q_As     = Q_batt_Ah * 3600.0;

%% ---------------------------------------------------------------
%  3. Backward DP recursion
%  ---------------------------------------------------------------
V      = BIG_M * ones(N+1, n_soc, n_I);
% Soft terminal cost instead of a hard in/out-of-band cliff: a quadratic
% penalty around the target, scaled so being 1% off SOC costs roughly as
% much as a moderate current spike would. This removes the incentive to
% take a large last-minute corrective swing just to land inside a narrow
% hard window -- the old BIG_M step function did exactly that.
W_terminal = 5e6;   % cost per (SOC fraction)^2 of terminal error
terminal_cost = W_terminal * (soc_grid - SOC_target).^2;   % 1 x n_soc
V(N+1, :, :) = reshape(repmat(terminal_cost', 1, n_I), [1, n_soc, n_I]);
policy = zeros(N, n_soc, n_I, 'int16');

Rk = R_int(soc_grid);   % 1 x n_soc, same at every step (no thermal coupling)

fprintf('Running DP backward pass (%d steps, %dx%d grid)...\n', N, n_soc, n_I);
tic
for k = N:-1:1
    I_lo = max(I_hard_min, I_req(k) - dI_tol);
    I_hi = min(I_hard_max, I_req(k) + dI_tol);
    cand_mask = (I_grid >= I_lo) & (I_grid <= I_hi);
    cand_idx  = find(cand_mask);
    cand_I    = I_grid(cand_idx);

    if isempty(cand_I)
        error('No feasible current at step %d — widen dI_tol or I_hard bounds.', k);
    end

    for ip = 1:n_I
        I_prev_val = I_grid(ip);
        best_cost = inf(1, n_soc);
        best_ci   = zeros(1, n_soc, 'int16');

        for c = 1:length(cand_I)
            I_val = cand_I(c);
            I_idx = cand_idx(c);

            stage_cost = I_val^2 * Rk * dt_dp ...
                       + alpha * ((I_val - I_prev_val) / dt_dp)^2 * dt_dp;

            soc_next   = soc_grid - I_val * dt_dp / Q_As;
            infeasible = (soc_next < SOC_min) | (soc_next > SOC_max);
            soc_next_c = min(max(soc_next, SOC_min), SOC_max);

            future_cost = interp1(soc_grid, squeeze(V(k+1, :, I_idx)), soc_next_c);

            total = stage_cost + future_cost;
            total(infeasible) = BIG_M;

            better = total < best_cost;
            best_cost(better) = total(better);
            best_ci(better)   = I_idx;
        end

        V(k, :, ip)      = best_cost;
        policy(k, :, ip) = best_ci;
    end
end
fprintf('DP backward pass done in %.1f s\n', toc);

%% ---------------------------------------------------------------
%  4. Forward rollout — DP-optimal trajectory
%  ---------------------------------------------------------------
soc_dp = zeros(N+1, 1);
I_dp   = zeros(N, 1);
soc_dp(1) = SOC0;
[~, Iprev_idx] = min(abs(I_grid - 0));

for k = 1:N
    [~, soc_idx] = min(abs(soc_grid - soc_dp(k)));
    I_idx = policy(k, soc_idx, Iprev_idx);
    I_dp(k) = I_grid(I_idx);
    soc_dp(k+1) = min(max(soc_dp(k) - I_dp(k)*dt_dp/Q_As, SOC_min), SOC_max);
    Iprev_idx = I_idx;
end

%% ---------------------------------------------------------------
%  5. Baseline — "before": your Simulink battery current/SoC,
%     resampled onto the same 1 s grid as the DP for a fair comparison.
%  ---------------------------------------------------------------
I_base   = interp1(t, net_power, t_dp, 'linear', 'extrap') / V_nom;
I_base   = min(max(I_base, I_hard_min), I_hard_max);
soc_base = interp1(t, SoC_corrected, t_dp, 'linear', 'extrap');

%% ---------------------------------------------------------------
%  6. Objective comparison
%  ---------------------------------------------------------------
loss_dp     = I_dp.^2 .* R_int(soc_dp(1:end-1)) .* dt_dp;
rate_pen_dp = alpha * [0; diff(I_dp)].^2 / dt_dp;
J_dp        = sum(loss_dp + rate_pen_dp);

loss_base     = I_base.^2 .* R_int(soc_base) .* dt_dp;
rate_pen_base = alpha * [0; diff(I_base)].^2 / dt_dp;
J_base        = sum(loss_base + rate_pen_base);

fprintf('========================================\n');
fprintf('   DP Current Optimization: Before vs After\n');
fprintf('========================================\n');
fprintf('Cycle length             : %.0f s\n', t_dp(end));
fprintf('DP sample time            : %.1f s\n', dt_dp);
fprintf('Required current range    : %.2f to %.2f A\n', min(I_req), max(I_req));
fprintf('Baseline cost J (before)  : %.4f\n', J_base);
fprintf('DP cost J (after)         : %.4f\n', J_dp);
fprintf('Cost reduction            : %.2f%%\n', 100*(J_base-J_dp)/J_base);
fprintf('Final SoC baseline        : %.4f\n', soc_base(end));
fprintf('Final SoC DP              : %.4f\n', soc_dp(end));
fprintf('Peak |current| baseline   : %.1f A\n', max(abs(I_base)));
fprintf('Peak |current| DP         : %.1f A\n', max(abs(I_dp)));
fprintf('========================================\n');

%% ---------------------------------------------------------------
%  7. Plots — battery current & SoC, before vs after
%  ---------------------------------------------------------------
fig1 = figure(11); clf(fig1);
plot(t_dp, I_req, ':', 'LineWidth', 1.0, 'Color', [0.5 0.5 0.5]); hold on;
plot(t_dp, I_base, '--', 'LineWidth', 2.5);
plot(t_dp, I_dp, '-', 'LineWidth', 2.0);
legend('Required current', 'Baseline (before)', 'DP-optimal (after)', 'Location', 'best');
xlabel('Time (s)'); ylabel('Battery Current (A)');
title('Battery Current: Before vs After DP Optimization');
grid on;
save_fig_dp(fig1, 'results/dp/dp_current_comparison.png');

fig2 = figure(12); clf(fig2);
plot(t_dp, soc_base, '--', 'LineWidth', 2.0); hold on;
plot(t_dp, soc_dp(1:end-1), '-', 'LineWidth', 2.0);
yline(SOC_target, ':', 'Target SoC', 'LineWidth', 1.5);
legend('Baseline (before)', 'DP-optimal (after)', 'Location', 'best');
xlabel('Time (s)'); ylabel('SoC');
title('Battery SoC: Before vs After DP Optimization');
grid on;
save_fig_dp(fig2, 'results/dp/dp_soc_comparison.png');

fig3 = figure(13); clf(fig3);
plot(t_dp, loss_base, '--', 'LineWidth', 2.0); hold on;
plot(t_dp, loss_dp, '-', 'LineWidth', 2.0);
legend('Baseline (before)', 'DP-optimal (after)', 'Location', 'best');
xlabel('Time (s)'); ylabel('I^2R loss per step (J)');
title('Resistive Heating: Before vs After DP Optimization');
grid on;
save_fig_dp(fig3, 'results/dp/dp_heat_loss_comparison.png');

%% Save the numeric comparison for later use (e.g. in a report)
comparison = table(t_dp, I_req, I_base, I_dp, soc_base, soc_dp(1:end-1), ...
    'VariableNames', {'time_s','I_req_A','I_base_A','I_dp_A','SoC_base','SoC_dp'});
writetable(comparison, 'results/dp/dp_comparison.csv');
fprintf('Saved comparison plots and results/dp/dp_comparison.csv\n');

%% ==================== Local functions ====================
function save_fig_dp(fig, filepath)
    figure(fig);
    drawnow;

    % Try exportgraphics first (best quality), then fall back to saveas,
    % then to print, so a quirk in any one save API doesn't block output.
    try
        exportgraphics(fig, filepath, 'Resolution', 150);
        return
    catch ME1
        warning('exportgraphics failed for %s (%s) — trying saveas.', filepath, ME1.message);
    end

    try
        saveas(fig, filepath);
        return
    catch ME2
        warning('saveas failed for %s (%s) — trying print.', filepath, ME2.message);
    end

    try
        print(fig, filepath, '-dpng', '-r150');
        return
    catch ME3
        warning('Could not save %s by any method: %s', filepath, ME3.message);
    end
end