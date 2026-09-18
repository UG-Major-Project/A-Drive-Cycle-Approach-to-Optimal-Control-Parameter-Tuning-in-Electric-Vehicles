%% EV Drive Cycle Results Analysis (v3 — reads root Outports via out.yout)
% Run this script after simulation completes in Simulink

clc;

%% Extract simulation outputs
t         = out.tout;
[net_power, ~, found_np] = get_signal(out, 'net_power_out');
if ~found_np
    error('net_power_out not found in out — check the To Workspace block is still named net_power_out.');
end

min_len   = min(length(t), length(net_power));
t         = t(1:min_len);
net_power = net_power(1:min_len);

% Separate motor and regen power
P_motor = max(net_power, 0);     % positive = consuming from battery
P_regen = max(-net_power, 0);    % negative = returning to battery

%% Pull the new signals (root outports: I_q_out, T_motor_out, SoC, V_terminal)
[I_q,      ~, have_Iq]    = get_signal(out, 'I_q_out');
[T_motor,  ~, have_Tmot]  = get_signal(out, 'T_motor_out');
[SoC_raw,  ~, have_SoC]   = get_signal(out, 'SoC_out');
[V_batt,   ~, have_Vbatt] = get_signal(out, 'V_terminal');

if have_Iq,    I_q     = I_q(1:min(length(I_q), min_len));         end
if have_Tmot,  T_motor = T_motor(1:min(length(T_motor), min_len)); end
if have_Vbatt, V_batt  = V_batt(1:min(length(V_batt), min_len));   end

%% Energy Calculations
E_consumed  = trapz(t, P_motor) / 3600;   % Wh
E_recovered = trapz(t, P_regen) / 3600;   % Wh
E_net       = E_consumed - E_recovered;    % Wh

%% Distance and Efficiency
min_len2    = min(length(t), length(speed_ms));
distance_km = trapz(t(1:min_len2), speed_ms(1:min_len2)) / 1000;
efficiency  = E_net / distance_km;         % Wh/km
regen_pct   = (E_recovered / E_consumed) * 100;

%% Range Estimate
battery_Wh = 50 * 360;                    % 50Ah x 360V = 18000 Wh
range_km   = battery_Wh / efficiency;

%% SoC — use the real SoC from the battery block if logged, else fall back
if have_SoC
    SoC_corrected = SoC_raw;
else
    battery_J      = battery_Wh * 3600;
    energy_used_J  = cumtrapz(t, net_power);
    SoC_corrected  = 1 - (energy_used_J / battery_J);
    SoC_corrected  = max(min(SoC_corrected, 1), 0);
end

%% Print Results
fprintf('========================================\n');
fprintf('   HWFET EV Drive Cycle Simulation\n');
fprintf('========================================\n');
fprintf('Distance Simulated  : %.2f km\n',  distance_km);
fprintf('Energy Consumed     : %.1f Wh\n',  E_consumed);
fprintf('Energy Recovered    : %.1f Wh\n',  E_recovered);
fprintf('Regen Recovery      : %.1f%%\n',   regen_pct);
fprintf('Efficiency          : %.1f Wh/km\n', efficiency);
fprintf('Estimated Range     : %.0f km\n',  range_km);
fprintf('SoC Start           : 100%%\n');
fprintf('SoC End             : %.1f%%\n',   SoC_corrected(end)*100);
fprintf('SoC Drop            : %.1f%%\n',   (1-SoC_corrected(end))*100);
if have_Tmot
    fprintf('Motor Temp Start    : %.1f degC\n', T_motor(1));
    fprintf('Motor Temp End      : %.1f degC\n', T_motor(end));
    fprintf('Motor Temp Peak     : %.1f degC\n', max(T_motor));
end
if have_Vbatt
    fprintf('Battery V Start     : %.1f V\n', V_batt(1));
    fprintf('Battery V Min       : %.1f V\n', min(V_batt));
end
if have_Iq
    fprintf('Motor Current Peak  : %.1f A\n', max(abs(I_q)));
end
fprintf('========================================\n');

%% Plot 1 — Drive Cycle Speed Profile
figure(1); clf;
plot(t(1:min_len2), speed_ms(1:min_len2)*3.6, 'b', 'LineWidth', 1.5);
title('HWFET Drive Cycle Speed Profile');
xlabel('Time (s)'); ylabel('Speed (km/h)');
xlim([0 765]); grid on;
saveas(gcf, 'results/speed_profile.png');

%% Plot 2 — Battery SoC
figure(2); clf;
plot(t, SoC_corrected, 'r', 'LineWidth', 1.5);
title('Battery State of Charge - HWFET Cycle');
xlabel('Time (s)'); ylabel('SoC (0 to 1)');
xlim([0 765]); ylim([0.97 1.0]);
grid on;
saveas(gcf, 'results/SoC_profile.png');

%% Plot 3 — Motor Phase Current
% NOTE: this is I_q (motor phase current), not battery pack current.
% To plot true battery pack current, add I_batt as a 5th output on the
% Battery SoC block and wire it to a new Outport, then it'll show up
% here automatically as out.I_batt_out.
if have_Iq
    figure(3); clf;
    plot(t, I_q, 'r', 'LineWidth', 1.5);
    title('Motor Phase Current (I_q)');
    xlabel('Time (s)'); ylabel('Current (A)');
    xlim([0 765]); grid on;
    saveas(gcf, 'results/motor_current.png');
else
    warning('I_q_out not found — skipping current plot.');
end

%% Plot 4 — Motor vs Regen Power
figure(4); clf;
plot(t, P_motor/1000, 'g', 'LineWidth', 1.5); hold on;
plot(t, P_regen/1000, 'b', 'LineWidth', 1.5);
legend('Motor Power (kW)', 'Regen Power (kW)');
title('Motor and Regenerative Braking Power');
xlabel('Time (s)'); ylabel('Power (kW)');
xlim([0 765]); grid on;
saveas(gcf, 'results/power_profile.png');

%% Plot 5 — Energy Summary Bar Chart
figure(5); clf;
bar([E_consumed, E_recovered, E_net], 0.5, ...
    'FaceColor', [0.2 0.6 0.8]);
set(gca, 'XTickLabel', {'Consumed (Wh)', 'Recovered (Wh)', 'Net (Wh)'});
title('Energy Summary - HWFET Cycle');
ylabel('Energy (Wh)');
grid on;
saveas(gcf, 'results/energy_summary.png');

%% Plot 6 — Motor Temperature
if have_Tmot
    figure(6); clf;
    plot(t, T_motor, 'Color', [0.85 0.33 0.1], 'LineWidth', 1.5);
    title('Motor Temperature - HWFET Cycle');
    xlabel('Time (s)'); ylabel('Temperature (degC)');
    xlim([0 765]); grid on;
    saveas(gcf, 'results/motor_temperature.png');
else
    warning('T_motor_out not found — skipping temperature plot.');
end

%% Plot 7 — Battery Terminal Voltage
if have_Vbatt
    figure(7); clf;
    plot(t, V_batt, 'Color', [0.1 0.5 0.3], 'LineWidth', 1.5);
    title('Battery Terminal Voltage - HWFET Cycle');
    xlabel('Time (s)'); ylabel('Voltage (V)');
    xlim([0 765]); grid on;
    saveas(gcf, 'results/battery_voltage.png');
else
    warning('V_terminal not found — skipping voltage plot.');
end

fprintf('All plots saved to /results folder.\n');

%% Local functions (must be at the end of a MATLAB script file)
function [data, tvec, found] = get_signal(simout, name)
% Pull a signal by name whether it's a top-level "out.<name>" (old-style
% To Workspace block) or bundled inside out.yout (root Outport, which is
% how I_q_out / T_motor_out / SoC / V_terminal are wired in your current
% diagram).
    data = []; tvec = []; found = false;
    if isprop(simout, name) || isfield(simout, name)
        try
            sig = simout.(name);
            data = sig.Data;
            tvec = sig.Time;
            found = true;
            return
        catch
        end
    end
    if isprop(simout, 'yout') || isfield(simout, 'yout')
        try
            el = simout.yout.getElement(name);
            data = el.Values.Data;
            tvec = el.Values.Time;
            found = true;
        catch
        end
    end
end