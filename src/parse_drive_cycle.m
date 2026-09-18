%% Parse HWFET Drive Cycle Data
% Run this script first before opening Simulink

clear; clc;

%% Load raw data
raw = readmatrix('hwy10hztable.txt', 'NumHeaderLines', 2);

% Remove empty/NaN rows
raw = raw(~all(isnan(raw), 2), :);

% Flatten 10 columns into single speed vector (10 Hz data)
speed_mph = reshape(raw(:, 2:end)', [], 1);

% Remove NaN values
speed_mph = speed_mph(~isnan(speed_mph));

% Convert mph to m/s
speed_ms = speed_mph * 0.44704;

% Build time vector at 0.1s intervals
dt = 0.1;
time = (0 : dt : (length(speed_ms)-1)*dt)';

%% Compute acceleration
accel = [0; diff(speed_ms)/dt];
accel = max(min(accel, 4), -6);   % clamp between -6 and +4 m/s^2

%% Package for Simulink From Workspace blocks
% Drive cycle input (matrix format)
drive_cycle_input = [time, speed_ms];

% Accel input (matrix format)
accel_input = [time, accel];

%% Save to workspace file
save('drive_cycle.mat', ...
     'drive_cycle_input', ...
     'accel_input', ...
     'time', ...
     'speed_ms', ...
     'accel', ...
     'dt');

%% Verify plot
figure;
plot(time, speed_ms * 3.6, 'b', 'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('Speed (km/h)');
title('HWFET Drive Cycle — Verification Plot');
grid on;

fprintf('Drive cycle parsed successfully.\n');
fprintf('Total duration: %.1f seconds\n', time(end));
fprintf('Total data points: %d\n', length(time));
fprintf('Max speed: %.1f km/h\n', max(speed_ms)*3.6);