clc
clear all
close all

mlgnori_62 = load('C:\Users\barze\Desktop\mouselgn\mlgnori_62');
spike_time = double(mlgnori_62.mlgn.spktimes);
mlgnori_56 = load('C:\Users\barze\Desktop\mouselgn\mlgnori_56');
spike_time_56 = double(mlgnori_56.mlgn.spktimes);


num_stimulus = size(spike_time,1);
sequence = 0:30:330;
stimulus = [360, sequence];
num_trial = size(spike_time,2);
num_trial_56 = size(spike_time_56,2);
time_pt = size(spike_time,3);
time = 1:time_pt;
delta_t = 1;

% for s = 1:num_stimulus  % Assuming 'stimulus' is an array containing the stimuli
%     figure();
%     hold on;
%     
%     for i = 1:num_trial
%         spike1 = spike_time(s, i, :);  % Access the spikes for the current stimulus
%         spike1 = reshape(spike1, time_pt, 1);
%         
%         plot(time, spike1 + (i-1), 'DisplayName', sprintf('Trial %d', i));
%         xlabel('Time (ms)');
%         ylabel('Trial Number');
%         title_text = sprintf('Spike Data for Each Trial (Orientation %d)', stimulus(stim));
%         title(title_text);
%     end
%     
%     ylim([0 num_trial]);
%     
%     legend show;
%     grid on;
% end




% % Define the window length and duration
% window_length = 200; % 200 ms
% window_duration = window_length / 1000; % duration in seconds
% num_windows = floor(time_pt / window_length); % Number of non-overlapping windows
% 
% % Preallocate array for storing window sums for all trials
% window_sums_all_trials = zeros(num_windows, num_trial);
% for s = 1:num_stimulus
%     % Sum spikes within each window for each trial
%     for i = 1:num_trial
%         spike1 = spike_time(s, i, :);
%         spike1 = reshape(spike1, time_pt, 1);
% 
%         for j = 1:num_windows
%             window_start = (j-1) * window_length + 1;
%             window_end = j * window_length;
%             window_sums_all_trials(j, i) = sum(spike1(window_start:window_end));
%         end
%     end
% 
%     % Calculate the average number of spikes in each window across all trials
%     average_window_sums = mean(window_sums_all_trials, 2);
% 
%     % Calculate the firing rate (spikes per second) in each window
%     firing_rate = average_window_sums / window_duration;
% 
%     % Time axis for the windows
%     window_time = (0:num_windows) * window_length;
% 
%     % Plot the average firing rate as a step function
%     figure();
%     stairs(window_time, [firing_rate; firing_rate(end)], 'LineWidth', 2);
%     xlim([0, 2000]);
%     ylim([0, 30]);
%     xlabel('Time (ms)');
%     ylabel('Average Firing Rate (spikes/s)');
%     title(sprintf('Average Firing Rate per 200 ms Window (Orientation %d)', stimulus(s)));
%     grid on;
% end


% % Define the window length and step size
% window_length = 200; % 200 ms
% step_size = 10; % 10 ms
% window_duration = window_length / 1000; % duration in seconds
% 
% % Create the window for convolution
% conv_window = ones(window_length, 1);
% 
% % Calculate the number of windows
% num_windows = floor(2000 / step_size);
% 
% % Preallocate array for storing convolved sums for all trials
% conv_sums_all_trials = zeros(num_windows, num_trial);
% 
% for s = 1:num_stimulus
%     % Convolve spikes with the window for each trial
%     for i = 1:num_trial
%         spike1 = spike_time(s, i, :);
%         spike1 = reshape(spike1, time_pt, 1);
%         
%         % Convolve the spike train with the window
%         conv_result = conv(spike1, conv_window, 'same');
% 
%         % Downsample the convolved result to match the step size
%         conv_result = conv_result(1:step_size:end);
%         
%         % Ensure the downsampled result fits into the preallocated array
%         if length(conv_result) > num_windows
%             conv_result = conv_result(1:num_windows);
%         elseif length(conv_result) < num_windows
%             conv_result(num_windows) = 0; % Zero-pad if necessary
%         end
% 
%         conv_sums_all_trials(:, i) = conv_result;
%     end
% 
%     % Calculate the average number of spikes in each window across all trials
%     average_conv_sums = mean(conv_sums_all_trials, 2);
% 
%     % Calculate the firing rate (spikes per second) in each window
%     firing_rate = average_conv_sums / window_duration;
% 
%     % Time axis for the windows
%     window_time = (0:num_windows-1) * step_size; % Center the windows on their time intervals
% 
%     % Adjust the window_time to match the end time of each window
%     window_time = window_time 
% 
%     % Plot the average firing rate
%     figure();
%     plot(window_time, firing_rate, 'LineWidth', 2);
%     xlim([0, 2000]);
%     ylim([0, 30]);
%     xlabel('Time (ms)');
%     ylabel('Average Firing Rate (spikes/s)');
%     title(sprintf('Average Firing Rate per 200 ms Window with 10 ms Step (Orientation %d)', stimulus(s)));
%     grid on;
% end



% % Define the window length and step size
% window_length = 200; % 200 ms
% step_size = 10; % 10 ms
% sigma = window_length / 2; % Sigma for Gaussian, so 2*sigma = 200 ms
% window_duration = window_length / 1000; % duration in seconds
% 
% % Create the Gaussian window for convolution
% half_window = floor(window_length / 2);
% t = -half_window:half_window;
% conv_window = exp(-t.^2 / (2 * sigma^2));
% conv_window = conv_window / mean(conv_window);
% 
% % Calculate the number of windows
% num_windows = floor(2000 / step_size);
% 
% % Preallocate array for storing convolved sums for all trials
% conv_sums_all_trials = zeros(num_windows, num_trial);
% 
% for s = 1:num_stimulus
%     % Convolve spikes with the window for each trial
%     for i = 1:num_trial
%         spike1 = spike_time(s, i, :);
%         spike1 = reshape(spike1, time_pt, 1);
%         
%         % Convolve the spike train with the Gaussian window
%         conv_result = conv(spike1, conv_window, 'same');
% 
%         % Downsample the convolved result to match the step size
%         conv_result = conv_result(1:step_size:end);
%         
%         % Ensure the downsampled result fits into the preallocated array
%         if length(conv_result) > num_windows
%             conv_result = conv_result(1:num_windows);
%         elseif length(conv_result) < num_windows
%             conv_result(num_windows) = 0; % Zero-pad if necessary
%         end
% 
%         conv_sums_all_trials(:, i) = conv_result;
%     end
% 
%     % Calculate the average number of spikes in each window across all trials
%     average_conv_sums = mean(conv_sums_all_trials, 2);
% 
%     % Calculate the firing rate (spikes per second) in each window
%     firing_rate = average_conv_sums / window_duration;
%     firing_rate = smoothdata(firing_rate, 'gaussian', 40);
% 
%     % Time axis for the windows
%     window_time = (0:num_windows-1) * step_size; % Center the windows on their time intervals
% 
%    
%     % Plot the average firing rate
%     figure();
%     plot(window_time, firing_rate, 'LineWidth', 2);
%     xlabel('Time (ms)');
%     ylabel('Average Firing Rate (spikes/s)');
%     xlim([0, 2000]);
%     ylim([0, 30]);
%     title(sprintf('Average Firing Rate with Gaussian Filter (Orientation %d)', stimulus(s)));
%     grid on;
% end

% % Define the window length and step size
% window_length = 200; % 200 ms
% step_size = 10; % 10 ms
% alpha = 1 / (window_length); % Alpha for the window function
% window_duration = window_length / 1000; % duration in seconds
% 
% % Create the window for convolution
% half_window = floor(window_length / 2);
% t = -half_window:half_window;
% conv_window = alpha^2 * t .* exp(-alpha * t);
% conv_window(t < 0) = 0; % Apply the [ ]_+ operation
% conv_window = conv_window / mean(conv_window); % Normalize
% 
% % Calculate the number of windows
% num_windows = floor(2000 / step_size);
% 
% % Preallocate array for storing convolved sums for all trials
% conv_sums_all_trials = zeros(num_windows, num_trial);
% 
% for s = 1:num_stimulus
%     % Convolve spikes with the window for each trial
%     for i = 1:num_trial
%         spike1 = spike_time(s, i, :);
%         spike1 = reshape(spike1, time_pt, 1);
%         
%         % Convolve the spike train with the custom window
%         conv_result = conv(spike1, conv_window, 'same');
% 
%         % Downsample the convolved result to match the step size
%         conv_result = conv_result(1:step_size:end);
%         
%         % Ensure the downsampled result fits into the preallocated array
%         if length(conv_result) > num_windows
%             conv_result = conv_result(1:num_windows);
%         elseif length(conv_result) < num_windows
%             conv_result(num_windows) = 0; % Zero-pad if necessary
%         end
% 
%         conv_sums_all_trials(:, i) = conv_result;
%     end
% 
%     % Calculate the average number of spikes in each window across all trials
%     average_conv_sums = mean(conv_sums_all_trials, 2);
% 
%     % Calculate the firing rate (spikes per second) in each window
%     firing_rate = average_conv_sums / window_duration;
% 
%     % Apply additional smoothing to the firing rate
%     firing_rate = smoothdata(firing_rate, 'gaussian', 40);
% 
%     % Time axis for the windows
%     window_time = (0:num_windows-1) * step_size; % Center the windows on their time intervals
% 
%     % Plot the average firing rate
%     figure();
%     plot(window_time, firing_rate, 'LineWidth', 2);
%     xlim([0, 2000]);
%     ylim([0, 30]);
%     xlabel('Time (ms)');
%     ylabel('Average Firing Rate (spikes/s)');
%     title(sprintf('Average Firing Rate with Alpha Function (Orientation %d)', stimulus(s)));
%     grid on;
% end



% % Define the window length and step size
% window_length = 200; % 200 ms
% step_size = 10; % 10 ms
% sigma = window_length / 2; % Sigma for Gaussian, so 2*sigma = 200 ms
% window_duration = window_length / 1000; % duration in seconds
% 
% % Create the Gaussian window for convolution
% half_window = floor(window_length / 2);
% t = 0:half_window;
% conv_window = exp(-t.^2 / (2 * sigma^2));
% 
% % Calculate the number of windows
% num_windows = floor(2000 / step_size);
% 
% % Preallocate array for storing convolved sums for all trials
% conv_sums_all_trials = zeros(num_windows, num_trial);
% 
% for s = 1:num_stimulus
%     % Convolve spikes with the window for each trial
%     for i = 1:num_trial
%         spike1 = spike_time(s, i, :);
%         spike1 = reshape(spike1, time_pt, 1);
%         
%         % Convolve the spike train with the Gaussian window
%         conv_result = conv(spike1, conv_window, 'same');
% 
%         % Downsample the convolved result to match the step size
%         conv_result = conv_result(1:step_size:end);
%         
%         % Ensure the downsampled result fits into the preallocated array
%         if length(conv_result) > num_windows
%             conv_result = conv_result(1:num_windows);
%         elseif length(conv_result) < num_windows
%             conv_result(num_windows) = 0; % Zero-pad if necessary
%         end
% 
%         conv_sums_all_trials(:, i) = conv_result;
%     end
% 
%     % Calculate the average number of spikes in each window across all trials
%     average_conv_sums = mean(conv_sums_all_trials, 2);
% 
%     % Calculate the firing rate (spikes per second) in each window
%     firing_rate = average_conv_sums / window_duration;
% 
%     % Time axis for the windows
%     window_time = (0:num_windows-1) * step_size; % Center the windows on their time intervals
% 
%    
%     % Plot the average firing rate
%     figure();
%     plot(window_time, firing_rate, 'LineWidth', 2);
%     xlabel('Time (ms)');
%     ylabel('Average Firing Rate (spikes/s)');
%     xlim([0, 2000]);
%     ylim([0, 30]);
%     title(sprintf('Average Firing Rate with Gaussian Filter (Positive)(Orientation %d)', stimulus(s)));
%     grid on;
% end


% % Define the window length and step size
% window_length = 200; % 200 ms
% 
% % Calculate the number of windows
% num_windows = floor(2000 / window_length);
% 
% % Preallocate array for storing spike counts for all trials
% spike_counts_all_trials = zeros(num_windows, num_trial);
% 
% for s = 1:num_stimulus
%     % Calculate spike counts in each bin for each trial
%     for i = 1:num_trial
%         spike1 = spike_time(s, i, :);
%         spike1 = reshape(spike1, time_pt, 1);
%         
%         % Bin the spikes in 200 ms windows
%         for w = 1:num_windows
%             bin_start = (w - 1) * window_length + 1;
%             bin_end = w * window_length;
%             spike_counts_all_trials(w, i) = sum(spike1(bin_start:bin_end));
%         end
%     end
% 
%     % Calculate the average spike count in each bin across all trials
%     average_spike_counts = sum(spike_counts_all_trials, 2);
% 
%     % Time axis for the bins
%     bin_time = (0:num_windows-1) * window_length + window_length;
% 
%     % Plot the histogram of average spike counts
%     figure();
%     bar(bin_time, average_spike_counts, 'BarWidth', 1);
%     xlim([0, 2200]);
%     ylim([0, 30]);
%     xlabel('Time (ms)');
%     ylabel('Average Spike Count');
%     title(sprintf('Average Spike Count per 200 ms Bin (Orientation %d)', stimulus(s)));
%     grid on;
% end



% % Define the window length and step size
% window_length = 200; % 200 ms
% window_duration = window_length / 1000; % duration in seconds
% 
% % Calculate the number of windows
% num_windows = floor(time_pt / window_length);
% 
% % Preallocate array for storing firing rates for all trials
% firing_rates_all_trials = zeros(num_stimulus, num_trial);
% 
% for s = 1:num_stimulus
%     % Calculate spike counts in each bin for each trial
%     for i = 1:num_trial
%         spike1 = spike_time(s, i, :);
%         spike1 = reshape(spike1, time_pt, 1);
%         
%         % Bin the spikes in 200 ms windows
%         spike_counts = zeros(num_windows, 1);
%         for w = 1:num_windows
%             bin_start = (w - 1) * window_length + 1;
%             bin_end = min(w * window_length, time_pt); % Ensure bin_end does not exceed the time points
%             spike_counts(w) = sum(spike1(bin_start:bin_end));
%         end
% 
%         % Calculate the average firing rate for the trial
%         firing_rates_all_trials(s, i) = mean(spike_counts) / window_duration;
%     end
% end
% 
% % Calculate the average firing rate for each stimulus orientation
% average_firing_rates = mean(firing_rates_all_trials, 2);
% 
% % Extract the first 12 points for the scatter plot and smoothing spline
% num_points = 13;
% stimulus_subset = stimulus(2:num_points);
% average_firing_rates_subset = average_firing_rates(2:num_points);
% 
% % Plot the tuning curve with dots
% figure();
% scatter(stimulus_subset, average_firing_rates_subset, 'filled');
% hold on;
% 
% % Fit a smoothing spline to the data with a low smoothing parameter
% smooth_curve = fit(stimulus_subset', average_firing_rates_subset, 'smoothingspline', 'SmoothingParam', 0.01);
% 
% % Plot the smooth line
% plot(smooth_curve, 'r-');
% 
% xlabel('Stimulus Orientation');
% ylabel('Average Firing Rate (spikes/s)');
% title('Tuning Curve');
% grid on;
% hold off;


% Define the total recording duration
recording_duration = 2; % 2 seconds

% Number of orientations
num_orientations = 13; % assuming there are 12 orientations based on the text

% Preallocate arrays for ISI, Fano factor, and CV calculations
all_isis = cell(num_stimulus, num_trial);
fano_factors = zeros(num_stimulus, 1);
cv_values = zeros(num_stimulus, 1);

for s = 2:num_stimulus
    spike_counts_per_trial = zeros(num_trial, 1); % for Fano factor
    isis_per_stimulus = []; % for CV calculation
    
    % Calculate spike counts and ISIs for each trial
    for i = 1:num_trial
        spike1 = spike_time(s, i, :);
        spike1 = reshape(spike1, time_pt, 1);
        
        % Calculate the total number of spikes in the trial
        total_spikes = sum(spike1);
        spike_counts_per_trial(i) = total_spikes; % store for Fano factor
        
        % Calculate ISIs for the trial
        spike_times = find(spike1); % get the spike times
        if length(spike_times) > 1
            isis = diff(spike_times) / 1000; % convert to seconds
            all_isis{s, i} = isis; % store ISIs
            isis_per_stimulus = [isis_per_stimulus; isis]; % accumulate ISIs for the stimulus
        end
    end
    
    % Calculate Fano factor for the stimulus
    mean_spike_count = mean(spike_counts_per_trial);
    variance_spike_count = var(spike_counts_per_trial);
    fano_factors(s) = variance_spike_count / mean_spike_count;
    
    % Calculate CV for the stimulus
    if ~isempty(isis_per_stimulus)
        mean_isi = mean(isis_per_stimulus);
        std_isi = std(isis_per_stimulus);
        cv_values(s) = std_isi / mean_isi;
    else
        cv_values(s) = NaN; % handle cases with no ISIs
    end
end

% Display Fano factors and CVs
disp('Fano Factors:');
disp(fano_factors);
disp('Coefficients of Variation (CV):');
disp(cv_values);

% Assume `stimulus` contains the stimulus orientations
% and it has at least `num_orientations` points
if length(stimulus) < num_orientations
    error('The number of stimuli is less than the number of orientations.');
end
stimulus_orientations = stimulus(2:num_orientations);

% Plot Fano Factor vs. Orientation
figure;
scatter(stimulus_orientations, fano_factors(2:num_orientations), 'filled');
xlabel('Stimulus Orientation');
ylabel('Fano Factor');
title('Fano Factor vs. Stimulus Orientation');
grid on;

% Plot CV vs. Orientation
figure;
scatter(stimulus_orientations, cv_values(2:num_orientations), 'filled');
xlabel('Stimulus Orientation');
ylabel('Coefficient of Variation (CV)');
title('Coefficient of Variation (CV) vs. Stimulus Orientation');
grid on;

% % Define parameters
% time_step = 1; % 1 ms time step in milliseconds
% max_lag = 500; % max lag for autocorrelation in milliseconds
% num_lags = max_lag / time_step; % number of lags for autocorrelation
% 
% % Calculate and plot autocorrelation for each stimulus
% for s = 2:num_stimulus
%     all_spike_trains = [];
%     
%     for i = 1:num_trial
%         spike1 = spike_time(s, i, :);
%         spike1 = reshape(spike1, time_pt, 1);
%         
%         % Create the spike train (binary vector indicating spike times)
%         spike_train = zeros(time_pt, 1);
%         spike_train(spike1 > 0) = 1;
%         
%         % Collect all spike trains for autocorrelation
%         all_spike_trains = [all_spike_trains, spike_train];
%     end
%     
%     % Average spike train across trials
%     avg_spike_train = mean(all_spike_trains, 2);
%     
%     % Calculate the autocorrelation
%     [autocorr_values, lags] = xcorr(avg_spike_train - mean(avg_spike_train), num_lags, 'coeff');
%     
%     % Convert lags to milliseconds
%     lags = lags * time_step;
%     
%     % Plot the autocorrelation
%     figure;
%     plot(lags, autocorr_values, 'b');
%     xlabel('Time (ms)');
%     ylabel('Spike Train Auto Correlation');
%     title(sprintf('Orientation = %d', stimulus(s)));
%     xlim([-max_lag, max_lag]);
%     ylim([0, 1]);
%     grid on;
% end
