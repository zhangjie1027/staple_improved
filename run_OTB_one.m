close all;clear;

%% Read params.txt
% params = readParams('params.txt');
params.grayscale_sequence = false; % suppose that sequence is colour
params.hog_cell_size = 4;
params.fixed_area = 150 ^ 2; % standard area to which we resize the target
params.n_bins = 2 ^ 5; % number of bins for the color histograms (bg and fg models)
params.learning_rate_pwp = 0.04; % bg and fg color models learning rate
params.feature_type = 'fhog';
% params.feature_type = 'gray';
params.inner_padding = 0.2; % defines inner area used to sample colors from the foreground
params.output_sigma_factor = 1/16; % standard deviation for the desired translation filter output
params.lambda = 1e-3; % regularization weight
params.learning_rate_cf = 0.01; % HOG model learning rate
params.merge_factor = 0.3; % fixed interpolation factor - how to linearly combine the two responses
params.merge_method = 'const_factor';
params.den_per_channel = false;

%% scale related
params.scale_adaptation = true;
params.hog_scale_cell_size = 4; % Default DSST=4
params.learning_rate_scale = 0.025;
params.scale_sigma_factor = 1/4;
params.num_scales = 33;
params.scale_model_factor = 1.0;
params.scale_step = 1.02;
params.scale_model_max_area = 32 * 16;

%% debugging stuff
params.visualization = 0; % show output bbox on frame
params.visualization_dbg = 0; % show also per-pixel scores, desired response and filter output
params.visualization_apce = 0;

%% load video info
video = 'Girl2';
base_path='E:\DataSets\CFTracker\People';

start_frame = 1;

[img_files, pos, target_sz, ground_truth, img_path] =load_video_info(base_path,video);

params.bb_VOT = ground_truth;
%         region = params.bb_VOT(start_frame, :);
img_files(1:start_frame - 1) = [];
params.img_files = img_files;
params.img_path = img_path;
im = imread([img_path img_files{1}]);
% is a grayscale sequence ?
if (size(im, 3) == 1)
    params.grayscale_sequence = true;
end

% init_pos is the centre of the initial bounding box
params.init_pos = pos;
params.target_sz = target_sz;
[params, bg_area, fg_area, area_resize_factor] = initializeAllAreas(im, params);


% in runTracker we do not output anything
params.fout = 0;
% start the actual tracking
res = trackerMain(params, im, bg_area, fg_area, area_resize_factor);
positions = res.res;

% writematrix(positions, ['staple_' video '.txt']);

fps = res.fps;
[DP, OP, CLE] = compute_performance_measures(positions, ground_truth);
fprintf('%12s - DP (20px):% 1.3f, OP:%1.3f, CLE (pixcel):%.5g, fFPS:% 4.2f\n', video, DP, OP, CLE, fps);