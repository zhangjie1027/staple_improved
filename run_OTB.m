function run_OTB
    close all;clear;
    % RUN_TRACKER  is the external function of the tracker - does initialization and calls trackerMain

    %% Read params.txt
%     params = readParams('params.txt');
    params.grayscale_sequence = false; % suppose that sequence is colour
    params.hog_cell_size = 4;
    params.fixed_area = 150 ^ 2; % standard area to which we resize the target
    params.n_bins = 2 ^ 5; % number of bins for the color histograms (bg and fg models)
    params.learning_rate_pwp = 0.04; % bg and fg color models learning rate
    params.feature_type = 'fhog';
%     params.feature_type = 'gray';
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
    base_path='E:\DataSets\CFTracker\People';
    dirs = dir(base_path);
    videos = {dirs.name};
    videos(strcmp('.', videos) | strcmp('..', videos) | ...
                strcmp('anno', videos) | ~[dirs.isdir]) = [];
    
    videos(strcmpi('Jogging', videos)) = [];
    videos(end + 1:end + 2) = {'Jogging.1', 'Jogging.2'};
%     videos = {'Jogging'};
    start_frame = 1;
    all_fps = zeros(numel(videos), 1);
    all_DP = zeros(numel(videos), 1);
    all_OP = zeros(numel(videos), 1);
    all_CLE = zeros(numel(videos), 1);
    for k = 1:numel(videos)
        [img_files, pos, target_sz, ground_truth, video_path] =load_video_info(base_path,videos{k});
 
        img_path = video_path; %图片路径

        params.bb_VOT = ground_truth;
%         region = params.bb_VOT(start_frame, :);

        img_files(1:start_frame - 1) = [];

        im = imread([img_path img_files{1}]);
        % is a grayscale sequence ?
        if (size(im, 3) == 1)
            params.grayscale_sequence = true;
        end

        params.img_files = img_files;
        params.img_path = img_path;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         if (numel(region) == 8)
%             % polygon format
%             [cx, cy, w, h] = getAxisAlignedBB(region);
%         else
%             x = region(1);
%             y = region(2);
%             w = region(3);
%             h = region(4);
%             cx = x + w / 2;
%             cy = y + h / 2;
%         end

        % init_pos is the centre of the initial bounding box
        params.init_pos = pos;
        params.target_sz = target_sz;
        [params, bg_area, fg_area, area_resize_factor] = initializeAllAreas(im, params);

        if params.visualization
            params.videoPlayer = vision.VideoPlayer('Position', [100 100 [size(im, 2), size(im, 1)] + 30]);
        end

        % in runTracker we do not output anything
        params.fout = 0;
        % start the actual tracking
        res = trackerMain(params, im, bg_area, fg_area, area_resize_factor);
        positions = res.res;
%         positions = rect(:, [1, 2]);
%         precisions = precision_plot(positions, ground_truth, videos{s}, 0);
        writematrix(positions,['E:\SourceCode\Matlab\paper_result\staple_' videos{k} '.txt']);
        [all_DP(k), all_OP(k), all_CLE(k)] = ...
        compute_performance_measures(positions, ground_truth);
        all_fps(k) = res.fps;
        fprintf('%12s - DP (20px):% 1.3f, OP:%1.3f, CLE (pixcel):%.5g, fFPS:% 4.2f\n', videos{k}, all_DP(k), all_OP(k), all_CLE(k), all_fps(k));
        fclose('all');
end
% fprintf('\nCenter Location Error: %.5g pixels\tDistance Precision: %1.4f\tOverlap Precision: %1.4f\tSpeed: %4.2f fps\n', ...
%     mean(all_center_location_error), 100*mean(all_distance_precision), 100*mean(all_overlap_precision), mean(all_fps));
fprintf('Average - DP (20px):% 1.3f, OP:%1.3f, CLE (pixcel):%.5g, fFPS:% 4.2f\n', mean(all_DP), mean(all_OP), mean(all_CLE), mean(all_fps));
end
