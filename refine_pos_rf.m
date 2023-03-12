function [pos, max_response] = refine_pos_rf(p, im, old_pos, old_max_response,detector, bg_area, area_resize_factor, hann_window, hf_num, hf_den,bg_hist, fg_hist)
    [bboxes, ~, labels] = detect(detector, im);
    idx = labels == 'person';
    bboxes = bboxes(idx, :);
    sample_pos = bboxes(:,[2,1]) + bboxes(:,[4,3])/2;
%     sample_target = bboxes(:, [4,3]);
    num = size(bboxes,1);
    max_responses = zeros(1,num);
    for i = 1 : num
        temp_pos = sample_pos(i,:);
        im_patch_cf = getSubwindow(im, temp_pos, p.norm_bg_area, bg_area);
        pwp_search_area = round(p.norm_pwp_search_area / area_resize_factor);
        % extract patch of size pwp_search_area and resize to norm_pwp_search_area
        im_patch_pwp = getSubwindow(im, temp_pos, p.norm_pwp_search_area, pwp_search_area);
        % compute feature map
        xt = getFeatureMap(im_patch_cf, p.feature_type, p.cf_response_size, p.hog_cell_size);
        % apply Hann window
        xt_windowed = bsxfun(@times, hann_window, xt);
        % compute FFT
        xtf = fft2(xt_windowed);
        % Correlation between filter and test patch gives the response
        % Solve diagonal system per pixel.
        if p.den_per_channel
            hf = hf_num ./ (hf_den + p.lambda);
        else
            hf = bsxfun(@rdivide, hf_num, sum(hf_den, 3) + p.lambda);
        end

        response_cf = ensure_real(ifft2(sum(conj(hf) .* xtf, 3)));

        % Crop square search region (in feature pixels).
        response_cf = cropFilterResponse(response_cf, ...
        floor_odd(p.norm_delta_area / p.hog_cell_size));

        if p.hog_cell_size > 1
            % Scale up to match center likelihood resolution.
            response_cf = mexResize(response_cf, p.norm_delta_area, 'auto');
        end

        [likelihood_map] = getColourMap(im_patch_pwp, bg_hist, fg_hist, p.n_bins, p.grayscale_sequence);
        % (TODO) in theory it should be at 0.5 (unseen colors shoud have max entropy)
        likelihood_map(isnan(likelihood_map)) = 0;

        % each pixel of response_pwp loosely represents the likelihood that
        % the target (of size norm_target_sz) is centred on it
        response_pwp = getCenterLikelihood(likelihood_map, p.norm_target_sz);

        %% ESTIMATION
        response = mergeResponses(response_cf, response_pwp, p.merge_factor, p.merge_method);
        max_responses(i) = max(response(:));
    end
    max_response = max(max_responses);
    tops = round(sample_pos(find(max_responses == max_response, 1),:));
    if max_response > 0.25
        pos = tops;
    else
        pos = old_pos;
        max_response = old_max_response;
    end
end

function y = floor_odd(x)
    y = 2 * floor((x - 1) / 2) + 1;
end

function y = ensure_real(x)
    assert(norm(imag(x(:))) <= 1e-5 * norm(real(x(:))));
    y = real(x);
end