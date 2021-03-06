function video_feat = process_box_feat(box_feat, video_info, max_pooling, featdim)

nVideo = size(video_info, 1);
box_feat = reshape(box_feat, featdim, []);
video_feat = zeros(size(box_feat, 1), nVideo);
for n = 1:nVideo
    feature_set = box_feat(:, video_info(n, 1):video_info(n, 2));
    if max_pooling==1
        video_feat(:, n) = max(feature_set, [], 2); % max pooling 
    else
        video_feat(:, n) = mean(feature_set, 2); % avg pooling
    end
end

% %% normalize train and test features
% sum_val = sqrt(sum(video_feat.^2));
% for n = 1:size(video_feat, 1)
%     video_feat(n, :) = video_feat(n, :)./sum_val;
% end