% TODO: ADD model_name HERE: model_name = XXX;
% model_name = 'resnet_v1_50';
model_name = 'mobilenet_v1';
addpath(genpath('utils/'));
normalization = 1;
max_pooling = 1; % if 1 use max_pooling else use mean_pooling

track_test = importdata('DukeMTMC_VideoReID_info/tracks_test_info.txt');
feature_path_1 = strcat('../results/DukeMTMC-VideoReID/', model_name, '/local/eight_part/features/');
feature_path_2 = strcat('../results/DukeMTMC-VideoReID/', model_name, '/global/eight_part/features/');
testfile = dir(feature_path_1);

box_feature_test_1 = [];
for j = 0:3
    path = strcat(feature_path_1, '/test_', num2str(j), '.mat');
    box_feature_test_j = importdata(path);
    box_feature_test_1 = [box_feature_test_1; box_feature_test_j];
    clearvars box_feature_test_j
end
box_feature_test_2 = [];
for j = 0:1
    path = strcat(feature_path_2, '/test_', num2str(j), '.mat');
    box_feature_test_j = importdata(path);
    box_feature_test_2 = [box_feature_test_2; box_feature_test_j];
    clearvars box_feature_test_j
end
%% permute python 3d data
test_feature_1 = permute(box_feature_test_1, [2,3,1]);
test_feature_2 = permute(box_feature_test_2, [2,3,1]);
test_feature = [test_feature_1; test_feature_2];
clearvars box_feature_test_1
clearvars box_feature_test_2
clearvars test_feature_1
clearvars test_feature_2
[fdim, npart, nsample] = size(test_feature);
featdim = fdim * npart;
if normalization==1
    test_feature = normc(reshape(test_feature,fdim, [])) / sqrt(npart);
end

% do pooling on each tracklet (mean/max-pooling)
video_feat_test = process_box_feat(test_feature, track_test, max_pooling, featdim); % video features for test (gallery+query)
clearvars test_feature
if normalization==1
    video_feat_test = normc(video_feat_test);
end

fprintf('prepare gallery & query data\n');
% prepare gallery & query data
query_IDX = importdata('DukeMTMC_VideoReID_info/query_IDX.txt')';  % load pre-defined query index
label_gallery = track_test(:, 3);
label_query = label_gallery(query_IDX);
cam_gallery = track_test(:, 4);
cam_query = cam_gallery(query_IDX);
feat_gallery = video_feat_test;
feat_query = video_feat_test(:, query_IDX);
clearvars video_feat_test
cam_amount = size(unique(cam_gallery),1); % how many unique cameras

fprintf('compute distances\n');
% compute distances
dist_eu = pdist2(feat_query', feat_gallery', 'euclidean');
clearvars feat_queryd
clearvars feat_gallery
% evaluate the results
[CMC_eu, map_eu, r1_pairwise, ap_pairwise] = evaluation_dukev(dist_eu', label_gallery, label_query, cam_gallery, cam_query, cam_amount);
clearvars dist_eu
CMC_eu = CMC_eu*100; 
map_eu = map_eu*100;
fprintf('Euclidean, r1 = %0.4f, r5 = %0.4f, r10 = %0.4f, r20 = %0.4f, mAP = %0.4f \n', CMC_eu(1), CMC_eu(5), CMC_eu(10), CMC_eu(20), map_eu);
