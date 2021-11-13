clear; 
% TODO: ADD model_name HERE: model_name = XXX;
addpath(genpath('utils/'));
model_name_ = 'resnet_v1_50';
% model_name_ = 'mobilenet_v1';
normalization = 1;
% CMC of 10 splits
all_CMC = [];
for split_count = 1:10
    split = strcat('split', num2str(split_count));
    model_name = strcat(split, '/', model_name_);
    flip = 1; % For PRID2011, always take the first cam as the probe set
    if flip ==1
        probeCAM = 1; galleryCAM = 2;
    else
        probeCAM = 2; galleryCAM = 1;
    end

    %% load features and corresponding IDs/CAMs of testdata
    feature_path_1 = strcat('../results/PRID2011/', model_name_, '/local/eight_part/features/', split, '/');
    feature_path_2 = strcat('../results/PRID2011/', model_name_, '/global/eight_part/features/', split, '/');
    testfile = dir(feature_path_1);
    fprintf(split);
    fprintf('\r\n');
    %% compute CMC
    filename = 'test.mat';
    fprintf('compute CMC of %s\n', filename);
    test_feature_1 = importdata(strcat(feature_path_1, filename));
    test_feature_1 = permute(test_feature_1, [2,3,1]);
    test_feature_2 = importdata(strcat(feature_path_2, filename));
    test_feature_2 = permute(test_feature_2, [2,3,1]);
    test_feature = [test_feature_1; test_feature_2];
    [fdim, npart, nsample] = size(test_feature);

    path = strcat('datasplits/PRID2011/', split, '/testID.mat');
    testID = importdata(path);
    path = strcat('datasplits/PRID2011/', split, '/testCAM.mat');
    testCAM = importdata(path);

    %% mean-pooling (average-pooling) on the data
    testID_unique = unique(testID);
    probe_feature = [];
    gallery_feature =[];
    for i = 1:length(testID_unique)
        probe_ind = find(testID==testID_unique(i) & testCAM==probeCAM);
        nprobe = length(probe_ind);
        gallery_ind = find(testID==testID_unique(i) & testCAM==galleryCAM);
        ngallery = length(gallery_ind);
        probeID(i) = testID_unique(i);
        galleryID(i) = testID_unique(i);
        % do max-pooling on the feature (generate a template feature for each
        % tracklet (normalization + max-pooling + normalization)
        if normalization==1
            tmp = normc(reshape(test_feature(:,:,probe_ind),fdim, [])) / sqrt(npart);
            tmp = max(reshape(tmp, fdim * npart, nprobe), [], 2);
%             tmp = tmp(1025:end);
            tmp = normc(tmp);
            probe_feature= [probe_feature tmp];

            tmp = normc(reshape(test_feature(:,:,gallery_ind),fdim, [])) / sqrt(npart);
            tmp = max(reshape(tmp, fdim * npart, ngallery), [], 2);
%             tmp = tmp(1025:end);
            tmp = normc(tmp);
            gallery_feature = [gallery_feature tmp];
            % probe_feature= [probe_feature normr(max(normr(test_feature(:, probe_ind)')))'];
            % gallery_feature = [gallery_feature normr(max(normr(test_feature(:, gallery_ind)')))'];
        else
            probe_feature= [probe_feature max(test_feature(:, probe_ind)')'];
            gallery_feature = [gallery_feature max(test_feature(:, gallery_ind)')'];
        end
    end

    %%
    dist = pdist2(probe_feature', gallery_feature', 'euclidean');
    n_probe = size(probe_feature, 2);
    n_gallery = size(gallery_feature, 2);

    %% initialize variables to record the re-id accuracy
    CMC = zeros(n_probe, n_gallery);

    %% compute CMC
    knn = 1; % number of expanded queries. knn = 1 yields best result
    for k = 1:n_probe
        % images with the same ID but different camera from the query
        good_index = find(galleryID == probeID(k))'; 
        score = dist(k,:);
        % sort database images according to Euclidean distance
        [~, index] = sort(score, 'ascend');  % ranking according to distances
        CMC(k, :) = compute_CMC(good_index, index);% compute CMC for single query
    %     fprintf('%d\n',k);
    end
    CMC = mean(CMC);
    all_CMC = [all_CMC; CMC];

    %% print result for each split
    fprintf('single query: r1 = %f, r5 = %f, r10 = %f, r20 = %f \r\n', CMC(1), CMC(5), CMC(10), CMC(20));
end

%% print final result
CMC = mean(all_CMC)*100;
fprintf('average all_CMC')
fprintf('\r\n');
fprintf('single query: r1 = %1f, r5 = %1f, r10 = %f, r20 = %f \r\n', CMC(1), CMC(5), CMC(10), CMC(20));
