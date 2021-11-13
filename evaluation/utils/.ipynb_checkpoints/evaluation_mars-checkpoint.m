function [CMC, map, r1_pairwise, ap_pairwise] = evaluation_mars(dist, label_gallery, label_query, cam_gallery, cam_query, cam_amount)

junk0 = find(label_gallery == -1);
ap = zeros(size(dist, 2), 1);
CMC = [];
r1_pairwise = zeros(size(dist, 2), cam_amount);% pairwise rank 1 precision  
ap_pairwise = zeros(size(dist, 2), cam_amount); % pairwise average precision
R5 = zeros(size(dist, 2), 1); % R5 for qualitative comparison
for k = 1:size(dist, 2)
    if k == 591
        a = 0
    end
    score = dist(:, k);
    q_label = label_query(k);
    q_cam = cam_query(k);
    pos = find(label_gallery == q_label);
    pos2 = cam_gallery(pos) ~= q_cam;
    good_image = pos(pos2);
    pos3 = cam_gallery(pos) == q_cam;
    junk = pos(pos3);
    junk_image = [junk0; junk];
    [~, index] = sort(score, 'ascend');
    [ap(k), CMC(:, k), R5(k)] = compute_AP(good_image, junk_image, index);
    ap_pairwise(k, :) = compute_AP_multiCam(good_image, junk, index, q_cam, cam_gallery, cam_amount); % compute pairwise AP for single query
    r1_pairwise(k, :) = compute_r1_multiCam(good_image, junk, index, q_cam, cam_gallery, cam_amount); % pairwise rank 1 precision with single query
end
CMC = sum(CMC, 2)./size(dist, 2);
CMC = CMC';
map = sum(ap)/length(ap);