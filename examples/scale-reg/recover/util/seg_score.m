function [ score_seg ] = seg_score(seg_bin, sk_nms)
%   get segmentation score map from skeleton map and nms(sk)
%   Input:

%      seg_bin:  binary segmentation map
%      sk_nms:   skeleton map after nms

assert(length(size(sk_nms)) == 2);
[~, bw_id] = bwdist(sk_nms);
[y1, x1] = find(seg_bin);
score_seg = zeros(size(sk_nms));
for i = 1:length(y1)
  y2 = y1(i); x2 = x1(i);
  [y3, x3] = ind2sub(size(sk_nms), bw_id(y2, x2));
  score_seg(y2, x2) = sk_nms(y3, x3);
end

end

