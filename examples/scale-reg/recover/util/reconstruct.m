function [ parts, scores, sk_map_nms, scale_map ] = reconstruct( sk, reg )
  [c, h, w] = size(sk);
  assert(c == 5);
  sk_map = squeeze(sum(sk(2:end, :, :), 1)); sk_map(sk_map < 0.1) = 0;
  sk_map_nms1 = nms(sk_map); sk_map_nms1(sk_map_nms1<0.2) = 0;
  sk_map_nms = sk_map_nms1;
  sk_map_nms1 = single(bwmorph(sk_map_nms1, 'skel'));
  %subplot(1,2,1); imshow(sk_map_nms);
  
  sk_map_nms1 = padarray(sk_map_nms1, [3, 3], 'both');
  
  [y1, x1] = find(sk_map_nms1 ~= 0);
  for i=1:length(y1)
     
     mask = sk_map_nms1(y1(i)-3:y1(i)+3, x1(i)-3:x1(i)+3);
     conv = mask.*ones(7);
     if sum(conv(:)) >= 15
        sk_map_nms1(y1(i), x1(i)) = 0; 
     end
  end
  sk_map_nms1 = sk_map_nms1(4:end-3, 4:end-3);
  %subplot(1,2,2); imshow(sk_map_nms);
  % remove junction points
  
  sk_map_label = bwlabel(sk_map_nms1);
  % remove small parts
  for j = 1:length(unique(sk_map_label))
    idx = find(sk_map_label == j);
    if numel(idx) < 5
      sk_map_label(idx)=0;
      sk_map_nms1(idx) = 0;
    end 
  end
  if nargin == 1
    scale = sk2scale(sk, sk_map_nms1);
  else
    scale = reg2scale(sk, sk_map_nms1, reg);
  end
  parts_id = unique(sk_map_label);
  parts_id = parts_id(2:end);  %ignore background: 0
  parts = zeros(h, w, numel(parts_id));
  scores = zeros(numel(parts_id), 1);
  for j = 1:numel(parts_id)
    idx = sk_map_label == parts_id(j);
    scale1 = scale; 
    scale1(~idx) = 0;
    parts(:, :, j) = mexScale2seg(single(scale1));
    % scores(j) = mean(sk_map_nms(idx));
    scores(j) = sum(sk_map_nms1(idx));
  end
  scores = scores/max(scores(:));
  scale_map = scale;
end

function [scale] = sk2scale(sk, sk_nms)
  scale_bin = [5, 18, 43, 105];
  idx = find(sk_nms ~= 0);
  scale = zeros(size(sk_nms));
  for j=1:length(idx)
     [y1, x1] = ind2sub(size(sk_nms), idx(j));
     pred = sk(2:end, y1, x1); [~, max_pred_id] = max(pred);
     scale(y1, x1) = scale_bin(max_pred_id);
  end
end

function [scale] = reg2scale(sk, sk_nms, reg)
  
  rf = [14, 40, 92, 196]';
  idx = find(sk_nms ~= 0);
  scale = zeros(size(sk_nms));
  for j=1:length(idx)
     [y1, x1] = ind2sub(size(sk_nms), idx(j));
     pred = sk(2:end, y1, x1); [~, max_pred_id] = max(pred);
     reg_pred = (reg(:, y1, x1) + 1).*rf / 2;
     scale(y1, x1) = reg_pred(max_pred_id);
  end
end