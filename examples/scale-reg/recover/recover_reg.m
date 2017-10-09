addpath(genpath(pwd));
rst_dir = '/media/data1/zk/TIP2016/results/tip_sk1491_sk1491';
sav_dir = './part';
if ~exist(sav_dir, 'dir')
   mkdir(sav_dir); 
end
items = dir(fullfile(rst_dir, '*_reg.mat'));
items = {items.name};
for i=1:2%length(items)
    fn = items{i}(1:end-8);
    sk = load(fullfile(rst_dir, [fn, '.mat'])); 
    sk = sk.sk;
    reg = load(fullfile(rst_dir, [fn, '_reg.mat'])); 
    reg = reg.reg;
    [parts, scores, sk_nms, scale] = reconstruct(sk, reg);
    segmentation = seg_score(logical(sum(parts, 3)), sk_nms);

    [h, w, n_parts] = size(parts);
    seg = zeros([h, w, 3], 'uint8');
    part3 = zeros(size(seg));
    for p = 1:n_parts
       part1 = parts(:,:,p);
       part3(:,:,1) = part1 * randperm(255, 1);
       part3(:,:,2) = part1 * randperm(255, 1);
       part3(:,:,3) = part1 * randperm(255, 1);
       seg = seg + uint8(part3); 
    end
    save(fullfile(sav_dir, [fn, '_parts.mat']), 'parts', 'scores');
    save(fullfile(sav_dir, [fn, '_scale.mat']), 'scale');
    save(fullfile(sav_dir, [fn, '_seg.mat']), 'segmentation');
    imwrite(segmentation, fullfile(sav_dir, [fn, '_seg_map.png']));
    imwrite(seg, fullfile(sav_dir, [fn, '_seg_color.png']));
    disp(['recover_reg: ' ,num2str(i), ' of ', num2str(length(items)),' saved at ', sav_dir ,' ,totally ', num2str(size(parts,3)), 'parts']);
end
