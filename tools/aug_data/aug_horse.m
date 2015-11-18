scales=[0.8, 0.9, 1, 1.2, 1.35, 1.5];
mirror=[0,1];
orients=[0,90,180,270];
crop = 1:5;
horDir = '../horse/';
edgeDir=[horDir,'basic/edge/'];
skelDir=[horDir,'basic/skel/'];
trnImgDir=[horDir,'images/','train/'];
testImgDir=[horDir,'images/','test/'];
imgSubDir='./train/img_s'
imgs=dir([trnImgDir,'*.jpg']);
imgs={imgs.name};
nImgs=length(imgs);

for i=1:nImgs,imgs{i}=imgs{i}(1:end-4);end
train_list=fopen('train_pair.txt','w');
test_list=fopen('test.txt','w');

for i=1:nImgs
   skel = load([skelDir,imgs{i},'.mat']); skel = skel.skel;
   edge = load([edgeDir,imgs{i},'.mat']); edge = edge.edge;
   im=imread([trnImgDir,imgs{i},'.jpg']);
   siz=size(skel);
   for s=scales
      siz1=siz; siz1(1)=round(siz(1)*s); siz1(2)=round(siz(2)*s); siz1=siz1(1:2);
      im1=imresize(im,siz1);   skel1=imresize(single(skel),siz1,'bilinear');  edge1 = imresize(single(edge),siz1);
      edge1(edge1~=0)=1;  skel1(skel1~=0)=1; 
      skel1 = bwmorph(skel1,'thin');  edge1 = logical(bwmorph(edge1,'thin'));
      sym1 = bwdist(edge1);  %sym1 = sym1.*skel1;
      ids=find(~logical(sym1).*skel1);sym1(ids)=1;  sym1=sym1.*skel1;
      sym1 = uint8(sym1*2); assert(max(sym1(:))<=255); 
      
      [max_scale,max_ind] = max(sym1(:));  [y,x] = ind2sub(size(sym1),max_ind);
      imshow(im1);rectangle('position',[x-round(max_scale/2),y-round(max_scale/2),max_scale,max_scale]);

      sym1 = repmat(sym1,[1,1,3]);
      for o=orients
          im2 = imrotate(im1,o);  sym2 = imrotate(sym1,o);
          for c=crop
              switch c
                  case 1
                      im3 = im2(:,20:end,:);
                      sym3 = sym2(:,20:end,:);
                  case 2
                      im3 = im2(:,1:end-20,:);
                      sym3 = sym2(:,1:end-20,:);
                  case 3
                      im3 = im2(20:end,:,:);
                      sym3 = sym2(20:end,:,:);
                  case 4
                      im3 = im2(1:end-20,:,:);
                      sym3 = sym2(1:end-20,:,:);
                  case 5
                      im3 = im2;
                      sym3 = sym2;
                  otherwise
                      error('bad crop type!');
              end
              for f=0:2
                  switch f
                      case 1
                        im4 = fliplr(im3); sym4 = fliplr(sym3);
                      case 2
                        im4 = flipud(im3); sym4 = flipud(sym3);
                      case 0
                        im4 = im3;  sym4 = sym3;   
                      otherwise
                          error('bad flip type.');
                  end
                 imgSubDir = ['train/img_scale',num2str(s),'/orient',num2str(o),'/crop',num2str(c),'/flip',num2str(f),'/'];
                 gitSubDIr = ['train/gt_scale',num2str(s),'/orient',num2str(o),'/crop',num2str(c),'/flip',num2str(f),'/'];
                 
                 if(~exist(imgSubDir,'dir')),mkdir(imgSubDir);end
                 if(~exist(gitSubDIr,'dir')),mkdir(gitSubDIr);end
                 
                 assert(nnz(size(im3)-size(sym3))==0);
                 
                 imwrite(im4,[imgSubDir,imgs{i},'.jpg']);
                 imwrite(sym4,[gitSubDIr,imgs{i},'.png']);
                 
                 fprintf(train_list,[imgSubDir,imgs{i},'.jpg ',gitSubDIr,imgs{i},'.png\r\n']);
              end
          end
      end
   end
   disp(['image ',num2str(i),' of ',num2str(nImgs)]);
end
fclose(train_list);
% write test list
testImgs=dir([testImgDir,'*.jpg']);
testImgs={testImgs.name};
nImgs=length(testImgs);
for i=1:nImgs,testImgs{i}=testImgs{i}(1:end-4);end
for i=1:nImgs
    disp(['test img ',testImgs{i},' of ',num2str(nImgs)]);
    fprintf(test_list,['test/',testImgs{i},'.jpg\r\n']);
end
fclose(test_list);
