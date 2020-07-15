clear;
clc;
close all;
warning off;

addpath(genpath('./.'));
addpath(genpath('/home/jerry/caffe/')) ;

caffe.set_mode_gpu();
caffe.set_device(0);
weights = fullfile('22','10_solver_iter_150000.caffemodel');
model = '22.prototxt';
net = caffe.Net(model, weights,'test');

maindir = 'test';
subdir =  dir(maindir);
subsubdata = 'blur';
subsublabel = 'sharp';
count = 0;
 for num = 1 : length(subdir)
        if( isequal(subdir(num).name, '.') || ...
                isequal(subdir(num).name, '..') || ...
                ~subdir(num).isdir)
            continue;
        end
        filepaths_data = dir(fullfile(maindir, subdir(num).name, subsubdata, '*.png'));
        filepaths_label = dir(fullfile(maindir, subdir(num).name, subsublabel, '*.png'));
        for i = 1 : length(filepaths_data)
            disp(count)
            image_blur = imread(fullfile(maindir, subdir(num).name,subsubdata, filepaths_data(i).name));
            image_blur = im2single(image_blur);
            image_blur_2 = imresize(image_blur, 1/2, 'bicubic');
            image_blur_4 = imresize(image_blur, 1/4, 'bicubic');
            
            image_sharp = imread(fullfile(maindir, subdir(num).name,subsublabel, filepaths_label(i).name));
            image_sharp = im2single(image_sharp);
            
            %tic
            net.blobs('data_4').reshape([size(image_blur_4,1) size(image_blur_4,2), size(image_blur_4,3), 1]);
            net.reshape();
            
            net.blobs('data_2').reshape([size(image_blur_2,1) size(image_blur_2,2), size(image_blur_2,3), 1]);
            net.reshape();
            
            net.blobs('data_1').reshape([size(image_blur,1) size(image_blur,2), size(image_blur,3), 1]);
            net.reshape();
            
            res = net.forward({image_blur_4, image_blur_2, image_blur});
            %T = gputimeit(@()net.forward({image_blur_4, image_blur_2, image_blur}))
            %toc
            result = res{1};
            D = net.blobs('Sum1_2').get_data();
            for n = 1:size(D,3)
                a = abs(D(:,:,n));
                b = (a - min(a(:)))/(max(a(:))-min(a(:)));
                imshow([a, b])
            end
%             B = net.blobs('Cwhitechannel1_2').get_data();
            
            imwrite(b, 'finalD.png')
            imwrite(B, 'wB.png')
 
            imwrite(im2uint8(result), ['result/Our_without/', fullfile(subdir(num).name,subsublabel, filepaths_data(i).name)])
            
            PSNRCur = psnr(im2uint8(image_sharp), im2uint8(result));
            SSIMCur = ssim(im2uint8(image_sharp), im2uint8(result));

            count = count +1;
            PSNRs(count) = PSNRCur;
            SSIMs(count) = SSIMCur;
        end
end
disp(mean(PSNRs))
disp(mean(SSIMs))
caffe.reset_all()
