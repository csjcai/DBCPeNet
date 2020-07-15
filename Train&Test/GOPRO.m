clear;
clc;
close all;
warning off;

addpath(genpath('./.'));
addpath(genpath('/home/jerry/caffe/')) ;

folder  = '22';
filepaths = dir(fullfile(folder, '*.caffemodel'));

load('Demo3.mat')

for j = 1 : length(filepaths)
    disp(j)
    caffe.set_mode_gpu();
    caffe.set_device(0);
    weights = fullfile(folder,filepaths(j).name);
    model = '22.prototxt';
    net = caffe.Net(model, weights,'test');
    

    for i = 1 : size(data,4)
        disp(i)
        %tic
        image_blur = data(:, :, :, i); 
        image_blur_2 = data_2(:, :, :, i);
        image_blur_4 = data_4(:, :, :, i);
        image_sharp = label(:, :, :, i);

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
        [PSNRCur, SSIMCur] = Cal_PSNRSSIM(im2uint8(image_sharp),im2uint8(result),0,0);
        
        PSNRs(i) = PSNRCur;
        SSIMs(i) = SSIMCur;    
    end
    disp(size(SSIMs))
    fid = fopen(fullfile(folder,'data.txt'),'a');
    fprintf(fid, weights);
    fprintf(fid, '\t%f\n', mean(PSNRs));
    %fprintf(fid, '\t%f\n', mean(SSIMs));
    caffe.reset_all()
end
caffe.reset_all()
fclose(fid);
