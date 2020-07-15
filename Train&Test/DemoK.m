clear;
clc;
close all;
warning off;

addpath(genpath('./.'));
addpath(genpath('/home/jerry/caffe/')) ;

folders  = '19/Save';
filepath = dir(fullfile(folders, '*.caffemodel'));
caffe.set_mode_gpu();
caffe.set_device(0);

folder  = 'testK/blur';
filepaths = dir(fullfile(folder, '*.png'));

for j = 1 : length(filepath)
    disp(j)
    weights = fullfile(folders,filepath(j).name);
    model = '19K.prototxt';
    net = caffe.Net(model, weights,'test');

    for i = 1 : length(filepaths)
        image_blur = imread(fullfile(folder,filepaths(i).name));
        image_blur  = im2single(image_blur);

        image_blur_2 = imresize(image_blur, 1/2, 'bicubic');
        image_blur_4 = imresize(image_blur, 1/4, 'bicubic');
        image_blur_8 = imresize(image_blur, 1/8, 'bicubic');
        
        net.blobs('data_8').reshape([size(image_blur_8,1) size(image_blur_8,2), size(image_blur_8,3), 1]);
        net.reshape();
        
        net.blobs('data_4').reshape([size(image_blur_4,1) size(image_blur_4,2), size(image_blur_4,3), 1]);
        net.reshape();
        
        net.blobs('data_2').reshape([size(image_blur_2,1) size(image_blur_2,2), size(image_blur_2,3), 1]);
        net.reshape();
        
        net.blobs('data_1').reshape([size(image_blur,1) size(image_blur,2), size(image_blur,3), 1]);
        net.reshape();
        
        res = net.forward({image_blur_8, image_blur_4, image_blur_2, image_blur});
        
        result = res{1};
        imwrite(im2uint8(result), ['evaluation_code/result/', filepaths(i).name])       
    end

%     psnr = start_eval_image();
%     cd ..
%     fid = fopen(fullfile(folders,'dataK.txt'),'a');
%     fprintf(fid, weights);
%     fprintf(fid, '\t%f\n',psnr);
%     caffe.reset_all()
end
% fclose(fid);
