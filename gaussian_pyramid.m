function out = gaussian_pyramid(img, level)
h = 1/16* [1, 4, 6, 4, 1]; %5-tap Gaussian filter
filt = h'*h; %making a 2D filter
out{1} = imfilter(img, filt, 'replicate', 'conv'); %filtering of the image
temp_img = img;
for i = 2 : level
    temp_img = temp_img(1 : 2 : end, 1 : 2 : end);
    out{i} = imfilter(temp_img, filt, 'replicate', 'conv');
end


