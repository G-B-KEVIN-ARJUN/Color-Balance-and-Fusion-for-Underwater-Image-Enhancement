%%%%%%
% method described on Color Balance and Fusion for Underwater Image 
% Enhancement by Codruta O. Ancuti , Cosmin Ancuti, Christophe De 
% Vleeschouwer , and Philippe Bekaert 

close all;
clear all;
clc;

%%% Underwater White Balance %%%

%% Load the image and split channels. 

rgbImage=double(imread('test13.jpg'))/255;

grayImage = rgb2gray(rgbImage); %conversion of rgb image to grayscale image

%splitting the three channels (R,G and B) separately
Ir = rgbImage(:,:,1);
Ig = rgbImage(:,:,2);
Ib = rgbImage(:,:,3);

%calculating mean intensity of each channel for color compensation
Ir_mean = mean(Ir, 'all');
Ig_mean = mean(Ig, 'all');
Ib_mean = mean(Ib, 'all');

%% Color compensation
%Color compensation is only for red and blue channel as green channel isn't
%affected much by color casting

alpha = 0.1; %empirical constant

%color compensation for red channel
Irc = Ir + alpha*(Ig_mean - Ir_mean);

alpha = 0; % 0 does not compensates blue channel. 
%color compensation for blue channel
Ibc = Ib + alpha*(Ig_mean - Ib_mean);

%% White Balance

I = cat(3, Irc, Ig, Ibc); %concatenating the compensated red and blue channel with original green channel
I_lin = rgb2lin(I); %applying gamma correction to linear RGB values in image I so that I_lin is in the sRGB color space, which is suitable for display.
percentiles = 5;
illuminant = illumgray(I_lin,percentiles); %estimating the illumination of the scene in I_lin by assuming that the average color of the scene is gray
I_lin = chromadapt(I_lin,illuminant,'ColorSpace','linear-rgb'); %adjusting the color balance
Iwb = lin2rgb(I_lin);


%figure('name', 'Underwater White Balance');
%imshow([rgbImage, I, Iwb])

%%% Multi-Scale fusion. 

%% Gamma Correction
%First input of image fusion process (Gamma corrected image)
Igamma = imadjust(Iwb,[],[],2); %adjusting the image intensity values

%figure('name', 'Gamma Correction');
%imshow([Iwb, Igamma])


%% image sharpening
%Second input of image fusion process (Sharpened image)
sigma = 20;
Igauss = Iwb; 
N = 30;
%denoising and smoothening of images using a Gaussian filter
for iter=1: N
   Igauss =  imgaussfilt(Igauss,sigma); 
   Igauss = min(Iwb, Igauss);
end

gain = 1; %in the paper is not mentioned, but sometimes gain <1 is better. 
Norm = (Iwb-gain*Igauss);
%Norm
%sharpening of images by a linear normalization operator (here it is
%histogram stretching)
for n = 1:3
   Norm(:,:,n) = histeq(Norm(:,:,n)); 
end
Isharp = (Iwb + Norm)/2;

% figure('name', 'image sharpening');
% imshow([Iwb,Igauss,Norm, Isharp])


%% weights calculation
%the three weights are calculated separately for each of the two inputs

% Lapacian contrast weight 
Isharp_lab = rgb2lab(Isharp);
Igamma_lab = rgb2lab(Igamma);

% input1
R1 = double(Isharp_lab(:, :, 1)) / 255;
% calculate laplacian contrast weight
WC1 = sqrt((((Isharp(:,:,1)) - (R1)).^2 + ...
            ((Isharp(:,:,2)) - (R1)).^2 + ...
            ((Isharp(:,:,3)) - (R1)).^2) / 3);
% calculate the saliency weight
WS1 = saliency_detection(Isharp);
WS1 = WS1/max(WS1,[],'all');
% calculate the saturation weight

WSAT1 = sqrt(1/3*((Isharp(:,:,1)-R1).^2+(Isharp(:,:,2)-R1).^2+(Isharp(:,:,3)-R1).^2));


%figure('name', 'Image 1 weights');
%imshow([WC1 , WS1, WSAT1]);


% input2
R2 = double(Igamma_lab(:, :, 1)) / 255;
% calculate laplacian contrast weight
WC2 = sqrt((((Igamma(:,:,1)) - (R2)).^2 + ...
            ((Igamma(:,:,2)) - (R2)).^2 + ...
            ((Igamma(:,:,3)) - (R2)).^2) / 3);
% calculate the saliency weight
WS2 = saliency_detection(Igamma);
WS2 = WS2/max(WS2,[],'all');

% calculate the saturation weight
WSAT2 = sqrt(1/3*((Igamma(:,:,1)-R1).^2+(Igamma(:,:,2)-R1).^2+(Igamma(:,:,3)-R1).^2));

%figure('name', 'Image 2 weights');
%imshow([WC2 , WS2, WSAT2]);

% calculate the normalized weight
W1 = (WC1 + WS1 + WSAT1+0.1) ./ ...
     (WC1 + WS1 + WSAT1 + WC2 + WS2 + WSAT2+0.2);
W2 = (WC2 + WS2 + WSAT2+0.1) ./ ...
     (WC1 + WS1 + WSAT1 + WC2 + WS2 + WSAT2+0.2);
 
 
%% Naive fusion
%summation of product of weights with the respective input

R = W1.*Isharp+W2.*Igamma;
%figure('name', 'Naive Fusion');
%imshow([I, Iwb, Isharp, Igamma, R]);


%% Multi scale fusion.
%Makes use of Gaussian and Laplacian pyramid
 img1 = Isharp;
 img2 = Igamma;

% calculate the gaussian pyramid
level = 10;
Weight1 = gaussian_pyramid(W1, level);
Weight2 = gaussian_pyramid(W2, level);

% calculate the laplacian pyramid
% input1
R1 = laplacian_pyramid(Isharp(:, :, 1), level);
G1 = laplacian_pyramid(Isharp(:, :, 2), level);
B1 = laplacian_pyramid(Isharp(:, :, 3), level);
% input2
R2 = laplacian_pyramid(Igamma(:, :, 1), level);
G2 = laplacian_pyramid(Igamma(:, :, 2), level);
B2 = laplacian_pyramid(Igamma(:, :, 3), level);

% fusion
%at each level, the corresponding Gaussian and Laplacian pyramids are
%multiplied along with the weights
for k = 1 : level
   Rr{k} = Weight1{k} .* R1{k} + Weight2{k} .* R2{k};
   Rg{k} = Weight1{k} .* G1{k} + Weight2{k} .* G2{k};
   Rb{k} = Weight1{k} .* B1{k} + Weight2{k} .* B2{k};
end

% reconstruct & output
R = pyramid_reconstruct(Rr);
G = pyramid_reconstruct(Rg);
B = pyramid_reconstruct(Rb);
fusion = cat(3, R, G, B);

%printing of results
figure('name', 'Multi scale fusion');
imshow([I, fusion])