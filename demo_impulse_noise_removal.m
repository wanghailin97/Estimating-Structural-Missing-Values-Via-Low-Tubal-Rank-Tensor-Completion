%% demo 
% demo of impluse noise removal using LRTC_STR, with comparison to baselines LRTC-SNN, LRTC-TNN and TRPCA-TNN

clc;
clear;close all;
addpath(genpath('LRTC-STR'));
addpath(genpath('baselines'));
addpath(genpath('utils'));
%% observation 
X = double(imread('lena.jpg'));
X = X/255;
maxP = max(abs(X(:)));

[n1,n2,n3] = size(X);
dim = size(X);
p = 0.5;
m = round(prod(dim)*p);
sort = randperm(prod(dim));
omega = sort(1:m);
omega_c = sort(m+1:prod(dim));

M = X;
density = 0.5;
M(omega_c) = imnoise(X(omega_c),'salt & pepper',density);

%% parameters settings
opts.rho = 1.25;
opts.DEBUG = 1;

%% LRTC-STR (proposed method)
opts.lambda = 1e-2;
tic
Xhat1 = lrtc_str(M,omega,opts);
time(1) = toc;

%% LRTC-SNN
lambda = [15 15 1.5]; % recmmended by Lu2019-TPAMI
tic
Xhat2 = lrtc_snn(M,omega,lambda,opts);
time(2) = toc;

%% LRTC-TNN
tic
Xhat3 = lrtc_tnn(M,omega,opts);
time(3) = toc;

%% TRPCA-TNN
lambda = 1/sqrt(n3*max(n1,n2));
tic
Xhat4 = trpca_tnn(M,lambda,opts);
time(4) = toc;

%% results comparison

psnr(1) = PSNR(X,Xhat1,maxP);
psnr(2) = PSNR(X,Xhat2,maxP);
psnr(3) = PSNR(X,Xhat3,maxP);
psnr(4) = PSNR(X,Xhat4,maxP);

re(1) = norm(X(:)-Xhat1(:))/norm(X(:));
re(2) = norm(X(:)-Xhat2(:))/norm(X(:));
re(3) = norm(X(:)-Xhat3(:))/norm(X(:));
re(4) = norm(X(:)-Xhat4(:))/norm(X(:));

%% show results 
figure(1)
subplot 231; imshow(X); title('Original');
subplot 232; imshow(M); title('Noisy');
subplot 233; imshow(Xhat1); title(['LRTC-STR, PSNR:', num2str(psnr(1))]);
subplot 234; imshow(Xhat2); title(['LRTC-SNN, PSNR:', num2str(psnr(2))]);
subplot 235; imshow(Xhat3); title(['LRTC-TNN, PSNR:', num2str(psnr(3))]);
subplot 236; imshow(Xhat4); title(['TRPCA, PSNR:', num2str(psnr(4))]);
