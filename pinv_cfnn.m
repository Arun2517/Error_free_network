clc;
clear;
close all;

%% Load Dataset

load('breastmnist.mat');

%% Convert to double

XTrain = double(train_images);
XTest  = double(test_images);

YTrain = double(train_labels);
YTest  = double(test_labels);

%% Flatten Images

XTrain = reshape(XTrain,size(XTrain,1),[]);
XTest  = reshape(XTest,size(XTest,1),[]);

%% Normalize

XTrain = XTrain/255;
XTest  = XTest/255;

%% One-Hot Encoding

numClasses = length(unique(YTrain));

T = zeros(length(YTrain),numClasses);

for i = 1:length(YTrain)
    T(i,YTrain(i)+1) = 1;
end

%% Network Parameters

HiddenNeurons = 200;

rng(1);

InputWeight = randn(size(XTrain,2),HiddenNeurons);

Bias = randn(1,HiddenNeurons);

%% Hidden Layer

H = 1./(1+exp(-(XTrain*InputWeight + Bias)));

%% Output Weight using PINV

Beta = pinv(H)*T;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% TRAINING %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

TrainOutput = H*Beta;

[~,trainPred] = max(TrainOutput,[],2);

trainPred = trainPred-1;

trainAcc = mean(trainPred==YTrain)*100;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% TESTING %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

HT = 1./(1+exp(-(XTest*InputWeight + Bias)));

TestOutput = HT*Beta;

[~,testPred] = max(TestOutput,[],2);

testPred = testPred-1;

testAcc = mean(testPred==YTest)*100;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% RESULTS %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n');
fprintf('Training Accuracy : %.2f %%\n',trainAcc);
fprintf('Testing Accuracy  : %.2f %%\n',testAcc);

figure;
confusionchart(YTest,testPred);
title('Confusion Matrix');