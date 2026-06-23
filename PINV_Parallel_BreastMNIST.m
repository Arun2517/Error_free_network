clc;
clear;
close all;

%% ===========================
% Load Dataset
%% ===========================

load('breastmnist.mat');

XTrainImg = double(train_images);
XTestImg  = double(test_images);

YTrain = double(train_labels);
YTest  = double(test_labels);

%% ===========================
% ORIGINAL FEATURES
%% ===========================

XTrain1 = reshape(XTrainImg,size(XTrainImg,1),[]);
XTest1  = reshape(XTestImg,size(XTestImg,1),[]);

XTrain1 = XTrain1/255;
XTest1  = XTest1/255;

%% ===========================
% HISTOGRAM EQUALIZATION FEATURES
%% ===========================

numTrain = size(XTrainImg,1);
numTest  = size(XTestImg,1);

XTrain2 = zeros(numTrain,28*28);
XTest2  = zeros(numTest,28*28);

for i = 1:numTrain

    img = uint8(squeeze(XTrainImg(i,:,:)));

    eqImg = histeq(img);

    eqImg = double(eqImg)/255;

    XTrain2(i,:) = eqImg(:)';

end

for i = 1:numTest

    img = uint8(squeeze(XTestImg(i,:,:)));

    eqImg = histeq(img);

    eqImg = double(eqImg)/255;

    XTest2(i,:) = eqImg(:)';

end
%% ===========================
% One-Hot Encoding
%% ===========================

numClasses = length(unique(YTrain));

T = zeros(length(YTrain),numClasses);

for i=1:length(YTrain)

    T(i,YTrain(i)+1)=1;

end

%% ===========================
% Network Parameters
%% ===========================

Hidden = 150;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% ORIGINAL PINV NETWORK %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1);
W1 = randn(size(XTrain1,2),Hidden);
B1 = randn(1,Hidden);

H1 = 1./(1+exp(-(XTrain1*W1+B1)));

Beta1 = pinv(H1)*T;

TrainOutput1 = H1*Beta1;

[~,TrainPred1] = max(TrainOutput1,[],2);

TrainPred1 = TrainPred1-1;

TrainAcc1 = mean(TrainPred1==YTrain)*100;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% HISTOGRAM PINV NETWORK %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(100);
W2 = randn(size(XTrain2,2),Hidden);
B2 = randn(1,Hidden);

H2 = 1./(1+exp(-(XTrain2*W2+B2)));

Beta2 = pinv(H2)*T;

TrainOutput2 = H2*Beta2;

[~,TrainPred2] = max(TrainOutput2,[],2);

TrainPred2 = TrainPred2-1;

TrainAcc2 = mean(TrainPred2==YTrain)*100;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% TESTING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

HT1 = 1./(1+exp(-(XTest1*W1+B1)));
TestOutput1 = HT1*Beta1;
[~,Pred1] = max(TestOutput1,[],2);
Pred1 = Pred1-1;

HT2 = 1./(1+exp(-(XTest2*W2+B2)));
TestOutput2 = HT2*Beta2;
[~,Pred2] = max(TestOutput2,[],2);
Pred2 = Pred2-1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% SOFT VOTING %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Equal Weight Soft Voting

CombinedOutput = TestOutput1 + TestOutput2;

[~,FinalPrediction] = max(CombinedOutput,[],2);

FinalPrediction = FinalPrediction - 1;

TestAccuracy = mean(FinalPrediction==YTest)*100;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% RESULTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


fprintf('\n');
fprintf('=====================================\n');
fprintf(' Original Network Train : %.2f %%\n',TrainAcc1);
fprintf(' Histogram Train        : %.2f %%\n',TrainAcc2);
fprintf(' Final Test Accuracy    : %.2f %%\n',TestAccuracy);
fprintf('=====================================\n');

%figure;
%confusionchart(YTest,FinalPrediction);
%title('Majority Voting Confusion Matrix');