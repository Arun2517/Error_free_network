
%% PINV_AutoSearch_BreastMNIST.m
% Automatic search for Hidden Neurons, Ridge Lambda and Voting Weights
clc; clear; close all;

%% Load Dataset
load('breastmnist.mat');

XTrainImg = double(train_images);
XValImg   = double(val_images);
XTestImg  = double(test_images);

YTrain = double(train_labels);
YVal   = double(val_labels);
YTest  = double(test_labels);

%% Original Features
XTrain1 = reshape(XTrainImg,size(XTrainImg,1),[])/255;
XVal1   = reshape(XValImg,size(XValImg,1),[])/255;
XTest1  = reshape(XTestImg,size(XTestImg,1),[])/255;

%% Histogram Features
XTrain2 = histFeature(XTrainImg);
XVal2   = histFeature(XValImg);
XTest2  = histFeature(XTestImg);

%% One-hot targets
classes = length(unique(YTrain));
T = zeros(length(YTrain),classes);
for i=1:length(YTrain)
    T(i,YTrain(i)+1)=1;
end

HiddenList = [50 100 150 200];
LambdaList = [0 1e-4 1e-3 1e-2];
WeightList = [0.5 0.6 0.7 0.8];

bestVal = -inf;

for Hidden = HiddenList
    for lambda = LambdaList

        % Original
        rng(1);
        W1 = randn(size(XTrain1,2),Hidden);
        B1 = randn(1,Hidden);
        H1 = sigmoid(XTrain1*W1+B1);
        Beta1 = solveBeta(H1,T,lambda);

        % Histogram
        rng(100);
        W2 = randn(size(XTrain2,2),Hidden);
        B2 = randn(1,Hidden);
        H2 = sigmoid(XTrain2*W2+B2);
        Beta2 = solveBeta(H2,T,lambda);

        % Validation outputs
        O1 = sigmoid(XVal1*W1+B1)*Beta1;
        O2 = sigmoid(XVal2*W2+B2)*Beta2;

        for w = WeightList
            comb = w*O1 + (1-w)*O2;
            [~,p] = max(comb,[],2);
            p = p-1;
            acc = mean(p==YVal)*100;

            if acc>bestVal
                bestVal = acc;
                bestHidden = Hidden;
                bestLambda = lambda;
                bestWeight = w;

                BW1=W1; BB1=B1; BBeta1=Beta1;
                BW2=W2; BB2=B2; BBeta2=Beta2;
            end
        end
    end
end

%% Test best model
TO1 = sigmoid(XTest1*BW1+BB1)*BBeta1;
TO2 = sigmoid(XTest2*BW2+BB2)*BBeta2;

Combined = bestWeight*TO1 + (1-bestWeight)*TO2;

[~,Pred] = max(Combined,[],2);
Pred = Pred-1;

TestAcc = mean(Pred==YTest)*100;

TP=sum((Pred==1)&(YTest==1));
TN=sum((Pred==0)&(YTest==0));
FP=sum((Pred==1)&(YTest==0));
FN=sum((Pred==0)&(YTest==1));

Precision=TP/(TP+FP+eps);
Recall=TP/(TP+FN+eps);
F1=2*Precision*Recall/(Precision+Recall+eps);

fprintf('\n=========== BEST MODEL ===========\n');
fprintf('Hidden Neurons : %d\n',bestHidden);
fprintf('Lambda         : %g\n',bestLambda);
fprintf('Original Weight: %.2f\n',bestWeight);
fprintf('Histogram Wt   : %.2f\n',1-bestWeight);
fprintf('Validation Acc : %.2f %%\n',bestVal);
fprintf('Test Accuracy  : %.2f %%\n',TestAcc);
fprintf('Precision      : %.4f\n',Precision);
fprintf('Recall         : %.4f\n',Recall);
fprintf('F1 Score       : %.4f\n',F1);

figure;
confusionchart(YTest,Pred);
title('Best PINV Model');

%% Local functions
function X = histFeature(imgs)
n=size(imgs,1);
X=zeros(n,28*28);
for k=1:n
    img=uint8(squeeze(imgs(k,:,:)));
    e=histeq(img);
    X(k,:)=double(e(:))/255;
end
end

function y=sigmoid(x)
y=1./(1+exp(-x));
end

function Beta=solveBeta(H,T,lambda)
if lambda==0
    Beta=pinv(H)*T;
else
    Beta=(H'*H + lambda*eye(size(H,2)))\(H'*T);
end
end
