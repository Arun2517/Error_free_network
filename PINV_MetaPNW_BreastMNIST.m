
%% PINV_MetaPNW_BreastMNIST.m
% Parallel PINV Neural Web with Meta-PINV Fusion
clc; clear; close all;

load('breastmnist.mat');

%% Data
XTr = double(train_images);
XV  = double(val_images);
XTe = double(test_images);

YTr = double(train_labels);
YV  = double(val_labels);
YTe = double(test_labels);

Hidden = 150;

%% ---------- Feature Extraction ----------
OrigTr = reshape(XTr,size(XTr,1),[])/255;
OrigV  = reshape(XV,size(XV,1),[])/255;
OrigTe = reshape(XTe,size(XTe,1),[])/255;

HistTr = histfeat(XTr);
HistV  = histfeat(XV);
HistTe = histfeat(XTe);

ClaheTr = clahfeat(XTr);
ClaheV  = clahfeat(XV);
ClaheTe = clahfeat(XTe);

%% One-hot
C = length(unique(YTr));
T = zeros(length(YTr),C);
for i=1:length(YTr)
    T(i,YTr(i)+1)=1;
end

%% Train three PINV branches
[W1,B1,Beta1] = trainPINV(OrigTr,T,Hidden,1);
[W2,B2,Beta2] = trainPINV(HistTr,T,Hidden,100);
[W3,B3,Beta3] = trainPINV(ClaheTr,T,Hidden,200);

%% Validation outputs
V1 = predictPINV(OrigV,W1,B1,Beta1);
V2 = predictPINV(HistV,W2,B2,Beta2);
V3 = predictPINV(ClaheV,W3,B3,Beta3);

MetaTrain = [V1 V2 V3];

%% Meta PINV
rng(500);
WM = randn(size(MetaTrain,2),30);
BM = randn(1,30);

HM = sigmoid(MetaTrain*WM+BM);
BetaM = pinv(HM)*fullTarget(YV,C);

%% Test outputs
T1 = predictPINV(OrigTe,W1,B1,Beta1);
T2 = predictPINV(HistTe,W2,B2,Beta2);
T3 = predictPINV(ClaheTe,W3,B3,Beta3);

MetaTest = [T1 T2 T3];

HMTest = sigmoid(MetaTest*WM+BM);
FinalOut = HMTest*BetaM;

[~,Pred] = max(FinalOut,[],2);
Pred = Pred-1;

acc = mean(Pred==YTe)*100;

TP=sum((Pred==1)&(YTe==1));
TN=sum((Pred==0)&(YTe==0));
FP=sum((Pred==1)&(YTe==0));
FN=sum((Pred==0)&(YTe==1));

prec=TP/(TP+FP+eps);
rec=TP/(TP+FN+eps);
f1=2*prec*rec/(prec+rec+eps);

fprintf('\n=========== Meta PINV Neural Web ===========\n');
fprintf('Accuracy : %.2f %%\n',acc);
fprintf('Precision: %.4f\n',prec);
fprintf('Recall   : %.4f\n',rec);
fprintf('F1 Score : %.4f\n',f1);

figure;
confusionchart(YTe,Pred);
title('Meta PINV Neural Web');

%% -------- Local Functions --------
function X=histfeat(imgs)
n=size(imgs,1); X=zeros(n,784);
for i=1:n
e=histeq(uint8(squeeze(imgs(i,:,:))));
X(i,:)=double(e(:))/255;
end
end

function X=clahfeat(imgs)
n=size(imgs,1); X=zeros(n,784);
for i=1:n
e=adapthisteq(uint8(squeeze(imgs(i,:,:))));
X(i,:)=double(e(:))/255;
end
end

function [W,B,Beta]=trainPINV(X,T,h,seed)
rng(seed);
W=randn(size(X,2),h);
B=randn(1,h);
H=1./(1+exp(-(X*W+B)));
Beta=pinv(H)*T;
end

function O=predictPINV(X,W,B,Beta)
H=1./(1+exp(-(X*W+B)));
O=H*Beta;
end

function y=sigmoid(x)
y=1./(1+exp(-x));
end

function T=fullTarget(Y,C)
T=zeros(length(Y),C);
for k=1:length(Y)
T(k,Y(k)+1)=1;
end
end
