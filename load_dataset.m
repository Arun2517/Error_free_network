clc
clear
close all

filename = "breastmnist.npz";

[XTrain,YTrain,XVal,YVal,XTest,YTest] = load_medmnist(filename);

disp("========== Dataset ==========")

fprintf("Training Images : %d\n",size(XTrain,1));
fprintf("Validation Images : %d\n",size(XVal,1));
fprintf("Testing Images : %d\n",size(XTest,1));

fprintf("Image Size : %d x %d\n",size(XTrain,2),size(XTrain,3));

disp("Training Labels")
disp(unique(YTrain))

disp("Testing Labels")
disp(unique(YTest))

%% 
figure

for i=1:16

    subplot(4,4,i)

    imagesc(squeeze(XTrain(i,:,:)))

    colormap gray
    axis image off

    title(num2str(YTrain(i)))

end

%%  
XTrain = reshape(XTrain,size(XTrain,1),[]);
XVal   = reshape(XVal,size(XVal,1),[]);
XTest  = reshape(XTest,size(XTest,1),[]);

fprintf("Flattened Size : %d\n",size(XTrain,2));
%% 
XTrain = XTrain/255;
XVal   = XVal/255;
XTest  = XTest/255;

