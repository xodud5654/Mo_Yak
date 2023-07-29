function label = classification(data)

% load("net_checkpoint__6024__2023_07_28__11_27_44.mat");
% load("class_trained_network.mat");
load("trained_resnet.mat");
label = classify(trainedNetwork_3,data);
label = string(label);
end