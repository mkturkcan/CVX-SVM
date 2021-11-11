%% CVX SVM Library Example
clear all; close all; clc;
addpath('cvx_svm');
%% Create a Basic Problem:
X = randn(5000,2);
noise_level = 0.1;
Y = double((X(:,1).^2+X(:,2).^2>1.00))*2-1;
X = zscore(X);

X_test = X(1001:end,:);
X = X(1:1000,:);
Y_test = Y(1001:end);
Y = Y(1:1000);
%% Find Results with the Built-in SVM as Baseline
kernel_setting = 'rbf'; % one of 'rbf', 'linear' and 'polynomial'
D = fitcsvm(X,Y,'KernelFunction', kernel_setting);
accuracy_baseline = mean(predict(D,X_test)==Y_test);
%% Train CVX-SVM
model = cvx_svm_init();
model.kernel = kernel_setting;
model = cvx_svm_train(X,Y,model);
disp(['Model Training Accuracy: ' num2str(100.*model.training_accuracy) '%']);
score = cvx_svm_predict(X_test,model,Y_test);
accuracy_cvx_svm_testing=score{2}; 
disp(['Model Testing Accuracy: ' num2str(100.*accuracy_cvx_svm_testing) '%']);
disp(['Baseline Implementation Testing Accuracy: ' num2str(100.*accuracy_baseline) '%']);