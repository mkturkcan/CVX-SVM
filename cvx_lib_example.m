%% CVX SVM Library Example
clear all; close all; clc;
addpath('cvx_svm');
%% Create a Basic Problem:
X = randn(1000,2);
Y = double((X(:,1)+1.0*randn(1000,1)>X(:,2)))*2-1;
X = zscore(X);

X_test = X(101:end,:);
X = X(1:100,:);
Y_test = Y(101:end);
Y = Y(1:100);
%% Find Results with the Built-in SVM as Baseline
D = fitcsvm(X,Y);
accuracy_baseline = mean(predict(D,X_test)==Y_test);
%%
model = cvx_svm_init();
model = cvx_svm_train(X,Y,model);
score = cvx_svm_predict(X_test,model,Y_test);
accuracy_cvx_svm_testing=score{2}; 
