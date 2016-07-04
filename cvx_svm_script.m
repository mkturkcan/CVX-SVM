%% Clean Up The Workspace
clear all; close all; clc;
%% Set Parameters:
C = 1;
%% Create a Basic Problem:
X = randn(1000,2);
Y = double((X(:,1)+1.0*randn(1000,1)>X(:,2)))*2-1;
X = zscore(X);

X_test = X(101:end,:);
X = X(1:100,:);
Y_test = Y(101:end);
Y = Y(1:100);
%% Build the Kernel Matrix
M=size(X,1);
K = zeros(M,M);
for i = 1:M
    for j = 1:M
%         K(i,j) = (1 + (X(i,:)*X(j,:)').^9);
        K(i,j) = (X(i,:)*X(j,:)');
%         K(i,j) = exp(-1*sum((X(i,:)-X(j,:)).^2));
    end
end
%% Find Results with the Built-in SVM as Baseline
D = fitcsvm(X,Y);
accuracy_baseline = mean(predict(D,X_test)==Y_test);
%% Begin CVX-SVM Training:
cvx_begin
    cvx_precision best
    variable svm_beta(M);
    minimize (0.5.*quad_form(Y.*svm_beta,K) - ones(M,1)'*(svm_beta));
    subject to
        svm_beta >= 0;
        svm_beta <= C;
        Y'*(svm_beta) == 0;
cvx_end
svm_bias=mean(Y-K*(svm_beta.*Y));
accuracy_cvx_svm_training=mean((double(K*(svm_beta.*Y)-svm_bias>0)*2-1)==Y);
%% Begin CVX-SVM Testing:
for i=1:size(X_test,1)
    for j = 1:size(X,1)
        K(i,j) = (X_test(i,:)*X(j,:)');
%         K(i,j) = exp(-1*sum((X_test(i,:)-X(j,:)).^2));
    end
end

accuracy_cvx_svm_testing=mean((double(K*(svm_beta.*Y)-svm_bias>0)*2-1)==Y_test); 
