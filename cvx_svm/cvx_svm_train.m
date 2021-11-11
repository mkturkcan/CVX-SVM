function [model] = cvx_svm_train(X,Y,varargin)

if isempty(varargin)
    model = cvx_svm_init();
else
    model = varargin{1};
end

K = cvx_svm_kernel(X,X,model);
M = size(X,1);
cvx_begin
    cvx_solver sedumi
    cvx_precision best
    variable svm_beta(M);
    minimize (0.5.*quad_form(Y.*svm_beta,K) - ones(M,1)'*(svm_beta));
    subject to
        svm_beta >= 0;
        svm_beta <= model.C;
        Y'*(svm_beta) == 0;
cvx_end
model.X = X(svm_beta>model.tol,:);
model.Y = Y(svm_beta>model.tol);
model.svm_beta = svm_beta(svm_beta>model.tol);
K = cvx_svm_kernel(model.X,model.X,model);
model.svm_bias = mean(model.Y-K*(model.svm_beta.*model.Y));
K = cvx_svm_kernel(X,model.X,model);
model.training_accuracy = mean((double(K*(model.svm_beta.*model.Y)+model.svm_bias>0)*2-1)==Y);

end