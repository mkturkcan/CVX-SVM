function [output] = cvx_svm_predict(X,model,varargin)

K = cvx_svm_kernel(X,model.X,model);

if isempty(varargin)
    output = (double(K*((model.svm_beta).*(model.Y))+(model.svm_bias)>0)*2-1);
else
    output{1} = (double(K*((model.svm_beta).*(model.Y))+(model.svm_bias)>0)*2-1);
    output{2} = mean(output{1}==varargin{1});
end

end

