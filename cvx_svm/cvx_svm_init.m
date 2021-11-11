function [model] = cvx_svm_init()
    model = [];
    model.C = 1;
    model.kernel = 'rbf';
    model.sigma = 1;
    model.order = 3;
    model.training_accuracy = 0;
    model.trained = 0;
    model.tol = 1e-10;
    model.X = [];
    model.Y = [];
end

