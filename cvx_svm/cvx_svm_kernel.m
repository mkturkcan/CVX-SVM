function [K] = cvx_svm_kernel(X_a,X_b,model)
K=zeros(size(X_a,1),size(X_b,1));
for i=1:size(X_a,1)
    for j = 1:size(X_b,1)
        if strcmp(model.kernel,'linear')
            K(i,j) = (X_a(i,:)*X_b(j,:)');
        end
        if strcmp(model.kernel,'polynomial')
            K(i,j) = (X_a(i,:)*X_b(j,:)')^(model.order);
        end
        if strcmp(model.kernel,'rbf')
            K(i,j) = exp(-1*sum((X_a(i,:)-X_b(j,:)).^2)/(2*(model.sigma)^2));
        end
    end
end
end

