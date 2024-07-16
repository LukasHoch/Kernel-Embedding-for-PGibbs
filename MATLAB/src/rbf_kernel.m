function K = rbf_kernel(X, V, W, P_Samples, M)
% RBF_KERNEL Compute kernel matrix.
    sigma_mult = [1.5 5 5 1];

    X_stacked = reshape(X(:,:,1:M), [], M);

    X_dis = zeros(M);
    for n = 1:M
        X_rep = repmat(X_stacked(:,n), 1, M);
        X_dis(n,:) = vecnorm(X_stacked - X_rep);
    end
    
    sigma = sigma_mult(1) * sqrt(0.5 * median(X_dis, 'all'));

    K_X = zeros(M);

    for k = 1:size(X_stacked, 1)
        K_X = K_X + (repmat(X_stacked(k, :), [M, 1]) - repmat(X_stacked(k, :)', [1, M])).^2;
    end

    K_X = exp(-K_X/(2*sigma^2));

    %Processing Noise Kernel
    V_stacked = reshape(V(:,:,1:M), [], M);

    V_dis = zeros(M);
    for n = 1:M
        V_rep = repmat(V_stacked(:,n), 1, M);
        V_dis(n,:) = vecnorm(V_stacked - V_rep);
    end
    
    sigma = sigma_mult(2) * sqrt(0.5 * median(V_dis, 'all'));

    K_V = zeros(M);

    for k = 1:size(V_stacked, 1)
        K_V = K_V + (repmat(V_stacked(k, :), [M, 1]) - repmat(V_stacked(k, :)', [1, M])).^2;
    end

    K_V = exp(-K_V/(2*sigma^2));


    %Measurement Noise Kernel
    W_stacked = reshape(W(:,:,1:M), [], M);

    W_dis = zeros(M);
    for n = 1:M
        W_rep = repmat(W_stacked(:,n), 1, M);
        W_dis(n,:) = vecnorm(W_stacked - W_rep);
    end
    
    sigma = sigma_mult(3) * sqrt(0.5 * median(W_dis, 'all'));
    %sigma = 1;

    K_W = zeros(M);

    for k = 1:size(W_stacked, 1)
        K_W = K_W + (repmat(W_stacked(k, :), [M, 1]) - repmat(W_stacked(k, :)', [1, M])).^2;
    end

    K_W = exp(-K_W/(2*sigma^2));

    N = size(P_Samples{1}.A, 1) * size(P_Samples{1}.A, 2);

    A_stacked = zeros(N, M);
    for n = 1:M
        A_stacked(:,n) = reshape(P_Samples{n}.A, N, 1);
    end

    A_dis = zeros(M);
    for n = 1:M
        A_rep = repmat(A_stacked(:,n), 1, M);
        A_dis(n,:) = vecnorm(A_stacked - A_rep);
    end
    
    sigma = sigma_mult(4) * sqrt(0.5 * median(A_dis, 'all'));

    K_A = zeros(M);

    for k = 1:size(A_stacked, 1)
        K_A = K_A + (repmat(A_stacked(k, :), [M, 1]) - repmat(A_stacked(k, :)', [1, M])).^2;
    end

    K_A = exp(-K_A/(2*sigma^2));


    K = K_X .* K_V .* K_W .* K_A;
    %K = K_X .* K_A;

    % K = (K_X + K_V + K_W + K_A) / 4;

    %K = 0.3 * K_X + 0.2 * K_V + 0.1 * K_W + 0.4 * K_A;

    % beta = [size(X_stacked, 1), size(V_stacked, 1), size(W_stacked, 1), size(A_stacked, 1)];
    % beta = beta / sum(beta);
    % K = beta(1) * K_X + beta(2) * K_V + beta(3) * K_W + beta(4) * K_A;
end