function epsilon = BootstrapAmbiguity(Kernel, B, beta)

    K = size(Kernel, 1);
    MMD = zeros(B,1);
    for b = 1:B
        idx = datasample(1:K, K);
        K_x = sum(sum(Kernel));
        K_y = sum(sum(Kernel(idx, idx)));
        K_xy = sum(sum(Kernel(idx, :)));

        MMD(b) = 1/K^2 * (K_x + K_y - 2*K_xy);
    end

    MMD = sort(MMD);
    epsilon = MMD(ceil(beta * B));
end