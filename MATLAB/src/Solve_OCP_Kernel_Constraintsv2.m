function [U_opt, X_opt, Y_opt] = Solve_OCP_Kernel_Constraintsv2(PG_samples, x_vec_0, v_vec, e_vec, H, K, phi, g, n_x, n_y, n_u, y_min, y_max, alpha, sigma_mult)

optimization_timer = tic;

x_vec_0 = x_vec_0(:,:,1:K);
v_vec = v_vec(:,:,1:K);

Kernel = rbf_kernel(x_vec_0, v_vec, e_vec, PG_samples, K, sigma_mult);
K_chol = chol(Kernel + 1e-7 * eye(K));
%epsilon = (1 + sqrt(2 * log(1 / alpha))) * sqrt(1 / K);

epsilon = BootstrapAmbiguity(Kernel, 1000, 0.95);

cvx_begin quiet
    variable U(n_u, H)
    variable gammaN(K)
    variable tk
    variable g0

    expression X(n_x, H+1, K)
    expression Y(n_y, H, K)

    expression g_rkhs(K)
    expression Eg_rkhs
    expression g_norm

    X(:, 1, :) = x_vec_0;

    for k = 1:K
        A = PG_samples{k}.A;
        f = @(x, u) A * phi(x, u);

        for t = 1:H
            X(:, t+1, k) = f(X(:, t, k), U(:, t)) + v_vec(:, t, k);
            Y(:, t, k) = g(X(:, t, k), U(:, t)) + e_vec(:, t, k);
        end
    end

    g_rkhs = Kernel * gammaN;
    Eg_rkhs = sum(g_rkhs) / K;
    g_norm = norm(K_chol * gammaN);

    minimize( sum(U.^2) )

    subject to
        U >= -10;
        U <= 10;
        g0 + Eg_rkhs + epsilon * g_norm <= tk * alpha;

        for t = 1:H
            if y_min(t) ~= -inf
                for k = 1:K
                    max((-1) * Y(:, t, k) + y_min(t) * ones(n_y, 1, 1) + tk, 0) <= g0 + g_rkhs(k)
                end
            end

            if y_max(t) ~= inf
                for k = 1:K
                    max(Y(:, t, k) - y_max(t) * ones(n_y, 1, 1) + tk, 0) <= g0 + g_rkhs(k)
                end
            end
        end
cvx_end

time_Kernel = toc(optimization_timer)

U_opt = U;
X_opt = X;
Y_opt = Y;

end