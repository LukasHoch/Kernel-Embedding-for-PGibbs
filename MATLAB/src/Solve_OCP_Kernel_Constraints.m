function [U_opt, X_opt, Y_opt] = Solve_OCP_Kernel_Constraints(PG_samples, x_vec_0, v_vec, e_vec, H, K, phi, g, n_x, n_y, n_u, y_min, y_max, alpha, sigma_mult)

optimization_timer = tic;

x_vec_0 = x_vec_0(:,:,1:K);
v_vec = v_vec(:,:,1:K);

Kernel = rbf_kernel(x_vec_0, v_vec, e_vec, PG_samples, K, sigma_mult);
K_chol = chol(Kernel + 1e-7 * eye(K));
epsilon = (1 + sqrt(2 * log(1 / alpha))) * sqrt(1 / K);


N_constr = length(y_min(y_min ~= -inf)) + length(y_max(y_max ~= inf));

cvx_begin quiet
    variable U(n_u, H)
    variable gammaN(K, N_constr)
    variable tk(N_constr)
    variable g0(N_constr)

    expression X(n_x, H+1, K)
    expression Y(n_y, H, K)

    expression g_rkhs(K, N_constr)
    expression Eg_rkhs(1, N_constr)
    expression g_norm(N_constr)

    X(:, 1, :) = x_vec_0;

    for k = 1:K
        A = PG_samples{k}.A;
        f = @(x, u) A * phi(x, u);

        for t = 1:H
            X(:, t+1, k) = f(X(:, t, k), U(:, t)) + v_vec(:, t, k);
            Y(:, t, k) = g(X(:, t, k), U(:, t)) + e_vec(:, t, k);
        end
    end
    
    for t = 1:N_constr
        g_rkhs(:, t) = Kernel * gammaN(:, t);
        Eg_rkhs(t) = sum(g_rkhs(:, t)) / K;
        g_norm(t) = norm(K_chol * gammaN(:, t));
    end

    minimize( sum(U.^2) )

    subject to
        U >= -10;
        U <= 10;
        for t = 1:N_constr
            g0(t) + Eg_rkhs(t) + epsilon * g_norm(t) <= tk(t) * alpha;
        end

        cnt = 1;
        for t = 1:H
            if y_min(t) ~= -inf
                for k = 1:K
                    max((-1) * Y(:, t, k) + y_min(t) * ones(n_y, 1, 1) + tk(cnt), 0) <= g0(cnt) + g_rkhs(k, cnt)
                end
                cnt = cnt + 1;
            end

            if y_max(t) ~= inf
                for k = 1:K
                    max(Y(:, t, k) - y_max(t) * ones(n_y, 1, 1) + tk(cnt), 0) <= g0(cnt) + g_rkhs(k, cnt)
                end
                cnt = cnt + 1;
            end
        end
cvx_end

time_Kernel = toc(optimization_timer)

U_opt = U;
X_opt = X;
Y_opt = Y;

end