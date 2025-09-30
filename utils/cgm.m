function X = cgm(X0, costGradOp, N_iter, alpha0, mu)
% CGM 使用共轭梯度法重构 3D 数据
%
%   X = cgm(X0, costGradOp, N_iter, alpha0, mu)
%
%   输入参数：
%       X0        - 重构初始猜测，3D 数组
%       costGradOp- 函数句柄，计算 [cost, grad] = costGrad(x)
%                   已固定 vx 和 U_FFT 参数，可定义为： @(x) costGrad(x, vx, U_FFT)
%       N_iter    - 迭代次数
%       alpha0    - 初始步长
%       mu        - 正则化参数（L2 正则项）
%
%   输出参数：
%       X         - 重构结果，3D 数据
%
%   注意：本函数内部采用 GPU 加速，输入和输出均为 3D 数据。

% 初始化步长调整系数与搜索参数
c1t    = [0.9, 0.9, 0.2, 0.2, 0.01, 0.01, 0.001, 0.001, 1e-4]; 
ls_iter = 15;      % 最大线搜索步数
c1_step = 5;       % 每 c1_step 次迭代更新一次系数
descent = 2;       % 1: 梯度下降，2: 共轭梯度

% 将数据转为 GPU 数组
X = gpuArray(X0);

% 初始化迭代变量
gamma     = 0;
gammacoef = gpuArray(zeros(N_iter, 1));
alcoef    = gpuArray(zeros(N_iter, 1));
alpha_cnt = 0;
fcost     = gpuArray(zeros(N_iter, 2));  % 存储每次线搜索信息
F_x       = gpuArray(zeros(N_iter, 1));    % 目标函数历史

% 初始成本与梯度（注意：costGradOp 返回的是不含正则项的梯度）
hWait = waitbar(0, 'Initializing forward computation...');
[~, cost, g] = costGradOp(X);
g = g + mu .* X;  % 加上正则项梯度
d  = -g;         % 初始下降方向
dp = d;
gp = g;

waitbar(0, hWait, 'Starting iterations...');
tic;  % 开始计时

for k = 1:N_iter
    % 根据当前迭代数选择步长调整系数
    c1 = c1t(min(floor((k - 1)/c1_step) + 1, length(c1t)));
    
    % 记录当前成本
    F_x(k) = cost;
    
    % 选择下降方向
    if descent == 1
        d = -g;
    elseif descent == 2
        if k > 1
            gamma = sum(g .* (g - gp), 'all') / sum(gp.^2, 'all');
        end
        % 限制 gamma 范围，防止数值不稳定
        gamma = min(max(gamma, 0), 10);
        d = -g + gamma * dp;
        dp = d;
        gp = g;
    end
    gammacoef(k) = gamma;
    
    % 线搜索确定步长 alpha
    alpha = alpha0;
    gsum  = abs(sum(-d .* g, 'all'));
    for ls_count = 1:ls_iter
        % 计算已用时间、平均每次迭代耗时及剩余时间估计（单位：小时/分钟）
        elapsed = toc;
        avg_time = elapsed / k;
        remaining_time = avg_time * (N_iter - k);
        hr = floor(remaining_time/3600);
        mn = floor(mod(remaining_time,3600)/60);
        % 更新进度条信息（包含线搜索步数和当前 alpha）
        waitbar(k/N_iter, hWait, sprintf(['Iteration %d/%d Cost: %.4g\n' ...
            'search: %d, alpha: %.3f Est.: %dh %dm'], ...
            k, N_iter, cost, ls_count, alpha, hr, mn));
        
        Xt = X + alpha * d;
        [~,cost_trial] = costGradOp(Xt);
        f1 = cost_trial;
        minimprove = cost + c1 * alpha * gsum;
        if f1 <= minimprove
            break;
        else
            alpha = alpha / 2;
            alpha_cnt = alpha_cnt + 1;
        end
    end
    alcoef(k) = alpha;
    fcost(k, :) = [f1, minimprove];
    
    % 更新 X，并在更新后显示迭代信息及预计剩余时间
    X = X + alpha * d;
    elapsed = toc;
    avg_time = elapsed / k;
    remaining_time = avg_time * (N_iter - k);
    hr = floor(remaining_time/3600);
    mn = floor(mod(remaining_time,3600)/60);
    msg = sprintf(['Iteration %d/%d Cost: %.4g\nUpdating X... ' ...
                   'Est: %dh %dm'], k, N_iter, f1, hr, mn);
    waitbar(k/N_iter, hWait, msg);
    
    % 重新计算成本与梯度
    [~, cost, g_new] = costGradOp(X);
    g = g_new + mu .* X;
    
    % 每 n_alpha 次更新alpha0
    n_alpha = 5;
    if mod(k, n_alpha) == 0 || k == N_iter
        if alpha_cnt == n_alpha * ls_iter
            alpha0 = alpha0 / 2;
        end
        alpha_cnt = 0;
    end
end

close(hWait);
X = gather(X);  % 将 GPU 数据转回 CPU
end
