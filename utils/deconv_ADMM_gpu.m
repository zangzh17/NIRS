function [output] = deconv_ADMM_gpu(y, psf, para, varargin)
%% ADMM_LSI_deconv_3D_multichannel - 适用于计算显微镜多通道3D反卷积算法
% 基于交替方向乘子法(ADMM)求解3D反卷积问题，结合L1正则化和3D总变差(TV)正则化
%
% 数学表示: y_i = CH_i x, 其中:
%   y_i: 第i通道的2D测量数据
%   C: 裁剪算子
%   H_i: 第i通道的卷积算子
%   x: 待重建的3D物体
%
% 优化目标: xhat = argmin sum_i 0.5*|CH_i x-y_i|^2 + tau_l1*|x|_1 + tau_tv*TV(x) + 非负指示函数{x}
%
% 输入参数:
%   y - 多通道2D测量图像，大小为[rows, cols, num_channels]
%   psf - 多通道3D点扩散函数(PSF)，大小为[rows, cols, layers, num_channels]
%   para - 包含算法参数的结构体
%   varargin - 可选参数，用于自定义显示区域

%% 准备数据和参数
[rows, cols, layers, num_channels] = size(psf);
mu1 = para.mu1;       % ADMM惩罚参数1(数据项)
mu2 = para.mu2;       % ADMM惩罚参数2(TV项)
mu3 = para.mu3;       % ADMM惩罚参数3(非负约束项)
mu4 = para.mu4;       % ADMM惩罚参数4(L1正则化项)
tau_l1 = para.tau_l1; % L1正则化参数
tau_tv = para.tau_tv; % 总变差正则化参数
maxiter = para.maxiter;               % 最大迭代次数
termination_ratio = para.termination_ratio;  % 终止比率阈值
plateau_tolerence = para.plateau_tolerence;  % 允许的连续平台次数
color = para.color;                   % 可视化的颜色图
clip_min = para.clip_min;             % 解的最小值限制
clip_max = para.clip_max;             % 解的最大值限制

rtol = para.rtol;                     % 残差容差比率
mu_ratio = para.mu_ratio;             % 惩罚参数更新比率
% img_save_period = para.img_save_period;  % 图像保存周期
% img_save_path = para.img_save_path;      % 图像保存路径

% 处理可选参数(用于自定义显示区域)
if length(varargin) == 3
    custom_display_region_flag = true;
    display_row_start = varargin{1};
    display_col_start = varargin{2};
    display_width = varargin{3};
else
    custom_display_region_flag = false;
end

%% 定义算子
F3D = @(x) fftshift(fftn(ifftshift(x)));        % 3D傅里叶变换
Ft3D = @(x) fftshift(ifftn(ifftshift(x)));      % 3D傅里叶逆变换
clip = @(x, vmin, vmax) max(min(x, vmax), vmin);  % 值裁剪函数
VEC = @(x) x(:);                                   % 向量化函数

%% ADMM算法初始化
% 为每个通道初始化PSF和相关算子
Hs = cell(num_channels, 1);
Hs_conj = cell(num_channels, 1);
H = cell(num_channels, 1);
HT = cell(num_channels, 1);
HTH = gpuArray.zeros(rows, cols, layers, 'single');

for ch = 1:num_channels
    Hs{ch} = F3D(psf(:,:,:,ch));       % 每个通道PSF的傅里叶变换
    Hs_conj{ch} = conj(Hs{ch});               % PSF傅里叶变换的共轭
    H{ch} = @(x) real(Ft3D(F3D(x).*Hs{ch}));  % 每个通道的卷积算子
    HT{ch} = @(x) real(Ft3D(F3D(x).*Hs_conj{ch}));  % 每个通道卷积算子的伴随
    HTH = HTH + abs(Hs{ch}.*Hs_conj{ch});     % 累计所有通道的H^T*H
end

% 初始化主变量
xt = zeros(rows, cols, layers, 'single');  % 当前解(3D体积)

% 初始化辅助变量和对偶变量
vtp = gpuArray.zeros(rows, cols, layers, num_channels, 'single');  % 数据项的辅助变量(每个通道)
gamma1 = gpuArray.zeros(rows, cols, layers, num_channels, 'single');  % 对偶变量1(数据项，每个通道)
gamma3 = gpuArray.zeros(rows, cols, layers, 'single');  % 对偶变量3(非负约束项)
gamma4 = gpuArray.zeros(rows, cols, layers, 'single');  % 对偶变量4(L1正则化项)

% 为每个通道准备填充的测量数据
CTy = gpuArray.zeros(rows, cols, layers, num_channels, 'single');
for ch = 1:num_channels
    CTy(:,:,:,ch) = CT3D(y(:,:,ch), layers);  % 填充的测量数据(每个通道)
end

% 为总变差(TV)项准备算子和变量
PsiTPsi = generate_laplacian_3D(rows, cols, layers);  % 生成拉普拉斯算子(用于TV计算)
gamma2_1 = zeros(rows-1, cols, layers, 'single');  % 对偶变量2-1(x方向梯度)
gamma2_2 = zeros(rows, cols-1, layers, 'single');  % 对偶变量2-2(y方向梯度)
gamma2_3 = zeros(rows, cols, layers-1, 'single');  % 对偶变量2-3(z方向梯度)


% 定义梯度算子(用于TV计算)
PsiT = @(P1,P2,P3) cat(1,P1(1,:,:),diff(P1,1,1),-P1(end,:,:)) + ...
            cat(2,P2(:,1,:),diff(P2,1,2),-P2(:,end,:)) + ...
            cat(3,P3(:,:,1),diff(P3,1,3),-P3(:,:,end));  % 梯度的伴随算子
Psi = @(x)deal(-diff(x,1,1),-diff(x,1,2),-diff(x,1,3));  % 梯度算子

% 初始化TV相关变量
[ut1, ut2, ut3] = Psi(gpuArray.zeros(rows, cols, layers, 'single'));
Psixt1 = ut1;  % x的x方向梯度
Psixt2 = ut2;  % x的y方向梯度
Psixt3 = ut3;  % x的z方向梯度

% 预计算x更新步骤中的掩码(提高效率)
x_mask = 1./(mu1*HTH + mu2*PsiTPsi + mu3 + mu4);  % x更新步骤的分母(在频域中对角化)

% 为每个通道计算v_mask
v_mask = cell(num_channels, 1);
for ch = 1:num_channels
    v_mask{ch} = 1./(CT3D(gpuArray.ones(size(y(:,:,ch))), layers) + mu1);  % v更新步骤的分母(本身已对角化)
end

% 初始化迭代参数
iter = 0;
Hxtp = gpuArray.zeros(rows, cols, layers, num_channels, 'single');  % H(x)的缓存，每个通道

% 创建可视化窗口
if para.display_flag
    f1 = figure('rend','painters','pos',[50 50 1500 900]);
end

%% ADMM主循环
next_iteration = 1;
num_plateaus = 0;
while (next_iteration) && (iter <= maxiter) 
   iter = iter + 1;
   Hxt = Hxtp;  % 保存上一次的H(x)结果
   
   % 1. 更新辅助变量
   for ch = 1:num_channels
       vtp(:,:,:,ch) = v_mask{ch}.*(mu1*Hxt(:,:,:,ch) + gamma1(:,:,:,ch) + CTy(:,:,:,ch));  % 每个通道的数据项辅助变量
   end
   
   wtp = clip(xt+gamma3/mu3, clip_min, clip_max);          % 非负约束的辅助变量
   ztp = soft_thresh(gamma4/mu4 + xt,tau_l1/mu4);          % L1正则化的辅助变量(软阈值)
   [ut1,ut2,ut3] = soft_thres_3d(Psixt1+gamma2_1/mu2,...   % TV正则化的辅助变量(3D向量软阈值)
                               Psixt2+gamma2_2/mu2,...
                               Psixt3+gamma2_3/mu2,tau_tv/mu2);
   
   % 2. 更新主变量x
   % 计算x更新的分子各部分
   tmp_part1 = zeros(rows, cols, layers, 'single');    % 初始化数据项贡献
   
   % 累计所有通道的数据项贡献
   for ch = 1:num_channels
       tmp_part1 = tmp_part1 + mu1*HT{ch}(vtp(:,:,:,ch)) - HT{ch}(gamma1(:,:,:,ch));
   end
   
   tmp_part2 = mu2*PsiT(ut1-gamma2_1/mu2,...              % TV项贡献
                        ut2-gamma2_2/mu2,...
                        ut3-gamma2_3/mu2);
   tmp_part3 = mu3*wtp - gamma3;                          % 非负约束项贡献
   tmp_part4 = mu4*ztp - gamma4;                          % L1正则化项贡献
   xtp_numerator = tmp_part1 + tmp_part2 + tmp_part3 + tmp_part4;  % x更新分子
   
   % 在频域中高效求解x
   xtp = real(Ft3D(F3D(xtp_numerator).*x_mask));
   
   % 3. 更新对偶变量
   for ch = 1:num_channels
       Hxtp(:,:,:,ch) = H{ch}(xtp);                        % 计算每个通道的新H(x)
       gamma1(:,:,:,ch) = gamma1(:,:,:,ch) + mu1*(Hxtp(:,:,:,ch) - vtp(:,:,:,ch));  % 更新每个通道的对偶变量1
   end
   
   [Psixt1,Psixt2,Psixt3] = Psi(xtp);            % 计算新的梯度
   gamma2_1 = gamma2_1 + mu2*(Psixt1 - ut1);     % 更新对偶变量2-1
   gamma2_2 = gamma2_2 + mu2*(Psixt2 - ut2);     % 更新对偶变量2-2
   gamma2_3 = gamma2_3 + mu2*(Psixt3 - ut3);     % 更新对偶变量2-3
   
   gamma3 = gamma3 + mu3*(xtp - wtp);            % 更新对偶变量3
   gamma4 = gamma4 + mu4*(xtp - ztp);            % 更新对偶变量4
  
   % 4. 更新惩罚参数
   % 计算所有通道的原始残差和对偶残差
   primal_residual_mu1 = 0;
   dual_residual_mu1 = 0;
   for ch = 1:num_channels
       primal_residual_mu1 = primal_residual_mu1 + norm(VEC(Hxtp(:,:,:,ch)-vtp(:,:,:,ch)));
       dual_residual_mu1 = dual_residual_mu1 + mu1*norm(VEC(Hxt(:,:,:,ch) - Hxtp(:,:,:,ch)));
   end
   [mu1, mu1_update] = ADMM_update_param(mu1,rtol,mu_ratio,primal_residual_mu1,dual_residual_mu1);
   
   primal_residual_mu2_1 = norm(VEC(Psixt1-ut1));
   primal_residual_mu2_2 = norm(VEC(Psixt2-ut2));
   primal_residual_mu2_3 = norm(VEC(Psixt3-ut3));
   primal_residual_mu2 = norm([primal_residual_mu2_1,primal_residual_mu2_2,primal_residual_mu2_3]);
   [Psixt1_last, Psixt2_last, Psixt3_last] = Psi(xt);
   dual_residual_mu2_1 = mu2*norm(VEC(Psixt1_last - Psixt1));
   dual_residual_mu2_2 = mu2*norm(VEC(Psixt2_last - Psixt2));
   dual_residual_mu2_3 = mu2*norm(VEC(Psixt3_last - Psixt3));
   dual_residual_mu2 = norm([dual_residual_mu2_1,dual_residual_mu2_2,dual_residual_mu2_3]);
   [mu2, mu2_update] = ADMM_update_param(mu2,rtol,mu_ratio,primal_residual_mu2,dual_residual_mu2);
   
   primal_residual_mu3 = norm(VEC(xtp-wtp));
   dual_residual_mu3 = mu3*norm(VEC(xt - xtp));
   [mu3, mu3_update] = ADMM_update_param(mu3,rtol,mu_ratio,primal_residual_mu3,dual_residual_mu3);
    
   primal_residual_mu4 = norm(VEC(xtp-ztp));
   dual_residual_mu4 = mu4*norm(VEC(xt - xtp));
   [mu4, mu4_update] = ADMM_update_param(mu4,rtol,mu_ratio,primal_residual_mu4,dual_residual_mu4);
   
   % 检查是否有惩罚参数更新
   if mu1_update || mu2_update || mu3_update || mu4_update
       mu_update = 1;
   else
       mu_update = 0;
   end
   
   % 5. 检查终止条件: 1)连续平台 AND 2)惩罚参数未更新
   xt_last = xt;
   xt = xtp;
   evolution_ratio_of_the_iteration = compute_evolution_ratio(xt, xt_last);
   if (evolution_ratio_of_the_iteration <= termination_ratio) && (mu_update == 0)
       num_plateaus = num_plateaus + 1;
   else
       num_plateaus = 0;
   end
   
   if num_plateaus >= plateau_tolerence
       next_iteration = 0;
   end
   
   % 输出迭代信息
   disp(['Iter: ',num2str(iter), ', Change rate: ', num2str(evolution_ratio_of_the_iteration),', num_plateaus: ',num2str(num_plateaus)]);
   if next_iteration
      disp('Next iteration...'); 
   else
      disp('Finish...');
      % write_mat_to_tif(uint8(255*rescale(xt)),[img_save_path,'_final_iter_',num2str(iter),'.tif']);
   end
   
   % % 保存中间结果
   % if mod(iter, img_save_period) == 0
   %     write_mat_to_tif(uint8(255*rescale(xt)),[img_save_path,'_iter_',num2str(iter),'.tif']);
   % end

   % 如果惩罚参数更新，重新计算掩码
   if mu_update
       disp(['Penalty updated. mu1: ',num2str(round(mu1,3)), ', mu2: ',num2str(round(mu2,3)),...
           ', mu3: ',num2str(round(mu3,3)), ', mu4: ',num2str(round(mu4,3))]);
       x_mask = 1./(mu1*HTH + mu2*PsiTPsi + mu3 + mu4);  % 重新计算x更新的分母
       
       % 为每个通道重新计算v_mask
       for ch = 1:num_channels
           v_mask{ch} = 1./(CT3D(ones(size(y(:,:,ch))), layers) + mu1);
       end
   end
   
   % 可视化和评估
   if para.display_flag
       % 显示最大投影结果
       img2display = gather(max(xt,[],3));
       
       figure(f1);
       if custom_display_region_flag
           subplot(1,2,1),imagesc(img2display(display_row_start:display_row_start+display_width-1,...
               display_col_start:display_col_start+display_width-1)),...
               colormap(color);axis image;colorbar;title(iter);
       else
           subplot(1,2,1),imagesc(img2display),colormap(color);axis image;colorbar;title(iter);
       end
       
       % 计算各项损失
       dterm = 0;
       for ch = 1:num_channels
           residual = abs(crop3d(H{ch}(xt))-y(:,:,ch));
           dterm = dterm + 0.5*sum(residual(:).^2);      % 所有通道的数据保真度项
       end
       
       [tv_x,tv_y, tv_z] = Psi(xt);
       tv_x = cat(1,tv_x, gpuArray.zeros(1,cols,layers,'single'));
       tv_y = cat(2,tv_y, gpuArray.zeros(rows,1,layers,'single'));
       tv_z = cat(3,tv_z, gpuArray.zeros(rows,cols,1,'single'));
       tvterm = tau_tv*sum(sqrt(tv_x(:).^2 + tv_y(:).^2 + tv_z(:).^2)); % TV正则化项
       l1term = tau_l1*sum(abs(xt(:)));                   % L1正则化项
       cost = dterm+tvterm+l1term;                        % 总损失
       
       % 绘制损失曲线
       subplot(1,2,2),plot(iter,log10(gather(cost)),'bo'),grid on,hold on;...
                plot(iter,log10(gather(dterm)),'ro'),hold on;...
                plot(iter,log10(gather(tvterm)),'go'),hold on;...
                plot(iter,log10(gather(l1term)),'mo'),hold on;...
                title('B: Loss; R: fidelity; G: TV; P: L1 (Log scale)');

       drawnow; 

       % 输出各项损失值
       loss_cvy = 0;
       loss_mu1 = 0;
       for ch = 1:num_channels
           loss_cvy_ch = crop3d(vtp(:,:,:,ch)) - y(:,:,ch);
           loss_cvy = loss_cvy + 0.5 * sum(VEC(loss_cvy_ch.^2));
           
           loss_mu1_ch = Hxtp(:,:,:,ch) - vtp(:,:,:,ch) + gamma1(:,:,:,ch)/mu1;
           loss_mu1 = loss_mu1 + 0.5 * mu1 * sum(VEC(loss_mu1_ch.^2));
       end
       
       loss_tau_tv = tau_tv * ( sum(abs(VEC(ut1))) + sum(abs(VEC(ut2))) + sum(abs(VEC(ut3))));
       loss_tau_l1 = tau_l1 * sum(abs(VEC(ztp)));
       
       loss_mu2_1 = Psixt1 - ut1 + gamma2_1/mu2;
       loss_mu2_2 = Psixt2 - ut2 + gamma2_2/mu2;
       loss_mu2_3 = Psixt3 - ut3 + gamma2_3/mu2;
       loss_mu2 = 0.5 * mu2 * (sum(VEC(loss_mu2_1.^2)) + sum(VEC(loss_mu2_2.^2)) + sum(VEC(loss_mu2_3.^2)));
       
       loss_mu3 = xtp - wtp + gamma3/mu3;
       loss_mu3 = 0.5 * mu3 * sum(VEC(loss_mu3.^2));
       
       loss_mu4 = xtp - ztp + gamma4/mu4;
       loss_mu4 = 0.5 * mu4 * sum(VEC(loss_mu4.^2));
       
       loss_tot = loss_cvy + loss_tau_tv + loss_tau_l1 + loss_mu1 + loss_mu2 + loss_mu3 + loss_mu4; 
       disp(['Iter: ',num2str(iter),', Data Err: ',num2str(round(loss_cvy,3)), ', TV: ',num2str(round(loss_tau_tv,3))...
           , ', L1: ',num2str(round(loss_tau_l1,3))...
           ,', mu1: ',  num2str(round(loss_mu1,3)) ,', mu2: ',  num2str(round(loss_mu2,3))...
           , ', mu3: ',  num2str(round(loss_mu3,3)), ', mu4: ',  num2str(round(loss_mu4,3))...
           , ', Total Loss: ',  num2str(round(loss_tot,3))]);
   end
end

output = gather(xt);  % 返回最终重建结果
end

%% 辅助函数

% 更新ADMM惩罚参数
function [mu_out, mu_update] = ADMM_update_param(mu, resid_tol, mu_ratio, r, s)
    % 根据原始残差r和对偶残差s的比例调整惩罚参数mu
    if r > resid_tol*s      % 如果原始残差太大
        mu_out = mu*mu_ratio;  % 增大惩罚参数
        mu_update = 1;
    elseif r*resid_tol < s  % 如果对偶残差太大
        mu_out = mu/mu_ratio;  % 减小惩罚参数
        mu_update = -1;
    else                    % 如果残差平衡
        mu_out = mu;        % 保持惩罚参数不变
        mu_update = 0;
    end
end

% 修改后的crop3d函数，处理多通道输入
function output = crop3d(input, ch)
    [r, c, z, num_ch] = size(input);
    
    % if nargin < 2
    %     if num_ch > 1
    %         % 如果是多通道输入且未指定通道，返回所有通道中间层的裁剪
    %         output = input(r/4+1:r*3/4, c/4+1:c*3/4, z/2, :);
    %     else
    %         % 单通道情况
    %         output = input(r/4+1:r*3/4, c/4+1:c*3/4, z/2);
    %     end
    % else
    %     % 返回指定通道的中间层裁剪
    %     output = input(r/4+1:r*3/4, c/4+1:c*3/4, z/2, ch);
    % end
    if nargin < 2
        if num_ch > 1
            % 如果是多通道输入且未指定通道，返回所有通道中间层
            output = input(:, :, z/2, :);
        else
            % 单通道情况
            output = input(:, :, z/2);
        end
    else
        % 返回指定通道的中间层
        output = input(:, :, z/2, ch);
    end
end

function output = CT3D(input, tot_layers)
    [r,c] = size(input);
    % output = padarray(input,[r/2, c/2]); 
    % output = cat(3, gpuArray.zeros(r*2, c*2, tot_layers/2-1, 'single'), output, gpuArray.zeros(r*2, c*2, tot_layers/2, 'single'));
    output = input;
    output = cat(3, gpuArray.zeros(r, c, tot_layers/2-1, 'single'), output, gpuArray.zeros(r, c, tot_layers/2, 'single'));
end


% 3D向量软阈值操作(用于TV正则化)
function [varargout] =  soft_thres_3d(v, h, z, tau)
    % 计算3D梯度向量的幅值
    mag = sqrt(cat(1,v,zeros(1,size(v,2),size(v,3),'single')).^2 + ...
            cat(2,h,zeros(size(h,1),1,size(h,3),'single')).^2 + ...
            cat(3,z, zeros(size(z,1),size(z,2),1,'single')).^2);
    magt = soft_thresh(mag,tau);  % 对幅值进行软阈值
    mmult = magt./mag;            % 计算缩放因子
    mmult(mag==0) = 0;            % 处理除零情况
    % 分别对各方向分量进行缩放
    varargout{1} = v.*mmult(1:end-1,:,:);    % x方向
    varargout{2} = h.*mmult(:,1:end-1,:);    % y方向
    varargout{3} = z.*mmult(:,:,1:end-1);    % z方向
end

% 修改generate_laplacian_3D函数以支持GPU
function PsiTPsi = generate_laplacian_3D(rows, cols, layers)
    F3D = @(x) fftshift(fftn(ifftshift(x)));
    
    % % 使用gpuArray创建拉普拉斯算子
    % lapl = gpuArray.zeros(2*rows, 2*cols, layers, 'single');
    % lapl(rows+1, cols+1, layers/2+1) = 6;
    % lapl(rows+1, cols+2, layers/2+1) = -1;
    % lapl(rows+2, cols+1, layers/2+1) = -1;
    % lapl(rows, cols+1, layers/2+1) = -1;
    % lapl(rows+1, cols, layers/2+1) = -1;
    % lapl(rows+1, cols+1, layers/2+2) = -1;
    % lapl(rows+1, cols+1, layers/2) = -1;
    % PsiTPsi = abs(F3D(lapl));

    % 使用gpuArray创建拉普拉斯算子（不填充的版本）
    lapl = gpuArray.zeros(rows, cols, layers, 'single');
    lapl(rows/2+1, cols/2+1, layers/2+1) = 6;
    lapl(rows/2+1, cols/2+2, layers/2+1) = -1;
    lapl(rows/2+2, cols/2+1, layers/2+1) = -1;
    lapl(rows/2, cols/2+1, layers/2+1) = -1;
    lapl(rows/2+1, cols/2, layers/2+1) = -1;
    lapl(rows/2+1, cols/2+1, layers/2+2) = -1;
    lapl(rows/2+1, cols/2+1, layers/2) = -1;
    PsiTPsi = abs(F3D(lapl));
end

% 填充3D PSF堆叠
function output = pad3d(input)
    [r,c,~] = size(input);
    output = padarray(input,[r/2,c/2,0]);  % 仅在xy平面填充
end

% 计算解的相对变化率
function evolution_ratio = compute_evolution_ratio(xt, xtm1)
    evolution_ratio = norm(xt(:) - xtm1(:)) / norm(xtm1(:));
end

% Soft thresholding
function y = soft_thresh(x,t)
%   Y = soft_thresh(X,T) thresholds X using the soft thresholding rule.
%   X is a real- or complex-valued double- or
%   single-precision vector or matrix. Thresholding is applied using the
%   real-valued, double- or single-precision, nonnegative threshold, T. T
%   must be a scalar or the same size as X.
tmp = (abs(x)-t);
tmp = (tmp+abs(tmp))/2;
y   = sign(x).*tmp;
end