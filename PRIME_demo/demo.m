clc; close all; clear
addpath('data', 'result', 'function');
rng(13);

load('Montrose.mat'); 
[r,c,M] = size(Zm3D);
Zm = reshape(Zm3D, r*c, M)';
save('./network/Zm3D.mat', 'Zm3D');

mode = 0; % 0: image_with_GT; 1: only_image
N = 6; % number of endmembers
gamma = 2; % sepctral upsampling factor

if N <= 8
    if M < N && M == 4
        %% PRIME
        [B_est, S_est, PRIME_time] = PRIME(Zm, N, gamma);
    elseif M < N && M ~= 4 
        %% PRIME
        [B_est, S_est, PRIME_time] = PRIME_wo_gamma(Zm, N);
    elseif M >= N
        %% HyperCSI
        [B_est, S_est, time] = HyperCSI(Zm, N);
    end
else
    error("PRIME is recommended only when the number of endmembers N ≤ 8.")
end

switch mode
    case 0
        [B_est_perm, S_est_perm] = permutation(S_GT_2D', S_est', B_est);
        S_est_perm = reshape(S_est_perm, size(ref_HSI, 1), size(ref_HSI, 2), N);

        %% MU-VCA
        load MU-VCA_result

        %% MU-NMF
        load MU-NMF_result

        [PRIME_RMSE, MUVCA_RMSE, MUNMF_RMSE] = show_abundance(S_GT, S_est_perm, vca_abundance_perm, nmf_abundance_perm);
        [PRIME_SAM, MUVCA_SAM, MUNMF_SAM] = show_signature(B_GT, B_est_perm, vca_signature_perm, nmf_signature_perm);

        fprintf('%-10s | SAM: %7.4f | RMSE: %7.4f | Time: %7.4f second\n', 'PRIME',  round(PRIME_SAM,4), round(PRIME_RMSE,4), round(PRIME_time,4));
        fprintf('%-10s | SAM: %7.4f | RMSE: %7.4f | Time: %7.4f second\n', 'MU-VCA', round(MUVCA_SAM,4), round(MUVCA_RMSE,4), round(vca_time,4));
        fprintf('%-10s | SAM: %7.4f | RMSE: %7.4f | Time: %7.4f second\n', 'MU-NMF', round(MUNMF_SAM,4), round(MUNMF_RMSE,4), round(nmf_time,4));
        fprintf('--------------------------------------------------------------\n');

    case 1

        S_est3D = reshape(S_est',r,c,N);

        figure;
        for i = 1:N
            subplot(2, N, i);
            plot(B_est(:,i),'--o','LineWidth',2.5,'DisplayName','GT','color',[0 0 0])

            subplot(2, N, i + N);
            imshow(S_est3D(:, :, i));
        end
end