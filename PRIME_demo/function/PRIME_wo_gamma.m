%=====================================================================
% Programmer: Jhao-Ting Lin
% E-mail: q38091534@gs.ncku.edu.tw
% Date: 2025/10/23
% -------------------------------------------------------
% Reference:
% C.-H. Lin, and J.-T. Lin,
% ``PRIME: Unsupervised multispectral unmixing using virtual quantum prism and convex geometry,"
% IEEE Transactions on Geoscience and Remote Sensing, 2025.
%======================================================================
% A multispectral unmixing algorithm
% [B_est, S_est, time] = PRIME_wo_gamma(Zm, N)
%======================================================================
%  Input
%  Zm is P-by-L data matrix, where P is the number of multispectral bands and L is the number of pixels.
%  N is the number of endmembers.
%----------------------------------------------------------------------
%  Output
%  B_est is P-by-N mixing matrix whose columns are estimated endmember signatures.
%  S_est is N-by-L source matrix whose rows are estimated abundance maps.
%  time is the computation time (in seconds).
%========================================================================

function [B_est, S_est, time] = PRIME_wo_gamma(Zm, N)
%% delete network parameter
check_n_parameter();

[P, L] = size(Zm);
t0 = datetime('now');
%% generate sepctral response matrix D
D = zeros(P, 8);
switch  P
    case 5
        splitBands = [2 3 4];
    case 6
        splitBands = [2 5];
    case 7
        splitBands = [4];
end

h_index = 1;
for m_index = 1:P
    if ismember(m_index, splitBands)
        D(m_index, h_index:h_index+1) = 0.5;
        h_index = h_index + 2;
    else
        D(m_index, h_index) = 1;
        h_index = h_index+ 1;
    end
end
save('./network/D.mat', 'D');

%% initialize Zh
noise_ratio = 0.05;
[Zh] = generate_Zh_from_Zm_wo_gamma(Zm, noise_ratio, splitBands);
Zh3D = reshape(Zh', sqrt(L), sqrt(L), []);
save('./network/Zh3D.mat', 'Zh3D');

%% init A, S
[A_est, S_est, ~] = HiSun(Zh, N);
time0 = seconds(datetime('now') - t0);

%% iteration
iter = 10;
time1 = zeros(1, iter);
M = 2 * eye(8) + D' * D;  % precalculate
for i = 1 : iter
    fprintf('iteration: %d\n', i);

    %% update f (call python)
    system('source activate mu && python ./network/main_wo_gamma.py');
        
    %% update Zh (load f(Zm) = Zh_model)
    t1 = datetime('now');
    load('./network/result/DL_result.mat', 'Zh_model');
    Zh_model = reshape(Zh_model, L, 8)';
    Zh = M \ (A_est * S_est + Zh_model + D' * Zm);    Zh(Zh < 0) = 0 ;
    Zh3D = reshape(Zh', sqrt(L), sqrt(L), 8);
    save('./network/Zh3D.mat', 'Zh3D');

    %% update (A,S)
    [A_est, S_est] = HyperCSI_modified(Zh, N);
    time1(i) = seconds(datetime('now') - t1);

end
load('./network/result/train_time_file', 'train_time_file');
time = time0 + sum(time1) + sum(train_time_file);
B_est = D * A_est;
