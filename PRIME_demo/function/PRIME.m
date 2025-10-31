%=====================================================================
% Programmer: Jhao-Ting Lin
% E-mail: q38091534@gs.ncku.edu.tw
% Date: 2025/03/17
% -------------------------------------------------------
% Reference:
% C.-H. Lin, and J.-T. Lin,
% ``PRIME: Unsupervised multispectral unmixing using virtual quantum prism and convex geometry,"
% IEEE Transactions on Geoscience and Remote Sensing, 2025.
%======================================================================
% A multispectral unmixing algorithm
% [B_est, S_est, time] = PRIME(Zm, N, gamma)
%======================================================================
%  Input
%  Zm is P-by-L data matrix, where P is the number of multispectral bands and L is the number of pixels.
%  N is the number of endmembers.
%  gamma is the sepctral upsampling factor.
%----------------------------------------------------------------------
%  Output
%  B_est is P-by-N mixing matrix whose columns are estimated endmember signatures.
%  S_est is N-by-L source matrix whose rows are estimated abundance maps.
%  time is the computation time (in seconds).
%========================================================================

function [B_est, S_est, time] = PRIME(Zm, N, gamma)
%% delete network parameter
check_n_parameter();

[P, L] = size(Zm);
t0 = datetime('now');
%% generate sepctral response matrix D
D = kron(eye(P), ones(1, gamma));
save('./network/D.mat', 'D');

%% initialize Zh
noise_ratio = 0.05;
[Zh] = generate_Zh_from_Zm(Zm, noise_ratio, gamma);
Zh3D = reshape(Zh', sqrt(L), sqrt(L), []);
save('./network/Zh3D.mat', 'Zh3D');

%% init A, S
[A_est, S_est, ~] = HiSun(Zh, N); 
time0 = seconds(datetime('now') - t0);

%% iteration
iter = 10; 
time1 = zeros(1, iter);
M = 2 * eye(gamma * P) + D' * D;  % precalculate
for i = 1 : iter
    fprintf('iteration: %d\n', i);
    
    %% update f (call python)
    system('source activate mu && python ./network/main.py');
    
    %% update Zh (load f(Zm) = Zh_model)
    t1 = datetime('now');
    load('./network/result/DL_result.mat', 'Zh_model');
    Zh_model = reshape(Zh_model, L, gamma * P)';
    Zh = M \ (A_est * S_est + Zh_model + D' * Zm);    Zh(Zh < 0) = 0 ;
    Zh3D = reshape(Zh', sqrt(L), sqrt(L), gamma * P);
    save('./network/Zh3D.mat', 'Zh3D');
    
    %% update (A,S)
    [A_est, S_est] = HyperCSI_modified(Zh, N);
    time1(i) = seconds(datetime('now') - t1);

end
load('./network/result/train_time_file', 'train_time_file');
time = time0 + sum(time1) + sum(train_time_file);
B_est = D * A_est;
