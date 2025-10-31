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
% The initialization of PRIME algorithm
% [ZhN] = generate_Zh_from_Zm(Zm, noise_ratio, gamma)
%======================================================================
%  Input
%  Zm is P-by-L data matrix, where P is the number of multispectral bands and L is the number of pixels.
%  noise_ratio is the energy of the Gaussian noise.
%  gamma is the sepctral upsampling factor.
%----------------------------------------------------------------------
%  Output
%  ZhN is M-by-L initialized virtual hyperspectral image, where M is the number of hyperspectral bands.
%========================================================================

function [Zh_N] = generate_Zh_from_Zm_wo_gamma(Zm, noise_ratio, splitBands)

%% dispersion 
Zh = perturbation(Zm, splitBands);

%% add Gaussian noise 
Zh_N = add_noise(Zh, noise_ratio);


function  [Zh] = perturbation(Zm, splitBands)
[P, L] = size(Zm);

coeffs = linspace(-((2 -  1) / 2), (2 - 1) / 2, 2); 
Zh = zeros(8, L);

h_index = 1;
for i = 1 : P
    if ismember(i, splitBands)
        bias = (Zm(i + 1, :) - Zm(i, :)) / 2; 
        for j = 1 : 2
            Zh(h_index - 1 + j, :) = (Zm(i, :) + coeffs(j) * bias) / 2;
        end  
        h_index = h_index + 2;
    else
        Zh(h_index, :) = Zm(i, :);
        h_index = h_index + 1;

    end
end
Zh(Zh < 0) = 0;

return;

function [Zh_N] = add_noise(Zh, noise_ratio)
[M, L] = size(Zh);
signal_power = norm(Zh, 'fro') ^ 2;
Noise = randn(M, L);
noise_power = norm(Noise, 'fro') ^ 2;
constant = sqrt((signal_power * noise_ratio) / noise_power); 

Noise = Noise * constant;
Zh_N = Zh + Noise;
Zh_N(Zh_N < 0)=0;
return;