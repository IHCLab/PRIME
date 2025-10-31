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

function [Zh_N] = generate_Zh_from_Zm(Zm, noise_ratio, gamma)

%% dispersion 
Zh = perturbation(Zm, gamma);

%% add Gaussian noise 
Zh_N = add_noise(Zh, noise_ratio);


function  [Zh] = perturbation(Zm, gamma)
[P, L] = size(Zm);
coeffs = linspace(-((gamma -  1) / 2), (gamma - 1) / 2, gamma); 
Zh = zeros(P * gamma, L);

for i = 1 : P - 1
        bias = (Zm(i + 1, :) - Zm(i, :)) / gamma; 
        for j = 1 : gamma
            Zh((i - 1) * gamma + j, :) = Zm(i, :) + coeffs(j) * bias;
        end
end

bias = (Zm(P, :) - Zm(P - 1, :)) / gamma;
    for j = 1 : gamma
        Zh((P - 1) * gamma + j, :) = Zm(P, :) + coeffs(j) * bias;
    end

Zh = Zh / gamma;
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