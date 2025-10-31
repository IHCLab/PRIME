function [PRIME_RMSE, MUVCA_RMSE, MUNMF_RMSE] = show_abundance(S_GT, S_est, vca_S_est, nmf_S_est)

[r,c,N] = size(S_GT);

title_name=strcat('All Abundances');
figure('Name',title_name)
for i=1:N
    subplot(4,N,i)
    imshow(S_GT(:,:,i))
    if i == 1
            ylabel('GT', 'FontSize', 15, 'FontName','Times New Roman');
    end

    subplot(4,N,N+i)
    imshow(S_est(:,:,i))
    if i == 1
            ylabel('PRIME', 'FontSize', 15, 'FontName','Times New Roman');
    end

    subplot(4,N,2*N+i)
    imshow(vca_S_est(:,:,i))
    if i == 1
            ylabel('MU-VCA', 'FontSize', 15, 'FontName','Times New Roman');
    end

    subplot(4,N,3*N+i)
    imshow(nmf_S_est(:,:,i))
    if i == 1
            ylabel('MU-NMF', 'FontSize', 15, 'FontName','Times New Roman');
    end
end
set(gcf, 'Position', [100, 100, 1200, 800]);  % [left, bottom, width, height]


PRIME_RMSE =  sqrt(sum((S_GT(:) - S_est(:)).^2) / (r * c * N));
MUVCA_RMSE =  sqrt(sum((S_GT(:) - vca_S_est(:)).^2) / (r * c * N));
MUNMF_RMSE =  sqrt(sum((S_GT(:) - nmf_S_est(:)).^2) / (r * c * N));

end
