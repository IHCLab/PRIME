function [PRIME_SAM, MUVCA_SAM, MUNMF_SAM] = show_signature(gt_signature, Proposed_signature_perm, vca_signature_perm, nmf_signature_perm)
concate=[gt_signature(:),Proposed_signature_perm(:),vca_signature_perm(:),nmf_signature_perm(:)];
max_val=max(concate(:))+0.05;
N=size(gt_signature,2);

PRIME_SAM = sum(acos(diag(gt_signature'*Proposed_signature_perm)./((sum(gt_signature.^2).*sum(Proposed_signature_perm.^2))'.^0.5)))/N*180/pi;
MUVCA_SAM = sum(acos(diag(gt_signature'*vca_signature_perm)./((sum(gt_signature.^2).*sum(vca_signature_perm.^2))'.^0.5)))/N*180/pi;
MUNMF_SAM = sum(acos(diag(gt_signature'*nmf_signature_perm)./((sum(gt_signature.^2).*sum(nmf_signature_perm.^2))'.^0.5)))/N*180/pi;

title_name=strcat('All Signatures');
figure('Name',title_name)
for i=1:N
    subplot(1,N,i)
    plot(gt_signature(:,i),'--o','LineWidth',2.5,'DisplayName','GT','color',[0 0 0])
    ylim([0 max_val]); 
    hold on
    plot(Proposed_signature_perm(:,i),'--o','LineWidth',2.5,'DisplayName','PRIME','color',[1 0 0])
    plot(nmf_signature_perm(:,i),'--o','LineWidth',2.5,'DisplayName','MU-NMF','color',[0 0 1])
    plot(vca_signature_perm(:,i),'--o','LineWidth',2.5,'DisplayName','MU-VCA','color',[0 1 0])
    set(gca,'FontSize',12);
    set(gca,'FontName','Times New Roman');
    hold off
    legend('Location','northwest')

end
set(gcf, 'Position', [100, 100, 1400, 250]); 

% [left, bottom, width, height]
end

