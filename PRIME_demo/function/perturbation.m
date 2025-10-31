function  [Zh]= perturbation(Zm,P)
division=4;
Zh=zeros(P*2,size(Zm,2));
for i=1:P-1
bias=(Zm(i+1,:)-Zm(i,:))/division;
Zh(2*i-1,:)=Zm(i,:)-bias;
Zh(2*i,:)=Zm(i,:)+bias;
end
% bias=(Zm(P,:)-Zm(P-1,:))/division;
Zh(2*P-1,:)=Zm(P,:)-bias;
Zh(2*P,:)=Zm(P,:)+bias;

    %% show divide itto 4 parts
%     [~, b]=max(vecnorm(Zh))
%     pixel=b; %3
%     plot([0:2:14],Zh(:,pixel),'*-');
%     hold on
%     plot([1:4:13],Zm(:,pixel),'o-');
%     legend('Zh','Zm','Location','southeast');
%     hold off

Zh=Zh/2;
Zh(Zh<0)=0;
return;
