function [B_est_perm, S_est_perm] = permutation(S_GT,S_est,B_est)

n=size(S_GT,2);
m=size(S_est,2);
r=min(m,n);

if n>m
    index=nchoosek(1:n,m);
    temp=S_est; S_est=S_GT; S_GT=temp;
elseif m>n
    index=nchoosek(1:m,n);
elseif m==n
    index=1:m;
end
L=size(index,1);

for j=1:L

S_est_perm=S_est(:,index(j,:));

CRD = corrcoef([S_GT S_est_perm]);
D = abs(CRD(r+1:2*r,1:r));  
% permute results
perm_mtx = zeros(r,r);
aux=zeros(r,1);
for i=1:r
    [ld cd]=find(max(D(:))==D); 
    ld=ld(1);cd=cd(1); % in the case os more than one maximum
    perm_mtx(ld,cd)=1; 
    D(:,cd)=aux; D(ld,:)=aux';
end
S_est_perm = S_est_perm * perm_mtx;
B_est_perm = B_est * perm_mtx;
end