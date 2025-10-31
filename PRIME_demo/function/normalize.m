function [Y]= normalize(Y)

Y=(Y-min(Y(:)))/(max(Y(:))-min(Y(:)));
