function [ E1 ] = nms( E )
%NMS Summary of this function goes here
%   Detailed explanation goes here
[Ox,Oy]=gradient2(convTri(E,4));
[Oxx,~]=gradient2(Ox); [Oxy,Oyy]=gradient2(Oy);
O=mod(atan(Oyy.*sign(-Oxy)./(Oxx+1e-5)),pi);
E1=edgesNmsMex(E,O,1,5,1.01,4); 
end