function [ output ] = dropout_train( a,rate )
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
bino=binornd(1,rate,size(a,1),size(a,2));
output=a.*bino;

end

