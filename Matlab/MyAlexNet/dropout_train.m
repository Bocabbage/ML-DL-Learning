function [ output ] = dropout_train( a,rate )
%UNTITLED �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
bino=binornd(1,rate,size(a,1),size(a,2));
output=a.*bino;

end

