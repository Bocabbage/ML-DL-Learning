function [ output ] = Pool_3D( A,b,stride,mode )
%UNTITLED7 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
test_result=Pool(A(:,:,1),b,stride,mode);
output=zeros(size(test_result,1),size(test_result,2),size(A,3));
for i=1:size(A,3)
    output(:,:,i)=Pool(A(:,:,i),b,stride,mode);
end


end

