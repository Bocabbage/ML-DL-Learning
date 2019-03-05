function [ output ] = Pool_3D( A,b,stride,mode )
%UNTITLED7 此处显示有关此函数的摘要
%   此处显示详细说明
test_result=Pool(A(:,:,1),b,stride,mode);
output=zeros(size(test_result,1),size(test_result,2),size(A,3));
for i=1:size(A,3)
    output(:,:,i)=Pool(A(:,:,i),b,stride,mode);
end


end

