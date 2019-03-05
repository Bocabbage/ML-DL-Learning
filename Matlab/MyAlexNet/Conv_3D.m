function [ output ] = Conv_3D( A,B,stride )

if (size(A)==size(B(:,:,:,1)))      %È«Á¬½Ó
    output=zeros(size(B,4),1);
    for i=1:size(B,4)
    output(i)=Conv(A,B(:,:,:,i),stride);
    end
    return
end
test_result=Conv(A,B(:,:,:,1),stride);
output=zeros(size(test_result,1),size(test_result,2),size(B,4));
for i=1:size(B,4)
    output(:,:,i)=Conv(A,B(:,:,:,i),stride);
end


end

