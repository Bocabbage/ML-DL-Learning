function [ output ] = LRN( A,k,n,alpha,beta )
N=size(A,3);    %¾í»ýºË×ÜÊý
output=zeros(size(A));
output_map=zeros(size(A(:,:,1)));
for m=1:N
    temp_map=A(:,:,m);

    for i=1:size(temp_map,1)
        for j=1:size(temp_map,2)
            vec=A(i,j,max(1,i-ceil(n/2)):min(N,i+ceil(n/2)));
            output_map(i,j)=temp_map(i,j)/((k+alpha*sum(vec.^2)))^beta;
        end
    end
    output(:,:,m)=output_map;
end
end


