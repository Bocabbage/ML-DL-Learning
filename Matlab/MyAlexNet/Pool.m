function [ Result ] = Pool( A,b,stride,mode)

size_A=size(A);
%size_B=size(B);
%%   检验

if(min(size_A(1:2))<b)
    error('池化范围太大');
end
re=zeros(2,1);
re(1)=rem(size_A(1)-b,stride);
re(2)=rem(size_A(2)-b,stride);

%% Padding
if(re(1) && re(2))
    pa_A=zeros(size_A(1)+rem(stride-re(1),stride),...
        size_A(2)+rem(stride-re(2),stride));
    pa_A(1:size_A(1),1:size_A(2))=A;
else
    pa_A=A;
end

size_paA=size(pa_A);
%% 
Result=zeros(1+(size_paA(1)-b)/stride,...
    1+(size_paA(2)-b)/stride);
for i=1:size(Result,1)
    for j=1:size(Result,2)
        temp=pa_A(((i-1)*stride+1):((i-1)*stride+b),...
            ((j-1)*stride+1):((j-1)*stride+b),:);
        if(mode=='max')
          Result(i,j)=max(max(temp)) ;
        else
            if(mode=='avr')
                 Result(i,j)=mean(mean(temp)) ;
            else error('方法未指定');
            end
        end
    end


end
end
