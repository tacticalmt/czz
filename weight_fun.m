function [L] = weight_fun( train_binary_data,train_binary_label,k )
%�����ڽӾ���ĺ���
%   Detailed explanation goes here
clear L;
[~,tempt_location] = unique(train_binary_label);%[a,b]=unique(A),a��������A�в��ظ���Ԫ�أ�ÿ��һ����b���ص�n��Ԫ������λ�ã��õ���n1Ϊ��һ����������������ǩ��һ���ܳ��ı�ǩ
%����������������������
n1=tempt_location(1);
n2=tempt_location(2)-tempt_location(1);%n1��n2�ֱ��ǵ�һ��͵ڶ����������


%�������ڹ�ϵ���������
%--------------------------
X=train_binary_data';%����������ת����ÿ��һ������
n=size(X,2);%�������������������ĸ���
A=diag(X'*X);%��������ƽ�� x^2
B=A';%AΪ��������BΪ������
D_squ=A*ones(1,n)+ones(n,1)*B-2.*(X'*X); %��������֮�����ƽ�� (a^2-b^2)
Dis_Sam=sqrt(D_squ);%����
%--------------------------
%k���ھ�����
[~,Loc_sam_1]=sort(Dis_Sam,2);%�Ծ���������򣬽��loc_sam��ŵĽ��ֻ��ǰk����������ֵ������,ֻ��ע��ֵ��λ�ã�ȡǰk��
[~,Loc_sam_2]=sort(Loc_sam_1,2);%��loc_sam_1��ŵĽ����λ�ã����õ�Loc_sam_2�Ľ����ֵΪloc_sam�е�λ�ã���ֵС��k�ģ���ʾΪk�����ڵ�����,Loc_sam_2����ֵΪԭʼ������λ��
                                %������Ϊ[3 1 2]ʱ�������ľ���ĵ�����Ϊԭʼ������һ��λ�õ�ֵ����ζ��ԭʼ�����е�һ��ֵ�Ǿ����3���
c_sort=reshape(Dis_Sam,1,n*n);%�ų�һ�������У����������ڹ����ڽ�����
c_sort(1,find(Loc_sam_2>k))=0;%find�����ҳ�������ֵ�������ţ�Loc_sam_2�е�����������ԭ��������λ�ã���[3 1 2]��3>k��3������Ϊ1�����һ������3���صõ�c_sort(1,1)�����Դ�c_sort(1,1)=0
c_sort(1,find(Loc_sam_2<=k+1))=1;
S_sorted_sample=reshape(c_sort,n,n);%����k���ھ���
S_sorted_sample=S_sorted_sample+diag(-diag(S_sorted_sample));%�ѶԽ���Ԫ������
Ww_ij=S_sorted_sample;%���ڹ�ϵ
Wb_ij=S_sorted_sample;%����ϵ

%�����������k���ھ���
Ww_ij(1:n1,n1+1:n2)=0;  %ֻ��1��n1������Ϊ��һ�࣬���Դ���n1��Ϊ�ڶ�����������1��n1��������n1+1��ĩβ�Ĺ�ϵΪ����ϵ���������ڹ�ϵ������ȫ������
Ww_ij(n1+1:end,1:n1)=0;
Wb_ij(1:n1,1:n1)=0;
Wb_ij(n1+1:end,n1+1:end)=0;
L.w=Ww_ij;
L.b=Wb_ij;


end

