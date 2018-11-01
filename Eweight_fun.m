function [ L ] = Eweight_fun( train_binary_data,train_binary_label,k )
%EWEIGHT_FUN Summary of this function goes here
%   Detailed explanation goes here
%������������2k���������k������С�ĵ�Ϊ����
clear L;
[class_label,tempt_location] = unique(train_binary_label);%[a,b]=unique(A),a��������A�в��ظ���Ԫ�أ�ÿ��һ����b���ص�n��Ԫ������λ�ã��õ���n1Ϊ��һ����������������ǩ��һ���ܳ��ı�ǩ
%����������������������
n1=length(find(train_binary_label==class_label(1)));
n2=length(find(train_binary_label==class_label(2)));%n1��n2�ֱ��ǵ�һ��͵ڶ����������
%�������ڹ�ϵ���������
%--------------------------
X=train_binary_data';%����������ת����ÿ��һ������
n=size(X,2);%�������������������ĸ���
A=diag(X'*X);%��������ƽ�� x^2
B=A';%AΪ��������BΪ������
D_squ=A*ones(1,n)+ones(n,1)*B-2.*(X'*X); %��������֮�����ƽ�� (a^2-b^2)
Dis_Sam=sqrt(D_squ);%����
[~,Loc_sam_1]=sort(Dis_Sam,2);%�Ծ������ÿһ�����򣬽��loc_sam��ŵĽ��ֻ��ǰk����������ֵ������,ֻ��ע��ֵ��λ�ã�ȡǰk��
[~,Loc_sam_2]=sort(Loc_sam_1,2);%��loc_sam_1��ŵĽ����λ�ã����õ�Loc_sam_2�Ľ����ֵΪloc_sam�е�λ�ã���ֵС��k�ģ���ʾΪk�����ڵ�����,Loc_sam_2����ֵΪԭʼ������λ��
                                %������Ϊ[3 1 2]ʱ�������ľ���ĵ�����Ϊԭʼ������һ��λ�õ�ֵ����ζ��ԭʼ�����е�һ��ֵ�Ǿ����3���
c_sort=reshape(Dis_Sam,1,n*n);%�ų�һ�������У����������ڹ����ڽ�����
c_sort(1,find(Loc_sam_2>k))=0;%find�����ҳ�������ֵ�������ţ�Loc_sam_2�е�����������ԭ��������λ�ã���[3 1 2]��3>k��3������Ϊ1�����һ������3���صõ�c_sort(1,1)�����Դ�c_sort(1,1)=0
c_sort(1,find(Loc_sam_2<=k+1))=1;
S_sorted_sample=reshape(c_sort,n,n);%����k���ھ���
S_sorted_sample=S_sorted_sample+diag(-diag(S_sorted_sample));%�ѶԽ���Ԫ������
%---------------2k����---
c_sort_e=reshape(Dis_Sam,1,n*n);%�ų�һ�������У����������ڹ����ڽ�����
c_sort_e(1,find(Loc_sam_2>2*k))=0;%find�����ҳ�������ֵ�������ţ�Loc_sam_2�е�����������ԭ��������λ�ã���[3 1 2]��3>k��3������Ϊ1�����һ������3���صõ�c_sort(1,1)�����Դ�c_sort(1,1)=0
c_sort_e(1,find(Loc_sam_2<=2*k+1))=1;
S_sorted_sample_e=reshape(c_sort_e,n,n);%����k���ھ���
S_sorted_sample_e=S_sorted_sample_e+diag(-diag(S_sorted_sample_e));%�ѶԽ���Ԫ������


%--------���ؽ���-----
entropy_sam=zeros(length(train_binary_label),1);%��ÿ����������
for i_sam_e=1:length(train_binary_label)
    knn_sum=sum(S_sorted_sample(i_sam_e,:));%��õ��ܵĽ�����
    knn_label=find(S_sorted_sample(i_sam_e,:)==1);%���ڵ�ȫ������������
    sam_num_1=sum(train_binary_label(knn_label)==class_label(1));%��һ������������
    sam_num_2=sum(train_binary_label(knn_label)==class_label(2));%�ڶ�������������
    p_1=sam_num_1/knn_sum;
    p_2=sam_num_2/knn_sum;
    if p_1==0||p_2==0
        entropy_sam(i_sam_e)=0;
    else
        entropy_sam(i_sam_e)=-1*p_1*log(p_1)+(-1)*p_2*log(p_2);
    end
end
%------�����ؾ���------
entro_matr=ones(length(train_binary_label),1)*entropy_sam';%�ؾ���
for i_ent=1:length(train_binary_label)
    entro_matr(i_ent,find(S_sorted_sample_e(i_ent,:)~=1))=10;%�������򣬰Ѳ���صĶ��趨Ϊ10
end
%------����k�ؽ���-----
[~,Loc_ent]=sort(entro_matr,2);
[~,Loc_ent_2]=sort(Loc_ent,2);
e_sort=reshape(entro_matr,1,n*n);
e_sort(1,find(Loc_ent_2>k))=0;
e_sort(1,find(Loc_sam_2<=k+1))=1;
W_sorted_sample=reshape(e_sort,n,n);
W_sorted_sample=W_sorted_sample+diag(-diag(W_sorted_sample));%�ѶԽ���Ԫ������
Ww_ij=W_sorted_sample;%���ڹ�ϵ
Wb_ij=W_sorted_sample;%����ϵ
%�����������k���ھ���
Ww_ij(1:n1,n1+1:n2)=0;  %ֻ��1��n1������Ϊ��һ�࣬���Դ���n1��Ϊ�ڶ�����������1��n1��������n1+1��ĩβ�Ĺ�ϵΪ����ϵ���������ڹ�ϵ������ȫ������
Ww_ij(n1+1:end,1:n1)=0;
Wb_ij(1:n1,1:n1)=0;
Wb_ij(n1+1:end,n1+1:end)=0;
L.w=Ww_ij;
L.b=Wb_ij;

end

