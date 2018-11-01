function [GLMatStruct,u_v_b,co_result] = MultiVLSDMatMHKS_fun(train_binary_data,train_binary_label,Sampled_maxtrix,InputPar,L,mat_sample_num,mat_sample_way)
%MULTIVLSDMATMHKS_FUN Summary of this function goes here
%   Detailed explanation goes here
%train_binary_data��ѵ�����ݼ���ÿһ��һ������
%train_binary_label��ѵ�����ݼ���Ӧ������
%InputPar������������k��C��lam��
%M_row����ǰ���󻯵�����
%M_col����ǰ���󻯵�����
%L��ARLE������ȫ�־ֲ���������
%mat_sample_numΪ�ӽ�����
%Matlab2010b
label_vector = unique(train_binary_label);%[a,b]=unique(A),a��������A�в��ظ���Ԫ�أ�ÿ��һ����b���ص�һ����ͬԪ�ص�λ��

n1=length(find(train_binary_label==label_vector(1)));
n2=length(find(train_binary_label==label_vector(2)));%n1��n2�ֱ��ǵ�һ��͵ڶ����������

train_binary_label(1:n1) = 1;%ת����Ϊ1��-1
train_binary_label(n1+1:n1+n2) = -1;

%��ʼ������
total_iter = 100;
M_row=zeros(mat_sample_num,1);%ÿһ��Ϊһ���ӽǵ�����ֵ
M_col=zeros(mat_sample_num,1);
del_var_2=0;
del_var_1=0;
cor_res=zeros(5,1);%��¼ÿ5�ֵ�ֵ
i_cov=1;
comp_co=0;%��¼��5�ֵĵ�һ��ֵ
cov=0;%��¼����ֵ��ÿ5������Сֵ�����cov�����жϲ�����
sa=0;
co_result='succeed';

for p_view=1:mat_sample_num  %p_viewΪ�ӽǱ��
        M_row(p_view) = mat_sample_way(p_view,1);%��ǰ���󻯺���������
        M_col(p_view) = mat_sample_way(p_view,2);%��ǰ���󻯺���������
        u{p_view} = zeros(M_row(p_view) + 1,total_iter);
        v{p_view} = zeros(M_col(p_view) + 1,total_iter);
        b{p_view} = zeros(n1+n2,total_iter);
        e{p_view} = zeros(n1+n2,total_iter);
        u{p_view}(:,1) = [InputPar.u_u * ones(M_row(p_view),1);1];%u�����һ��Ϊ1
        b{p_view}(:,1) = InputPar.b_b * ones(n1+n2,1);
        S_1{p_view} = M_row(p_view)*eye(M_row(p_view));
        S_2{p_view} = M_col(p_view)*eye(M_col(p_view));
        S1{p_view} = [S_1{p_view} zeros(size(S_1{p_view},1),1);zeros(1,size(S_1{p_view},2)) 1];
        S2{p_view} = [S_2{p_view} zeros(size(S_2{p_view},1),1);zeros(1,size(S_2{p_view},2)) 1];
end%for


I = ones(n1+n2,1);%��ʽ���õ���ȫ1����


rho = 0.99;%p
eta = 10^(-4);%�ж���ֹ������
k_iter = 1;
for i_mat=1:mat_sample_num  %i_matΪ��ǰ�ӽǱ��
u_v_b{i_mat}=zeros(5+M_row(i_mat)+1+M_col(i_mat)+1+n1+n2,total_iter);%ÿһ�ε����洢u,v,b��ֵ
end
%��ʼ����ѵ��
while (k_iter < total_iter)
    
    for p_view=1:mat_sample_num  %p_viewΪ�ӽǱ��
 %       M_row = mat_sample_way(p_view,1);%��ǰ���󻯺���������
%        M_col = mat_sample_way(p_view,2);%��ǰ���󻯺���������
if p_view==InputPar.view1selected||p_view==InputPar.view2selected
    if p_view==InputPar.view1selected
    InputPar.curC=InputPar.C;
    else
        InputPar.curC=InputPar.C2;
    end
    v{p_view}(:,k_iter) = get_v(train_binary_label,Sampled_maxtrix,InputPar,S2{p_view},I,u{p_view}(:,k_iter),b{p_view}(:,k_iter),L,mat_sample_num,p_view,u,v,k_iter);
    
    e{p_view}(:,k_iter) = get_e(train_binary_label,Sampled_maxtrix,I,p_view,u{p_view}(:,k_iter),v{p_view}(:,k_iter),b{p_view}(:,k_iter));
    b{p_view}(:,k_iter+1) = (b{p_view}(:,k_iter) + rho*(e{p_view}(:,k_iter) + abs(e{p_view}(:,k_iter))));
 
%    stop_tag = norm(b{p_view}(:,k_iter+1) - b{p_view}(:,k_iter),2);
%     tag(k_iter)=stop_tag;
    u_v_b{p_view}(:,k_iter)=[InputPar.k;InputPar.C;InputPar.lam;M_row(p_view);M_col(p_view);u{p_view}(:,k_iter);v{p_view}(:,k_iter);b{p_view}(:,k_iter)];
    if k_iter==total_iter
        break;
    end
    u{p_view}(:,k_iter+1) = get_u(train_binary_label,Sampled_maxtrix,InputPar,M_row(p_view),S1{p_view},I,v{p_view}(:,k_iter),b{p_view}(:,k_iter),L,mat_sample_num,p_view,u,v,k_iter);
end%end if    
    
%    if stop_tag < eta% ���޸�
%        if p_view==mat_sample_num
%            break;
%        end
%        continue;
%    else
%        u{p_view}(:,k_iter+1) = get_u(train_binary_data,train_binary_label,InputPar,M_row(p_view),M_col(p_view),S1{p_view},I,v{p_view}(:,k_iter),b{p_view}(:,k_iter),L,mat_sample_num,p_view,mat_sample_way,u,v,k_iter);
%    end%end_if
    
    end%for P_view
    if k_iter>1
        del_var_2=MultiVLSDLk(train_binary_data,train_binary_label,InputPar,S1,S2,I,u,v,b,k_iter,L,mat_sample_num,mat_sample_way);
        del_var_1=MultiVLSDLk(train_binary_data,train_binary_label,InputPar,S1,S2,I,u,v,b,k_iter-1,L,mat_sample_num,mat_sample_way);
        
    end
    op=norm(del_var_2-del_var_1,2);
    ab=op/norm(del_var_1,2);
    %------------------�����ж�------------
    if i_cov==1
        comp_co=ab;
    end
    
    cor_res(i_cov)=ab;
    if i_cov==5
        i_cov=1;
        cov=min(cor_res);
        sa=comp_co-cov;%covӦ��С��comp_co��Ϊ������������ʱsaΪ��ֵ
    else 
        i_cov=i_cov+1;
    end
    if sa<0
        co_result='reject';
    end
    %-------------------------------------
    %-------------------------------------
    if ab<eta||sa<0
        break;
    end
    
    k_iter = k_iter + 1;
    
    
end%while
for final_v=1:mat_sample_num
if k_iter == total_iter
    GLMatStruct.u{final_v} = u{final_v}(:,k_iter);
    GLMatStruct.v{final_v} = v{final_v}(:,k_iter-1);
else
    GLMatStruct.u{final_v} = u{final_v}(:,k_iter);
    GLMatStruct.v{final_v} = v{final_v}(:,k_iter);    
end%end_if
end%end for final_v
end

