function [u] = get_u(train_binary_label,Sampled_maxtrix,InputPar,M_row,S1,I,v,b,L,mat_sample_num,p_view,u_view,v_view,k_iter)

%迭代中用来获取权重向量u的函数
%M为第几个视角标号
clear u;
Z=[];
Z_view=[];%%为每个视角存Z值
Z_w=zeros(size(S1));
Z_b=zeros(size(S1));
Z_view=cell(mat_sample_num,1);
% Z_wb=zeros(size(S1));
for p_u = 1:size(train_binary_label)
%     if p_u == 1
%         A = reshape(train_binary_data(p_u,:),M_row,M_col);
%         B=[A zeros(size(A,1),1);zeros(1,size(A,2)) 1];
%         z = B*v;
%         Z = train_binary_label(p_u)*z;
%         Z_sw=z;
%         clear A;clear B;
%     else

%--------------------求每个视角的uF-----------------------
for c_u=1:mat_sample_num
    
    if c_u==InputPar.view1selected||c_u==InputPar.view2selected
%        M_row_view=mat_sample_way(c_u,1);
%        M_col_view=mat_sample_way(c_u,2);
%        A_view=reshape(train_binary_data(p_u,:),M_row_view,M_col_view);
        A_view=Sampled_maxtrix{c_u}{p_u};
        B_view=[A_view zeros(size(A_view,1),1);zeros(1,size(A_view,2)) 1];
        z_view=train_binary_label(p_u)*B_view*v_view{c_u}(:,k_iter);%%%
        Z_view{c_u}=[Z_view{c_u},z_view];
    end%end if
end%for c_u
%_____________
        A = Sampled_maxtrix{p_view}{p_u};
        B=[A zeros(size(A,1),1);zeros(1,size(A,2)) 1];
        z = train_binary_label(p_u)*B*v;
        Z = [Z,z];
%         clear A;clear B;
%     end%end_if
index_w=find(L.w(p_u,:)==1);
index_b=find(L.b(p_u,:)==1);
for p_u2=1:length(index_w)
%    A2 = reshape(train_binary_data(index_w(p_u2),:),M_row,M_col);
    A2=Sampled_maxtrix{p_view}{index_w(p_u2)};
    B2=[A2 zeros(size(A2,1),1);zeros(1,size(A2,2)) 1];
    Z_w=Z_w+(B-B2)*v*v'*(B-B2)'*(InputPar.lam*L.w(p_u,index_w(p_u2)));%计算正则化项R_local
end
for p_u2=1:length(index_b);
    A2 = Sampled_maxtrix{p_view}{index_b(p_u2)};
    B2=[A2 zeros(size(A2,1),1);zeros(1,size(A2,2)) 1];
    Z_b=Z_b+(B-B2)*v*v'*(B-B2)'*((InputPar.lam-1)*L.b(p_u,index_b(p_u2)));%计算正则化项R_local
end
end%end_p_u
Z = Z';
Z_wb=Z_w+Z_b;

clear M_row_view;
clear M_col_view;
[samp_size,~]=size(train_binary_label);
Z_all=zeros(1,samp_size);
%_____________求视角总和_____________
for c_u=1:mat_sample_num
     if c_u==InputPar.view1selected||c_u==InputPar.view2selected
    if p_view==InputPar.view2selected&&p_view==c_u
        break;
    end
    
    if p_view==c_u
        continue;
       % c_u=c_u+1;
    end
    u_used=u_view{c_u}(:,k_iter);
    Z_all=Z_all+(u_used'*Z_view{c_u});
     end%end if
end
% Z_all=Z'*Z +Z_wb;
% if isempty(find(Z_all)==inf)==0%isnan(1./Z_all)==1
%     Z_all=zeros(size(S1));
% end

u1 = pinv(InputPar.curC*S1 +(1+InputPar.gamma*((mat_sample_num-1)/mat_sample_num)^2)*Z'*Z +Z_wb)*Z'*(I+b+InputPar.gamma*((mat_sample_num-1)/(mat_sample_num^2))*Z_all');
u=u1./norm(u1(1:M_row));

u = [u(1:M_row);1];%让u的最后一位总是为1
end
