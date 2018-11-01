function [v] = get_v(train_binary_label,Sampled_maxtrix,InputPar,S2,I,u,b,L,mat_sample_num,p_view,u_view,v_view,k_iter)

%迭代中用来获取权重向量v的函数
%mat_sample_num为视角总数
%p_view为当前视角标号
%mat_sample_way为视角组合方式的矩阵
%u_view为全部视角的u，v_view同理
%k_iter为当前迭代轮次
clear v;
Y=[];
Y_view=cell(mat_sample_num,1);%%为每个视角存Y值
Y_w=zeros(size(S2));
Y_b=zeros(size(S2));
% Y_wb=zeros(size(S1));
for p_v = 1:size(train_binary_label)
%     if p_v == 1
%         A = reshape(train_binary_data(p_v,:),M_row,M_col);
%         B=[A zeros(size(A,1),1);zeros(1,size(A,2)) 1];
%         y = (u'*B)';
%         Y = train_binary_label(p_v)*y;
%         Y_sw=y;
%         clear A;clear B;
%     else
%------------求每个视角的Yv-------------------------------------
        for c_v=1:mat_sample_num
            
    if c_v==InputPar.view1selected||c_v==InputPar.view2selected
%        M_row_view=mat_sample_way(c_v,1);
%        M_col_view=mat_sample_way(c_v,2);
%        A_view=reshape(train_binary_data(p_v,:),M_row_view,M_col_view);
        A_view=Sampled_maxtrix{c_v}{p_v};
        B_view=[A_view zeros(size(A_view,1),1);zeros(1,size(A_view,2)) 1];
        y_view=train_binary_label(p_v)*(u_view{c_v}(:,k_iter)'*B_view)';%%%
        Y_view{c_v}=[Y_view{c_v},y_view];
        
    end%end if     
        end%for c_v
%--------------------------------------------------------------
        
        A = Sampled_maxtrix{p_view}{p_v};
        B=[A zeros(size(A,1),1);zeros(1,size(A,2)) 1];
        
        y = train_binary_label(p_v)*(u'*B)';
        Y = [Y,y];
        
%         clear A;clear B;
index_w=find(L.w(p_v,:)==1);
index_b=find(L.b(p_v,:)==1);
for p_v2=1:length(index_w)
%    A2 = reshape(train_binary_data(index_w(p_v2),:),M_row,M_col);
    A2=Sampled_maxtrix{p_view}{index_w(p_v2)};
    B2=[A2 zeros(size(A2,1),1);zeros(1,size(A2,2)) 1];
    Y_w=Y_w+(B-B2)'*u*u'*(B-B2)*(InputPar.lam*L.w(p_v,index_w(p_v2)));%计算正则化项R_local
end
for p_v2=1:length(index_b)
    A2 = Sampled_maxtrix{p_view}{index_b(p_v2)};
    B2=[A2 zeros(size(A2,1),1);zeros(1,size(A2,2)) 1];
    Y_b=Y_b+(B-B2)'*u*u'*(B-B2)*((InputPar.lam-1)*L.b(p_v,index_b(p_v2)));%计算正则化项R_local
end
%     end%end_if
end%end_p_v 
Y = Y';
Y_wb=Y_w+Y_b;
clear M_row_view;
clear M_col_view;
[samp_size,~]=size(train_binary_label);
Y_all=zeros(samp_size,1);
%_____________求视角总和_____________
for c_v=1:mat_sample_num
    
    if c_v==InputPar.view1selected||c_v==InputPar.view2selected
    if p_view==InputPar.view2selected&&p_view==c_v
        break;
    end
    
    if p_view==c_v
        continue;
        %c_v=c_v+1;
    end
    Y_view_temp=Y_view{c_v}';
    Y_all=Y_all+(Y_view_temp*v_view{c_v}(:,k_iter));
    end%end if
end
% Y_all=Y'*Y +Y_wb;

% if isempty(find(Y_all)==inf)==0%isnan(1./Y_all)==1
%     Y_all=zeros(size(S2));
% end

v1 = pinv(InputPar.curC*S2 + (1+InputPar.gamma*((mat_sample_num-1)/mat_sample_num)^2)*Y'*Y +Y_wb)*Y'*(I+b+InputPar.gamma*((mat_sample_num-1)/(mat_sample_num^2))*Y_all);
v=v1./norm(v1);


end

