function [ output_args ] = EnsembleMVLSD_main(dataset,Basic_para,time_sampling,k,C,lam,gamma,u_u,b_b )
%ENSEMBLEMVLSD_MAIN Summary of this function goes here
%   Detailed explanation goes here
%dataset���ݼ����� Ϊһ���ṹ��  
%vail_typeΪ������֤����
%ktimes������֤����
%------------------------------
%-------��ʼ����������----------
%------------------------------
ktimes=Basic_para.ktimes;
std_vec=Basic_para.std_vec';
vail_type=Basic_para.vail_type;
label=Basic_para.label;
dataname=Basic_para.datanName;
IR=Basic_para.IR;%��ƽ����
para_sampling.IR=IR;
sam_ratio=Basic_para.samp_ratio;
sub=time_sampling;
time = zeros(ktimes,1);%������¼ѵ��ʱ�������
[mat_sample_way,mat_sample_num]=Matrixlize_fun(std_vec);%���󻯣�[�����Ժ����Ϸ�ʽ����Ϸ�ʽ�ĸ���]=������������Ŀ�ĺ���(temptvec_data),std_vecΪһ����������
%������ƫ��������i���ƫ�Ƽ�¼��bias_class(i)��
para_sampling.Mways=mat_sample_num;
T_combin=(mat_sample_num*(mat_sample_num-1))/2;%���������
if IR>(T_combin*sub)
    sub=sam_ratio*ceil(IR/T_combin);
end
bias_class = zeros(length(unique(label)),1);
for i_label = 1:length(unique(label))
    i_label_tempt = find(label==unique(i_label));
    bias_class(i_label) = i_label_tempt(1);
end%for_i_label
bias_class = bias_class - 1;

sum_class = numel(unique(label));%��������
AUC=zeros(ktimes,1);
T_AUC=zeros(ktimes,1);
GMM=ones(ktimes,1);
accuracy = zeros(ktimes,1);%��¼��ȷ�ȵ�����

InputPar.k = k;
InputPar.C = C;
InputPar.C2= C;
InputPar.curC=0;%��ǰ�ӽ��õ�C
InputPar.lam = lam;
% InputPar.ita=ita;
InputPar.u_u = u_u;
InputPar.b_b = b_b;
InputPar.dataname= dataname;
InputPar.gamma=gamma;
InputPar.M_row=0;
InputPar.M_col=0;
global_covflag='succeed';
%------------------------------
%------------------------------
%----��ӡ��Ļ-----------------
disp(['Setting:Dataset-',dataname,',Cross Vaildation-',vail_type,' ,k-',num2str(k),' ,C-',num2str(C),' ,lamda-',num2str(lam),' ,view num-',num2str(mat_sample_num),' ,gamma-',num2str(gamma)]);%��ӡ����Ļ��
disp(['--------------------------------------']);

%----------------------------
%-----------�ļ������¼-------------
%��¼ÿ�����ݼ��Ľ��
file_name_result=['..\result_all\',dataname,'_ensembleMVLSD_para_result','.txt'];
file_id_result=fopen(file_name_result,'at+');
%��¼ÿ�ֽ��
file_name_mccv=['..\result_mccv\',dataname,'_para_',vail_type,'_cv','.txt'];
file_id_mccv=fopen(file_name_mccv,'at+');
fprintf(file_id_mccv,'the parameter k- %3d ,C1-%3.3f, lamda-%3.3f ,gamma-%3.3f  \r\n',k,C,lam,gamma);
%-----------------------------------
%--------------------------------------
%-------------------------------------
%--------��¼����--------------------

%-----------------ѵ������-----------------
%---------------------------------------
for i_iter=1:ktimes
    tic;%��¼ѵ��ʱ��
    for i_classone = 1:(sum_class-1)
        for i_classtwo = (i_classone+1):sum_class
         
            for i_view_one=1:(max(mat_sample_num)-1)
                for i_view_two=(i_view_one+1):max(mat_sample_num)
                    InputPar.view1selected=i_view_one;
                    InputPar.view2selected=i_view_two;
                    for i_ensemble=1:sub%��������
                        [train_binary_data,train_binary_label]=Sample_Genaration(dataset,vail_type,i_iter,i_classone,i_classtwo,bias_class,para_sampling);
                        Matrixized_sam=pre_save_sample(train_binary_data,train_binary_label,mat_sample_num,mat_sample_way,InputPar);
                        [L] = Eweight_fun(train_binary_data,train_binary_label,InputPar.k);%����k���ھ���
                    [EnsMV(i_classone,i_classtwo).candidate{i_ensemble}.view_result{i_view_one,i_view_two},EnsMV_u_v_b(i_classone,i_classtwo).candidate{k}.view_result{i_view_one,i_view_two},covar_res]=MultiVLSDMatMHKS_fun(train_binary_data,train_binary_label,Matrixized_sam,InputPar,L,mat_sample_num,mat_sample_way);%����ѵ��������u��v
                    if strcmp(covar_res,'reject')==1
                        global_covflag=covar_res;
                    end
                    end%ensemble
                end%1end view2
            end%end view
        end%end class2
    end%end class1
    time(ktimes) = toc;%end
%testing

[test_data_final,test_label]=TestSample_Genaration(dataset,vail_type,i_iter);
matrix_vote = zeros(length(test_label),1);%����ͶƱ����ÿһ����һ��Group��ѡ����1������Ʊ��ͳ��
        for i_testone = 1:(sum_class-1)
            for i_testtwo = (i_testone+1):sum_class
                %���δ���������ѵ����õ����ݽ��в���       
                for i_ensemble=1:sub
                    for i_view_one=1:(max(mat_sample_num)-1)
                        for i_view_two=(i_view_one+1):max(mat_sample_num)
                            view_par=[i_view_one i_view_two];
                    [Group,dv] = MultiVLSDMatMHKS_test(EnsMV(i_classone,i_classtwo).candidate{i_ensemble}.view_result{i_view_one,i_view_two},test_data_final,i_testone,i_testtwo,mat_sample_num,mat_sample_way,view_par);
                    matrix_vote = cat(2,matrix_vote,Group);%�����
                    clear Group;
                        end%view2
                    end%view1
                end%ensemble k
            end%for_i_testtwo
        end%for_i_testone 
        
%         i_candidate = (sum_class)*(sum_class-1)/2;
%         for i_poll = 1:length(test_label)
%             vector_vote = matrix_vote(i_poll,2:(i_candidate+1));
%             matrix_vote(i_poll,1) = mode(vector_vote);
%         end%for_i_poll
        
        for i_poll = 1:length(test_label)
            vector_vote = matrix_vote(i_poll,2:end);
            matrix_vote(i_poll,1) = mode(vector_vote);
        end%for_i_poll
        
        accuracy(i_iter,1) = 100*(1-(length(find((test_label - matrix_vote(:,1))~=0))/length(test_label)));
        [~,tempt_location] = unique(test_label);%���ز��ظ�Ԫ�صĸ�����[a,b]=unique(A),a��������A�в��ظ���Ԫ�أ�ÿ��һ����b���ص�һ����ͬԪ�ص�λ��
           tempt_location1=[0;tempt_location];
           [tem_clas,~]=unique(test_label);
           for i_num=1:sum_class
               tem_num=find(test_label==tem_clas(i_num));
               class_cur_num(i_num)=length(tem_num);
           end
          for i=1:sum_class
             AUC(i_iter)=AUC(i_iter)+100*(1-(length((find((test_label(find(test_label==i))-matrix_vote(find(test_label==i),1))~=0)))/class_cur_num(i)));
%           AUC(i_iter)=AUC(i_iter)+100*(1-(length(find((test_label(1+tempt_location1(i):tempt_location1(i+1)) - matrix_vote(1+tempt_location1(i):tempt_location1(i+1),1))~=0))/length(test_label(1+tempt_location1(i):tempt_location1(i+1)))));
%              GMM(i_iter)=GMM(i_iter)*(1-(length(find((test_label(1+tempt_location1(i):tempt_location1(i+1)) - matrix_vote(1+tempt_location1(i):tempt_location1(i+1),1))~=0))/length(test_label(1+tempt_location1(i):tempt_location1(i+1)))));
             GMM(i_iter)=GMM(i_iter)*(1-(length((find((test_label(find(test_label==i))-matrix_vote(find(test_label==i),1))~=0)))/class_cur_num(i)));
          end
          AUC(i_iter)=AUC(i_iter)/sum_class;
          GMM(i_iter)=100*GMM(i_iter)^(1/sum_class);
           [~,~,~,T_AUC(i_iter)]=perfcurve(test_label,dv,'1');
           disp(['The present accuracy of ',num2str(i_iter),' iteration in MCCV is: ',num2str(accuracy(i_iter))]);%��ӡ����Ļ��
        disp(['The present AUC of ',num2str(i_iter),' iteration in MCCV is: ',num2str(AUC(i_iter))]);
        disp(['The present TAUC of ',num2str(i_iter),' iteration in MCCV is: ',num2str(T_AUC(i_iter))]);
        disp(['The present GMM of ',num2str(i_iter),' iteration in MCCV is: ',num2str(GMM(i_iter))]);
        disp(['------']);
        
        %------------------------------------
        fprintf(file_id_mccv,' %s   %3.3f   %3.3f   %3.3f\r\n',covar_res,accuracy(i_iter),AUC(i_iter),GMM(i_iter));
        %------------------------------------
        
end%end ktimes
    disp(['The average accuracy is: ',num2str(mean(accuracy))]);%��ӡ����Ļ��
    disp(['The average AUC is: ',num2str(mean(AUC))]);%��ӡ����Ļ��
    disp(['The average true AUC is: ',num2str(mean(T_AUC))]);%��ӡ����Ļ��
    disp(['The average GMM is: ',num2str(mean(GMM))]);%��ӡ����Ļ��
    disp(['The std of accuracies is: ',num2str(std(accuracy))]);
    disp(['The std of AUC is: ',num2str(std(AUC))]);%��ӡ����Ļ��
    disp(['The std of GMM is: ',num2str(std(GMM))]);%��ӡ����Ļ��
    disp(['The average time(s) is: ',num2str(mean(time))]);
    disp(['--------------------------------------']);
    fprintf(file_id_result,'%s %3d  %3.3f  %3.3f  %3.3f  %3.3f    %3.3f  %3.3f  %3.3f  %3.3f  %3.3f  %3.3f %3.3f \r\n',global_covflag,k,C,lam,gamma,mean(accuracy),mean(AUC),mean(GMM),std(accuracy),std(AUC),std(GMM),mean(time),mean(T_AUC));
    fprintf(file_id_mccv,'the acc of mean is %3.3f, auc of mean is %3.3f\r\n',mean(accuracy),mean(AUC));
%     results_detail(1,1:end)=[k,C,C2,lam,gamma,accuracy',mean(accuracy)];

    fclose(file_id_mccv);
    delete file_id_mccv;
    fclose(file_id_result);%���ļ�
    delete file_id_result;
end

    
