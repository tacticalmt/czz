function [combine_mat_num,r_valid] = Matrixlize_fun(vec_num)

%���ھ��󻯵ĺ���
%vec_num�������Ŀ������
%combine_mat_num����ž��󻯿�����������͵ľ��󣨸ú�������ֵ��
%r_valid����Ч����
%r_vec_num��Ŀ����������
%c_vec_num��Ŀ����������
%sum_vec_num��Ŀ���������ȣ���Ϊ��һά���������л�������
%zeromat_num��������Ԫ�����ģ��ֵ
%tempt_value���ж��Ƿ��������м�ֵ
%r_combine_num:���󻯺�������
%c_combine_num:���󻯺�������
%m��nѭ�����Ʊ���
%mat_num:���󻯺����ɵľ���

while(1),
    [r_vec_num,c_vec_num]=size(vec_num); %�����飨������vec_num��������ֵ��r_vec_num��������ֵ��c_vec_num

    if(r_vec_num+c_vec_num == 2) %����������һ�����������������򱨴�
        disp('The input value is not a vector!It is a number! ');
        break;%����ѭ��
    end

    sum_vec_num = 0;    %�ж��ǲ���һά�����������һά�����Ͱѳ��ȸ���sum_vec_num��������һά�����ͱ���
    if(c_vec_num == 1)
        sum_vec_num = r_vec_num;
    elseif(r_vec_num == 1)
        sum_vec_num = c_vec_num;
    else
        disp('The input value is not a vector! It is a matrix! ');
        break;%����ѭ��
    end

    zeromat_num = sum_vec_num;    %������Ԫ����Ĵ�С
    combine_mat_num = zeros(zeromat_num,2); %����һ����Ԫ������zeromat_num�к�2�У���һ�д�������Ŀ�����ϣ��ڶ��д�������Ŀ�����ϣ��������������
    r_valid = 0;%������Ч����
    %for n = 2:(sum_vec_num-2) %����һ��1��sum_vec_num����������󻯣���sum_vec_num-1���Բ��ᱻsum_vec_num����
    for n = 1:sum_vec_num %��������������󻯺��������
         tempt_value = mod(sum_vec_num,n);%�жϵ�ǰ��n�Ƿ�������sum_vec_num�������tempt_value
        if(tempt_value == 0)%��������������������ͽ�ֱ���Ϊ�¹��ɾ����������������������ϴ���ղŴ����ľ���combine_mat_num��
            r_valid = r_valid+1;%��¼��Ч���������������
            r_combine_num = sum_vec_num/n;
            c_combine_num = n;
            combine_mat_num(r_valid,1) = r_combine_num;
            combine_mat_num(r_valid,2) = c_combine_num;
        else
            continue;
        end %end_if
    end %end_for

    %disp([r_valid]); %������
    
    if(r_valid == 0)  %���û���ҵ���Ч����Ϸ����ͱ���
        disp('This vector can not be matrixlized!');
        break;%����ѭ��
%    else %������Ϸ���
%        for m = 1:r_valid     %��ʾ���󻯺�ĸ�������
%            mat_num = reshape(vec_num,combine_mat_num(m,1),combine_mat_num(m,2));
%            disp(['�� ',num2str(m),' �ַֽⷽ���ǣ�']);
%            disp([mat_num]);
%        end
    end

    %y=1;%�����ã�Ϊ�˷��غ�����ֵ
    break;
end