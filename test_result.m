clc
clear
load('PU_alff_ds.mat')
x = xlsread('C:\Users\admin\Desktop\PU_score_prediction_result\output.xlsx');

label_p = x(:,3:2:end);
score_p = x(:,4:2:end);

label_g = x(:,1);   %groundtruth
score_g = x(:,2);   %groundtruth


% MMSE for all subject
tmp_ = score_p - score_g*ones(1,size(score_p,2));
tmp_ = tmp_.^2;
MMSE = mean(tmp_(:));
% NMSE =mean(sqrt(sum(tmp_(:))))

tmp_ = score_p - score_g*ones(1,size(score_p,2));
MAE = mean(abs(tmp_(:)));

[a ,b] = corr(score_g,mean(score_p,2))

%  MMSE for all ADHD subject
tmp_ = score_p(label_g==0,:) - score_g(label_g==0,:)*ones(1,size(score_p,2));
tmp_ = tmp_.^2;
MMSE_ADHD = mean(tmp_(:));





% 
% % for i=1:size(label_p,1)
% %     [~,nn]=find(label_p(i,:)==label_avg(i));
% %     score_avg(i)=mean(score_p(i,nn)) ;
% % end
% 
% label_p = x(label_avg==0,3:2:end);
% score_p = x(label_avg==0,4:2:end);
% score = score(label_avg==0);
% 
% % for i=1:size(label_p,1)
% %     score_avg(i)=mean(score_p(i,:)) ;
% % end
% 
% 
% 
% subplot(2,1,1)
% bar(score)
% subplot(2,1,2)
% bar(score_avg,'r')
% 
% mean((score'-score_avg).^2)
% mean(abs(score'-score_avg))
% 
% aaa=1