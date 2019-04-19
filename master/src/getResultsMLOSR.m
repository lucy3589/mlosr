clc; close all;clear all;

%% initialization
dataset_name = 'cifar10';
view_error_histogram=1;
hist_bins=500;
worked_flag=1;
prior0=0.5;

%% loading data
load(['../../save_folder/results/' dataset_name '/mlosr_scores.mat']);
load(['../../save_folder/results/' dataset_name '/mlosr_mse.mat']);
load(['../../save_folder/results/' dataset_name '/label.mat']);

label=label';
mlosr_mse=mlosr_mse';

pred_evt=zeros(length(mlosr_mse),1);
p_mlosr_evt=zeros(length(mlosr_mse),1);

acc_mlosr_evt=[];fm_mlosr_evt=[];
tp_mlosr_evt=zeros(1,1);tn_mlosr_evt=zeros(1,1);fp_mlosr_evt=zeros(1,1);fn_mlosr_evt=zeros(1,1);

kwn = mlosr_mse(label~=-1);
unk = mlosr_mse(label==-1);

x = (min(mlosr_mse)-0.05):0.012:(max(mlosr_mse)+0.05);
hist_unk = hist(unk,x)/length(unk);
hist_kwn = hist(kwn,x)/length(kwn);

hist_overlap = sum(min([hist_unk;hist_kwn]))/length(hist_unk);

if(view_error_histogram)
    figure;
    histogram(unk,x,'FaceColor','r','LineStyle','--','EdgeAlpha',0.1,'Normalization','probability');
    hold on;
    histogram(kwn,x,'FaceColor','g','EdgeAlpha',0.1,'Normalization','probability');
    legend('Unknown','Known');
    xlim([0 max(unk)]);
    pause(1)
end



%% training GPD
val_match = sort(kwn,'descend');
tailsize_match = 30;

gpd_para_match = gpfit(val_match(1:tailsize_match)-val_match(tailsize_match)+eps);

%% estimating the threshold
thr_prob_evt = 0.5;

prob_vals = 1-gpcdf(mlosr_mse-val_match(tailsize_match)+eps, gpd_para_match(1), gpd_para_match(2),0);
prob_vals = prob_vals/max(prob_vals);
id = find(prob_vals<=thr_prob_evt);
thr_rec_evt = mlosr_mse(id(1));


%% testing on data
for i=1:length(mlosr_mse)
    
    % get softmax probs
    s_mlosr = mlosr_scores(i,:);
    [mlosr_score, mlosr_id] = max(s_mlosr);
    
    % get mlosr match score for evt
    p_match = (1-gpcdf(mlosr_mse(i), gpd_para_match(1),gpd_para_match(2),0));
    p_mlosr_evt(i) = p_match;
    
    % lcos predictions with evt
    if(mlosr_mse(i)<=thr_rec_evt)
        pred_evt(i) = mlosr_id-1;
    else
        pred_evt(i) = -1;
    end
    

end


%% get error stats
for i=1:length(mlosr_mse)
    % tp, tn, fp, fn for lcos evt
    [tn_mlosr_evt, tp_mlosr_evt,...
        fn_mlosr_evt, fp_mlosr_evt] = getTFNP(label(i), pred_evt(i), tn_mlosr_evt,...
                               tp_mlosr_evt, fn_mlosr_evt, fp_mlosr_evt);
    
end


%% get fmeasure and accuracy
[fm_mlosr_evt, ~, ~] = getPRF(tp_mlosr_evt, fp_mlosr_evt, fn_mlosr_evt, tn_mlosr_evt);
acc_mlosr_evt=sum(pred_evt==label)/length(mlosr_mse);

%% display results
disp(['F-measure is (ours)  : ' num2str(fm_mlosr_evt)])

