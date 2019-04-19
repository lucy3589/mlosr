function [f, tpr, fpr] = getPRF(tp, fp, fn, tn)
    
    prcn = tp/(tp+fp);
    rcll = tp/(tp+fn);
    
    tpr = tp/(tp+fn);
    fpr = fp/(fp+tn);
    f = 2*( (prcn*rcll)/(prcn+rcll) );
    
end