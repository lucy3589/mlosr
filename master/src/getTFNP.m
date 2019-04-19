function [tn, tp, fn, fp] = getTFNP(label, pred, tn, tp, fn, fp)
    
	if(label==-1)
        if(label==pred)
            tn=tn+1;
        else
            fp=fp+1;
        end
    else
        if(label==pred)
            tp=tp+1;
        else
            fn=fn+1;
        end
    end
    
end