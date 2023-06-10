function R2oos_stat=R2oostest(r,r_bar,r_hat,Nlag)
% This function is to test whether the forecast of r_hat outperform the
% historcal sample mean by calculating R2_OS and the significance/p_value with CW's adjusted-MSPE
% Nlag, to adjust overlapping data
% r: realized return out of sample
% r_bar: sample mean
% r_hat: estimate with other approach

T = length(r);
for t = 1:T %cumulative out of sample mean squared error
 se_R(t) = (r_bar(1:t)-r(1:t))'*(r_bar(1:t)-r(1:t)); % with sample mean
 se_U(t) = (r_hat(1:t)-r(1:t))'*(r_hat(1:t)-r(1:t));
 f_hat(t) = (r(t)-r_bar(t))^2 - (r(t)-r_hat(t))^2+(r_bar(t)-r_hat(t))^2;%compute MSPE-adjusted
end
%plot(1:T,se_R-se_U);

R2os = 100*(1 - se_U(end)/se_R(end));

% ENC_hat = T*(se_R(end) -(r_hat-r)'*(r_bar-r))/se_U(end);
% MSE_hat = T*(se_R(end)-se_U(end))/se_U(end);
ENC_hat = (T-Nlag+1)*(se_R(end) -(r_hat-r)'*(r_bar-r))/se_U(end);
MSE_hat = (T-Nlag+1)*(se_R(end)-se_U(end))/se_U(end);


%[b_hat,bse_hat,bt_hat,r2_hat,F_hat] = regress_jc(f_hat',[ones(T,1)],0);
%[B,TSTAT,S2,VCVNW,R2,RBAR,YHAT] = olsnw(f_hat',[ones(T,1)],0,Nlag);
[bv,sebv,R2v,R2vadj,v,F]=olsgmm(f_hat',[ones(T,1)],Nlag,1) ;


MSPE = 1-normcdf(bv/sebv);
%MSPE = 1-normcdf(TSTAT);

%R2oos_stat = [R2os, MSPE,  ENC_hat MSE_hat bv/sebv];
%R2oos_stat = [R2os, bv/sebv, MSPE];
R2oos_stat = [R2os, MSPE];
