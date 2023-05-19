function [p_star]=Select_AR_lag_SIC(y,h,p_max)

% Please note that this code comes with no performance guarantees.
% User assumes all risks!
%
% Last modified: 01-24-2016
%
% Input
%
% y     = T-vector of dependent variable observations
% h     = horizon
% p_max = maximum AR lag length
%
% Output
%
% p_star = lag length selected by SIC

T=size(y,1);
y_h=zeros(T-(h-1),1);
for t=1:T-(h-1);
    y_h(t)=mean(y(t:t+(h-1)));
end;
y=y(1:size(y_h,1));
y_h=y_h(p_max+1:end);
lag_order=(0:1:p_max)';
y_lags=[];
for j=1:p_max;
    y_lags=[y_lags y(p_max-(j-1):T-j-(h-1))];
end;
T=size(y_h,1);
SIC=zeros(p_max+1,1);
for j=1:p_max+1;
    if j==1;
        Z=ones(T,1);
    else
        Z=[ones(T,1) y_lags(:,1:j-1)];
    end;
    a_hat=inv(Z'*Z)*(Z'*y_h);
    e_hat=y_h-Z*a_hat;
    SIC(j)=log((e_hat'*e_hat)/T)+log(T)*length(a_hat)/T;
end;
[~,index_min]=min(SIC);
p_star=lag_order(index_min);
