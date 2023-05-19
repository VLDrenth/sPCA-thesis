function [a_hat res]=Estimate_AR_res(y,h,p)

% Please note that this code comes with no performance guarantees.
% User assumes all risks!
%
% Last modified: 01-14-2016
%
% Input
%
% y = T-vector of dependent variable observations
% h = horizon
% p = AR lag length
%
% Output
%
% a_hat = (p+1)-vector of AR OLS coefficient estimates (intercept first)

T=size(y,1);
y_h=zeros(T-(h-1),1);
for t=1:T-(h-1);
    y_h(t)=mean(y(t:t+(h-1)));
end;
y_h=y_h(p+1:end);
y_lags=[];
for j=1:p;
    y_lags=[y_lags y(p-(j-1):T-j-(h-1))];
end;
Z=[ones(size(y_h,1),1) y_lags];
a_hat=inv(Z'*Z)*(Z'*y_h);
res=y_h-Z*a_hat;