function [c_hat]=Estimate_ARDL_multi(y,z,h,p)

% Please note that this code comes with no performance guarantees.
% User assumes all risks!

% Last modified: 01-24-2016

% Estimates intercept/slope parameters for the ARDL(p1,p2) model
%
% y(t,h) = a(0) + a(1)*y(t-1) + ... + a(p1)*y(t-p1) + ...
%          b(1)*z(t-1) + ... + b(p2)*z(t-p2) + e(t,h),
%
% where
%
% y(t,h) = horizon-h mean of y = (1/h)*sum_(j=1)^(h)y(t+(j-1)) 
%
% Input
%
% y = vector of y observations
% z = vector of z observations
% h = forecast horizon
% p = vector of ARDL lags (p1,p2)
%
% Output
%
% c_hat = (p1+p2+1)-vector of coefficient estimates
%         [a(0),a(1),...,a(p1),b(1),...,b(p2)]

% Take care of Preliminaries
sz=size(z,2);
T=size(y,1);
y_h=zeros(T-(h-1),1);
for t=1:T-(h-1);
    y_h(t)=mean(y(t:t+(h-1)));
end;

% Create regressand/regressors

p1=p(1);
p2=p(2);
p_max=max([p1 ; p2]);
y_h=y_h(p_max+1:end);
y_lags=nan(length(y_h),p_max);
z_lags=nan(length(y_h),p_max*sz);
for j=1:p_max;
    y_lags(:,j)=y(p_max-(j-1):T-j-(h-1));
    z_lags(:,(j-1)*sz+1: j*sz)=z( p_max-(j-1):T-j-(h-1) , :);
end;
if p1==0;
    Z=[ones(length(y_h),1) z_lags];
else
    Z=[ones(length(y_h),1) y_lags(:,1:p1) z_lags(:,1:p2*sz)];
end;

% Estimating parameters

c_hat=inv(Z'*Z)*(Z'*y_h);