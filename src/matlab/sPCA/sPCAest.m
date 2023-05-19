% ?Please cite the following paper:
% ?Huang D., Jiang F., Li K., Tong G. and Zhou G., "Scaled PCA: A New Approach to Dimension Reduction", Management Science 68(3), 2022, 1591-2376. 

function f=sPCAest(target, X, nfac)

%   Input variables:
%   ---------------------------------------------------------
%   X               = T by N Matrix of X variables values
%   target          = T by 1 vector of Y variable values
%   nfac            = number of sPCA factors to be extracted

%   Output
%   f               = sPCA factors

T=length(target);
if (length(X) ~= T)
    error('X and Y varibles not of equal length');
end

% standardize X to Xs
Xs=standard(X);
beta=NaN(1,size(Xs,2));
for j=1:size(Xs,2)
    xvar=[ones(T,1) Xs(:,j)];
    parm=xvar\target;
    beta(j)=parm(2);
end

% one can choose to winsorize to remove extreme values
% beta_win=winsor(abs(beta),[0 99]);

scaleXs=NaN(size(Xs,1),size(Xs,2));
for j=1:size(Xs,2)
     scaleXs(:,j)=Xs(:,j)*beta(j); 
%    scaleXs(:,j)=Xs(:,j)*beta_win(j);  
end

[~,f,~,~, ~]=pc_T(scaleXs,nfac);


%% sub functions
% principal components with normalization F'F/T=I

% X is observed

% r is the true number of true factors

% F is T by r matrix of true factors

% Lambda N by r is the true loading matrix

% C=F*Lambda' T by N is the true common component

% chat is the estimated common component

function [ehat,fhat,lambda,ve2,ss]=pc_T(y,nfac) %adding ve2

[bigt,bign]=size(y);

yy=y*y';

[Fhat0,eigval,Fhat1]=svd(yy); %for semi-def symmetric matrix, same as eig value decomposition but sorts in descending order

fhat=Fhat0(:,1:nfac)*sqrt(bigt);

lambda=y'*fhat/bigt;

%chi2=fhat*lambda';

%diag(lambda'*lambda)

%diag(fhat'*fhat)                % this should equal the largest eigenvalues

%sum(diag(eigval(1:nfac,1:nfac)))/sum(diag(eigval))

%mean(var(chi2))                 % this should equal decomposition of variance



ehat=y-fhat*lambda';



ve2=sum(ehat'.*ehat')'/bign;

ss=diag(eigval);


function[y,varargout] = winsor(x,p)
% WINSOR     Winsorize a vector 
% INPUTS   : x - n*1 data vector
%            p - 2*1 vector of cut-off percentiles (left, right) 
% OUTPUTS  : y - winsorized x, n*1 vector
%            i - (optional) n*1 value-replaced-indicator vector
% NOTES    : Let p1 = prctile(x,p(1)), p2 = prctile(x,p(2)). (Note
%            that PRCTILE ignores NaN values). Then 
%            if x(i) < p1, y(i) = min(x(j) | x(j) >= p1)
%            if x(i) > p2, y(i) = max(x(j) | x(j) <= p2)
% EXAMPLE  : x = rand(10,1), y = winsor(x,[10 90])
% AUTHOR   : Dimitri Shvorob, dimitri.shvorob@vanderbilt.edu, 4/15/07

if ~isvector(x)
   error('Input argument "x" must be a vector')
end  
if nargin < 2
   error('Input argument "p" is undefined')
end 
if ~isvector(p)
   error('Input argument "p" must be a vector')
end  
if length(p) ~= 2
   error('Input argument "p" must be a 2*1 vector')
end  
if p(1) < 0 || p(1) > 100
   error('Left cut-off percentile is out of [0,100] range')
end  
if p(2) < 0 || p(2) > 100
   error('Right cut-off percentile is out of [0,100] range')
end  
if p(1) > p(2)
   error('Left cut-off percentile exceeds right cut-off percentile')
end  
p = prctile(x,p);
i1 = x < p(1); v1 = min(x(~i1));
i2 = x > p(2); v2 = max(x(~i2));
y = x;
y(i1) = v1;
y(i2) = v2;
if nargout > 1
   varargout(1) = {i1 | i2};
end   

function x=standard(y)
T=size(y,1);
%N=size(y,2);
my=repmat(mean(y),T,1);
sy=repmat(std(y),T,1);
x=(y-my)./sy;