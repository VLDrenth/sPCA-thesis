%% Set-up of forecasting exercise
h   = 1;   % Forecast horizon
K   = 5;   % Number of (scaled) principal components
p   = 300; % Initial estimation size

%% Load and process data
load('MacroTarget.mat');
load('FRED.mat');

%log_cpi=log(cpi_level(:,2));
%dlog_cpi=log_cpi(2:end)-log_cpi(1:end-1);
%cpidates=cpi_level(2:end,1);
 
log_ip=log(ip_level(:,2));
dlog_ip=log_ip(2:end)-log_ip(1:end-1);
ipdates=ip_level(2:end,1);

%tt1=(cpidates(:,1)>=196001 & cpidates(:,1)<=201912);
tt1=(ipdates(:,1)>=196001 & ipdates(:,1)<=201912);
tt2=(yymm(:,1)>=196001 & yymm(:,1)<=201912);

%Y=dlog_cpi(tt1,1);
%Y=test_y;
Y=dlog_ip(tt1,1);
X=macro_nm2(tt2,:);

% Determine the size of the data
[T,N] = size(X);

%% Forecasting exercise
YPred = NaN(T,3);
for t = p:T-1
    disp(['Forecast ',num2str(t+1-p),' of ',num2str(T-p)])
    % Obtain estimation data
    Xt =  X(1:t,:);
    Yt  = Y(1:t);
    [Tt,N]  = size(Xt);

    % Normalize data
    MXt = mean(Xt,'omitnan');
    WXt = std(Xt,'omitnan');
    XNt = (Xt-repmat(MXt,Tt,1))./repmat(WXt,Tt,1);  % Standardize series
    
    % Conduct standard PCA
    if Tt<N 
        % Singular value decomposition
        [ev,~,~]=svd(XNt*XNt'); 
        
        % Components
        PC = sqrt(Tt)*ev; 
    else 
        % Singular value decomposition
        [ev,~,~]=svd(XNt'*XNt);

        % Loadings
        Lambda0=sqrt(N)*ev;

        % Components
        PC = XNt*Lambda0/N;
    end

    % Conduct scaled PCA
    for i = 1:N
        gamma_i = [ones(Tt-h,1),XNt(1:end-h,i)]\Yt(h+1:end);
        gamma(i) = gamma_i(2);
    end
    [gamma_win, argout_n]=winsor(abs(gamma),[0 90]);
    sXNt = gamma_win.*XNt;
   if Tt<N 
        % Singular value decomposition
        [ev,eigval,~]=svd(sXNt*sXNt'); 
        
        % Components
        sPC = sqrt(Tt)*ev; 
    else 
        % Singular value decomposition
        [ev,eigval,~]=svd(sXNt'*sXNt);

        % Loadings
        Lambda0=sqrt(N)*ev;

        % Components
        sPC = sXNt*Lambda0/N;
    end
    
    % Estimate factor-augmented regressions
    bAR =  [ones(Tt-h,1),Yt(1:end-h)]\Yt(1+h:end);
    bPC  = [ones(Tt-h,1),Yt(1:end-h),PC(1:end-h,1:K)]\Yt(1+h:end);
    bsPC = [ones(Tt-h,1),Yt(1:end-h),sPC(1:end-h,1:K)]\Yt(1+h:end);
    
    YPred(t+1,1) = [1,Yt(end)]*bAR;
    YPred(t+1,2) = [1,Yt(end),PC(end,1:K)]*bPC;
    YPred(t+1,3) = [1,Yt(end),sPC(end,1:K)]*bsPC;
    disp(YPred(t+1, 1))
end

%%
MSPE_AR  = mean((Y(p+1:end) - YPred(p+1:end,1)).^2);
MSPE_PC  = mean((Y(p+1:end) - YPred(p+1:end,2)).^2);
MSPE_sPC = mean((Y(p+1:end) - YPred(p+1:end,3)).^2);

R_sq_PC  = (1 - MSPE_PC/MSPE_AR)*100;
R_sq_sPC = (1 - MSPE_sPC/MSPE_AR)*100;