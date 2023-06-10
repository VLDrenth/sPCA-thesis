clear;clc;
load('MacroTarget.mat');
load('FRED.mat');

log_cpi=log(cpi_level(:,2));
dlog_cpi=log_cpi(2:end)-log_cpi(1:end-1);
cpidates=cpi_level(2:end,1);

tt1=(cpidates(:,1)>=196001 & cpidates(:,1)<=201912);
tt2=(yymm(:,1)>=196001 & yymm(:,1)<=201912);

y=dlog_cpi(tt1,1);
z=macro_nm2(tt2,:); 
%z=z(:,[1 2 3 4 7 8 11:end]);

% out-of-sample
out=cell(1,4);
horizon=[1 3 6 12];
maxp=[1 3 3 3];
for kk=1:1
h=horizon(kk); % forecast horizon
T=length(y);
M=(1984-1959)*12; % 1961:07-1979:12 in-sample period
N=T-M; % 1980:01-2017:12 out-of-sample period
FC_AR=nan(N-(h-1),1); % AR forecast
%FC_PM=nan(N-(h-1),1); % prevailing mean forecast
% FC_PCA=nan(N-(h-1),1); % diffusion index forecast
% FC_tPCA1=nan(N-(h-1),1); 
% FC_tPCA2=nan(N-(h-1),1); 
% FC_sPCA=nan(N-(h-1),1); 
FC_ARDL_DI=nan(N-(h-1),10); % diffusion index forecast
FC_PCA=nan(N-(h-1),10);
FC_sPCA=nan(N-(h-1),10);
p_max=maxp(kk); % max lag
%p=[0 1]'; % predictive regression = ARDL(0,1)
actual_y=nan(N-(h-1),1);
tic;
for n=1:N-(h-1);
    actual_y(n)=mean(y(M+n:M+n+(h-1)));
    y_n=y(1:M+(n-1));
    %FC_PM(n)=mean(y_n);
    Z_n=z(1:M+(n-1),:);
    Zs_n=standard(Z_n);
    
    T_n=size(y_n,1);
    y_n_h=zeros(T_n-(h-1),1);
    for t=1:T_n-(h-1);
    y_n_h(t)=mean(y_n(t:t+(h-1)));
    end;
    
    [p_AR_star_n]=Select_AR_lag_SIC(y_n,h,p_max);
    %p_AR_star_n=1;
    if p_AR_star_n>0;
        %[a_hat_n]=Estimate_AR(y_n,h,p_AR_star_n);
         [a_hat_n res_n_h]=Estimate_AR_res(y_n,h,p_AR_star_n);
         y_n_last=flipud(y_n(end-(p_AR_star_n-1):end));
%          [a_hat_n res_n_h]=Estimate_AR_GT(y_n,h,p_AR_star_n);
%          y_n_last=flipud(y_n_h(end-(p_AR_star_n-1):end));
        FC_AR(n)=[1 y_n_last']*a_hat_n;
    else
        FC_AR(n)=mean(y_n);
    end;
    

%     T_n=size(y_n,1);
%     y_n_h=zeros(T_n-(h-1),1);
%     for t=1:T_n-(h-1);
%     y_n_h(t)=mean(y_n(t:t+(h-1)));
%     end;
    
    beta_n=NaN(1,size(Zs_n,2));
    tstat_n=NaN(1,size(Zs_n,2));
    for j=1:size(Zs_n,2)
        [parm_n,~,t_n,~,~,reg_se]=linear_reg(y_n_h(2:end),Zs_n(1:end-1-(h-1),j),1,'NW',h);
        %[parm_n,~,t_n,~,~,reg_se]=linear_reg(y_n(2:end),Zs_n(1:end-1,j),1,'NW',h);
        %[parm_n,~,t_n,~,~,reg_se]=linear_reg(res_n_h(1:end),Zs_n(1+(p_AR_star_n-1):end-1-(h-1),j),1,'NW',h);
    beta_n(j)=parm_n(2);
    tstat_n(j)=t_n(2);
    end
    
    
    %[beta_n_win, argout_n]=winsort(abs(beta_n),abs(tstat_n),[0 90]);
    [beta_n_win, argout_n]=winsor(abs(beta_n),[0 90]);
    transZs_n=NaN(size(Zs_n,1),size(Zs_n,2));
    for j=1:size(Zs_n,2)
    %transXs_n(:,j)=Xsplus_n(:,j)*beta_n_win(j);
    transZs_n(:,j)=Zs_n(:,j)*beta_n_win(j);
    end
    
         [~,z_pc_n,~,~, ~]=pc_T(Zs_n,10); 
         [~,z_trans_n,~,~, ~]=pc_T(transZs_n,10); 
         
    factor_n=cell(10,2); 
    for cc=1:10
    factor_n{cc,1}=z_pc_n(:,1:cc); 
    factor_n{cc,2}=z_trans_n(:,1:cc);
    end
    
    
    for jj=1:2
        for cc=1:10
            z_factor_n=factor_n{cc,jj};
            
%             if p_AR_star_n>0;
%                c_hat_n=linear_reg(res_n_h(1:end),z_factor_n(1+(p_AR_star_n-1):end-1-(h-1),:),0,'Skip',h);
%                FC_ARDL_DI(n,cc)=[z_factor_n(end,:)]*c_hat_n+FC_AR(n);
%             else
%                c_hat_n=linear_reg(y_n_h(2:end),z_factor_n(1:end-1-(h-1),:),1,'Skip',h);
%                FC_ARDL_DI(n,cc)=[1 z_factor_n(end,:)]*c_hat_n;
%             end;

            p_ARDL_star_DI_n=[p_AR_star_n 1];
            if p_AR_star_n>0;
               [c_hat_n]=Estimate_ARDL_multi(y_n,z_factor_n,h,p_ARDL_star_DI_n);
               %c_hat_n=linear_reg(y_n_h(2+(p_AR_star_n-1):end),[y_n(1+(p_AR_star_n-1):end-1-(h-1)) z_factor_n(1+(p_AR_star_n-1):end-1-(h-1),:)],1,'Skip',h);
               FC_ARDL_DI(n,cc)=[1 y_n_last' z_factor_n(end,:)]*c_hat_n;
            else
               c_hat_n=linear_reg(y_n_h(2:end),z_factor_n(1:end-1-(h-1),:),1,'Skip',h);
               FC_ARDL_DI(n,cc)=[1 z_factor_n(end,:)]*c_hat_n;
            end
        end
        
        if jj==1
            FC_PCA(n,:)=FC_ARDL_DI(n,:);
        else
            FC_sPCA(n,:)=FC_ARDL_DI(n,:);
        end
    end
 
%      b_pc_n = linear_reg(y_n_h(2:end),[z_pc_n(1:end-1-(h-1),:) ],1,'Skip',18);
%      b_trans_n = linear_reg(y_n_h(2:end),[z_trans_n(1:end-1-(h-1),:) ],1,'Skip',18);
%      b_target1_n = linear_reg(y_n_h(2:end),[z_target1_n(1:end-1-(h-1),:) ],1,'Skip',18);
%      b_target2_n = linear_reg(y_n_h(2:end),[z_target2_n(1:end-1-(h-1),:) ],1,'Skip',18);
%      
%      FC_PCA(n)=[1 z_pc_n(end,:)]*b_pc_n; 
%      FC_sPCA(n)=[1 z_trans_n(end,:) ]*b_trans_n;
%      FC_tPCA1(n)=[1 z_target1_n(end,:) ]*b_target1_n;
%      FC_tPCA2(n)=[1 z_target2_n(end,:) ]*b_target2_n;   
end
toc;
outpca=nan(10,2);outspca=nan(10,2);
for tt=1:10
    outpca(tt,:)=R2oostest(actual_y,FC_AR,FC_PCA(:,tt),h);
    outspca(tt,:)=R2oostest(actual_y,FC_AR,FC_sPCA(:,tt),h);
end

out{kk}=[outpca(:,1) outspca(:,1)];
end

output=[out{1}; out{2}; out{3}; out{4}];

output=roundn(output,-2)

figure;
subplot(2,2,1)
hold on
plot(1:10,output(1:10,1),'--b');
plot(1:10,output(1:10,2),'-r');
title('Panel A: Predict inflation at 1-month horizon','Interpreter','latex','FontSize',10);
ylabel('OOS-$R^2$ (\%)','Interpreter','latex');
legend('PCA','sPCA');

subplot(2,2,2)
hold on
plot(1:10,output(1:10,1),'--b');
plot(1:10,output(1:10,2),'-r');
title('Panel B: Predict inflation at 3-month horizon','Interpreter','latex','FontSize',10);
ylabel('OOS-$R^2$ (\%)','Interpreter','latex');

subplot(2,2,3)
hold on
plot(1:10,output(21:30,1),'--b');
plot(1:10,output(21:30,2),'-r');
title('Panel C: Predict inflation at 6-month horizon','Interpreter','latex','FontSize',10);
ylabel('OOS-$R^2$ (\%)','Interpreter','latex');

subplot(2,2,4)
hold on
plot(1:10,output(31:40,1),'--b');
plot(1:10,output(31:40,2),'-r');
title('Panel D: Predict inflation at 1-year horizon','Interpreter','latex','FontSize',10);
ylabel('OOS-$R^2$ (\%)','Interpreter','latex');

%save2pdf('Inflation_oos',gcf,600);

