clear;clc;
load('MacroTarget.mat');
load('FRED.mat');

log_cpi=log(cpi_level(:,2));
dlog_cpi=log_cpi(2:end)-log_cpi(1:end-1);
cpidates=cpi_level(2:end,1);

tt1=(cpidates(:,1)>=196001 & cpidates(:,1)<=201912);
tt2=(yymm(:,1)>=196001 & yymm(:,1)<=201912);

y=dlog_cpi(tt1,1);
%y=standard(y);
z=macro_nm2(tt2,:); 

%in-sample
out=cell(1,4);
out_table=cell(1,4);
loadings=cell(1,3);
horizon=[1];
maxp=[3];
kn=15;
for kk=1:1
h=horizon(kk);
Zs=standard(z);

    T=size(y,1);
    y_h=zeros(T-(h-1),1);
    for t=1:T-(h-1);
    y_h(t)=mean(y(t:t+(h-1)));
    end;

    p_max=maxp(kk);
    [p_AR_star_n]=Select_AR_lag_SIC(y,h,p_max);
    %p_AR_star_n=1;
    [a_hat res_h]=Estimate_AR_res(y,h,p_AR_star_n);
    
beta=NaN(1,size(Zs,2));
tstat=NaN(1,size(Zs,2));
for j=1:size(Zs,2)
    [parm,~,t_stats]=linear_reg( y_h(2:end), Zs(1:end-1-(h-1),j) ,1,'NW',h);
    %[parm,~,t_stats]=linear_reg( res_h(1:end), Zs(1:end-1-(h-1)-(p_AR_star_n-1),j) ,1,'NW',h);
    %[parm,~,t_stats]=linear_reg( y(2:end), Zs(1:end-1,j) ,1,'NW',1);
    beta(j)=parm(2);
    tstat(j)=t_stats(2);
end


beta_win=winsor(abs(beta),[0 100]);
scaleZs=NaN(size(Zs,1),size(Zs,2));
for j=1:size(Zs,2)
    scaleZs(:,j)=Zs(:,j)*beta_win(j);    
end


[~,z_pc,~,~, ~]=pc_T(Zs,kn);  [~,loadings_pc,var_explained_pc] = pca_eig(Zs,kn,'No');
[~,z_spc,~,~, ~]=pc_T(scaleZs,kn); [~,loadings_spc,var_explained_spc] = pca_eig(scaleZs,kn,'No');
if h==1
    loadings{1}=loadings_pc;
    loadings{2}=loadings_spc;
    loadings{3}=beta_win;
end
% [~,z_target1,~,~, ~]=pc_T(targetZs{1},10); [~,z_target2,~,~, ~]=pc_T(targetZs{2},10);

z_pc=standard(z_pc);  
z_spc=standard(z_spc);  
% z_target1=standard(z_target1); z_target2=standard(z_target2); 

adr2_pc=nan(kn,1);
adr2_spc=nan(kn,1);
% adr2_target1=nan(10,1);
% adr2_target2=nan(10,1);

for ll=1:kn
[b_pc,~,tb_pc,r2_pc,adr2_pc(ll)] = linear_reg(res_h(1:end),[z_pc(1+(p_AR_star_n-1):end-1-(h-1),1:ll) ],0,'NW',h);
[b_spc,~,tb_spc,r2_spc,adr2_spc(ll)] = linear_reg(res_h(1:end),[z_spc(1+(p_AR_star_n-1):end-1-(h-1),1:ll)],0,'NW',h);
% [b_target1,~,tb_target1,r2_target1,adr2_target1(ll)] = linear_reg(res_h(1:end),[z_target1(1:end-1-(h-1)-(p_AR_star_n-1),1:ll) ],1,'NW',h);
% [b_target2,~,tb_target2,r2_target2,adr2_target2(ll)] = linear_reg(res_h(1:end),[z_target2(1:end-1-(h-1)-(p_AR_star_n-1),1:ll) ],1,'NW',h); 
end

out{kk}=[adr2_pc adr2_spc]*100;
out_table{kk}=[var_explained_pc(1:kn) var_explained_spc(1:kn)];
end
output=[out{1}; out{2}; out{3}; out{4}];


output=roundn(output,-2);

