%%This code is used for comparing the U-function and the EIER function
%%developed in the following paper
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Wei, P., Zheng, Y., Fu, J., Xu, Y., & Gao, W. (2023). An expected integrated error reduction function for accelerating            %%%%
%%   Bayesian active learning of failure probability. Reliability Engineering & System Safety, 231, 108971.                            %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%Initialized with the same set of N0 training points, the code first uses U-function for learning
%%Then the EIER function is utilized, but for this implementation, Ncand candidate samples are first pre-selected using the U-function, and then
%%the optimal one is refined with the EIER function from these Ncand canidate samples

clear
clc
warning off
g = @(x)10 - (x(:,1).^2-5*cos(0.8*pi*x(:,1)))-(x(:,2).^2-5*cos(0.8*pi*x(:,2)));


d=2;
N = 1e4;% size of sample pool
N0 = 12;%% initial training sample size
NGPR = 1000;%sample size of GPR moodel
Alpha = 0.95; % for computing the confidence interval of estimation
Ncand = 500;% number of candidate samples to be pre-selected using the U-function
XSamp = lhsnorm([0,0],[1,0;0,1],N); % creat sample pool
pf = mean(g(XSamp)<0);% compute the reference value 

%%%for plotting the training details
Ngrid = 1000;
Xgrid = linspace(-4,4,Ngrid);
[Xmesh1,Xmesh2] = meshgrid(Xgrid,Xgrid);
for i=1:Ngrid
    Ymesh(:,i) = g([Xmesh1(:,i),Xmesh2(:,i)]);
end

%%%%%initialize the two implementations with the same set of N0 training
%%%%%points randomly generated from the sample pool
Ind_train1 = randperm(N,N0)';
X0 = XSamp(Ind_train1,:);
Xtrain1 = X0;
Ytrain1 = g(X0);%%Xtrain1 used for initilizing the U-function based active learning
Ind_train2 = Ind_train1;
Xtrain2 = Xtrain1;
Ytrain2 = Ytrain1;%%Xtrain2 used for initializing the modified U-function active learning
Ntrain1 = N0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%start the training with U-function
U1 = 1e5*ones(N,1);
StopFlag1 = 0;
while (1==1)
   GPRmodel1 = fitrgp(Xtrain1,Ytrain1,'KernelFunction','ardsquaredexponential'...
    ,'BasisFunction','linear','Sigma', 1.001e-6, 'ConstantSigma', true, ...
    'SigmaLowerBound', eps);
   [ypred,ysd,~] = predict(GPRmodel1,XSamp);
   U1 = abs(ypred)./ysd;
   U1(Ind_train1,:) = 1e5;
   GPRSamp1 = GPRSampling(GPRmodel1,Ind_train1,Xtrain1,Ytrain1,XSamp,NGPR);%%sampling from the GPR model
   PfSamp1 = mean(GPRSamp1<0,1);
   PostCovPf1(Ntrain1-N0+1) = sqrt(var(PfSamp1))/mean(PfSamp1);
   fprintf('Current COV of Pf(U function)： %.4f\n',PostCovPf1(Ntrain1-N0+1));
   fprintf('Current Ncall（U function）： %d\n', Ntrain1);
   if PostCovPf1(Ntrain1-N0+1)<0.03
      StopFlag1 = StopFlag1+1;
   else
       StopFlag1 = 0;
   end
   if StopFlag1 == 2
       break;
   end
   [SortU1,Ind1] = sort(U1);
   Ind_train1 = [Ind_train1;Ind1(1)];
   Xtrain1 = [Xtrain1;XSamp(Ind1(1),:)];
   Ytrain1 = [Ytrain1;g(XSamp(Ind1(1),:))];
   Ntrain1 = Ntrain1+1;
end
GPRSamp1 = GPRSampling(GPRmodel1,Ind_train1,Xtrain1,Ytrain1,XSamp,NGPR);%%GPR conditional sampling
PfSamp1 = mean(GPRSamp1<0,1);

PostMeanPf1 = mean(PfSamp1);
PostCIPf1(1) = quantile(PfSamp1,(1-Alpha)/2);
PostCIPf1(2) = quantile(PfSamp1,Alpha/2+1/2);%credible inteval

%%%pre-select Ncand points, and then specify the optimal one using the EIER function
Ntrain2 = N0;
StopFlag2 = 0;
while (1==1)
   GPRmodel2 = fitrgp(Xtrain2,Ytrain2,'KernelFunction','ardsquaredexponential'...
    ,'BasisFunction','linear','Sigma', 1.001e-6, 'ConstantSigma', true, ...
    'SigmaLowerBound', eps);
   [ypred,ysd,~] = predict(GPRmodel2,XSamp);
   U = abs(ypred)./ysd;
   U(Ind_train2,:) = 1e5;
   Sigma0 = GPRmodel2.KernelInformation.KernelParameters(end);
   Sigmal = GPRmodel2.KernelInformation.KernelParameters(1:end-1);
   Sigman = GPRmodel2.Sigma;
   Beta0 = GPRmodel2.Beta;
   kfcn = @(XN,XM) Sigma0^2*exp(-(pdist2(XN,XM,'seuclidean',Sigmal).^2)/2)+ 1.001e-6.^2;%%kernel function
   K = kfcn(Xtrain2,Xtrain2);
   r_XSamp_Xtrain = kfcn(XSamp,Xtrain2);
   InvK = pinv(K);
   GPRSamp2 = GPRSampling(GPRmodel2,Ind_train2,Xtrain2,Ytrain2,XSamp,NGPR);%%sampling from the GPR mopdel
   PfSamp2 = mean(GPRSamp2<0,1);
   PostCovPf2(Ntrain2-N0+1) = sqrt(var(PfSamp2))/mean(PfSamp2);
   fprintf('Current COV of Pf(EIER function)： %.4f\n',PostCovPf2(Ntrain2-N0+1));
   fprintf('Current Ncall（EIER function）： %d\n', Ntrain2);
   if PostCovPf2(Ntrain2-N0+1)<0.03
      StopFlag2 = StopFlag2+1;
   else
       StopFlag2 = 0;
   end
   if StopFlag2 == 2
       break;
   end
   [Uinc,Uindex] = sort(U);
  
   %%%EIER acquisition function
   CandInd = Uindex(1:Ncand);
   AcqFun = zeros(N,1);
   parfor j=1:Ncand
       i = CandInd(j);
       if ismember(i,Ind_train2) == 1
          AcqFun(j) = 0;
       else%in case the i-th sample is not a training sample
          r_Train_Xi = kfcn(Xtrain2,XSamp(i,:)); %训练样本和第i个样本之间得协方差向量
          r_XSamp_Xi = kfcn(XSamp,XSamp(i,:));
          r_Xi_Xi = Sigma0^2; %第i个样本得自协方差
          Denominator = r_Xi_Xi - r_Train_Xi'*InvK*r_Train_Xi;
          K11 = InvK + InvK * (r_Train_Xi*r_Train_Xi')*InvK/Denominator;
          K12 = -InvK*r_Train_Xi/Denominator;
          K22 = 1/Denominator;
          InvKStar = [K11,K12;K12',K22];%第i个点加入训练集后Gram矩阵的逆
          r_updated = [r_XSamp_Xtrain,r_XSamp_Xi];
          PostVar = abs(Sigma0^2*ones(N,1) - diag(r_updated*InvKStar*r_updated'));%将第i个样本加入训练样本集后各备选样本的方差
          PostMean = repmat([ones(N,1),XSamp]*Beta0,1,NGPR) + r_updated*InvKStar*([repmat(Ytrain2,1,NGPR);GPRSamp2(i,:)]-repmat([ones(Ntrain2+1,1),[Xtrain2;XSamp(i,:)]]*Beta0,1,NGPR));
          ProbDecSamp = mean(max(repmat(normcdf(-U),1,NGPR) - normcdf(-abs(PostMean)./repmat(sqrt(PostVar),1,NGPR)),zeros(N,NGPR)),1);%总体判断错误率减小值的样本
          AcqFun(j) = mean(ProbDecSamp);%mean(max(ProbDecSamp,zeros(1,NGPR)));
       end
   end
   [~,Ind_max_AcqFun] = max(AcqFun); %找到学习函数值最大的点加入训练样本点
   Ntrain2 = Ntrain2 + 1;
   Ind_train2(Ntrain2,:) = CandInd(Ind_max_AcqFun);
   Xtrain2(Ntrain2,:) = XSamp(CandInd(Ind_max_AcqFun),:);
   Ytrain2(Ntrain2,:) = g(Xtrain2(Ntrain2,:)); 
end
PostMeanPf2 = mean(PfSamp2);
PostCIPf2(1) = quantile(PfSamp2,(1-Alpha)/2);
PostCIPf2(2) = quantile(PfSamp2,Alpha/2+1/2);%置信区间

fprintf('Mean estimate of pf with U function： %.4f\n',PostMeanPf1);
fprintf('Upper Credible bound of Pf using U function： %.5f\n', PostCIPf1(2));
fprintf('Lower Credible bound of Pf using U function： %.5f\n', PostCIPf1(1));
fprintf('Ncall by U function： %d\n', Ntrain1);

fprintf('Mean estimate of pf with EIER function： %.4f\n',PostMeanPf2);
fprintf('Upper Credible bound of Pf using EIER function： %.5f\n', PostCIPf2(2));
fprintf('Lower Credible bound of Pf using EIER function： %.5f\n', PostCIPf2(1));
fprintf('Ncall by EIER function： %d\n', Ntrain2);

%%%%for plotting the training details
Ngrid = 1000;
Xgrid = linspace(-4,4,Ngrid);
[Xmesh1,Xmesh2] = meshgrid(Xgrid,Xgrid);
for i=1:Ngrid
    Ymesh(:,i) = g([Xmesh1(:,i),Xmesh2(:,i)]);
end
figure
subplot(1,2,1)
a1 = pcolor(Xmesh1,Xmesh2,Ymesh);
set(a1,'edgecolor','none','facecolor','interp');
colorbar
hold on
contour(Xmesh1,Xmesh2,Ymesh,[0,0],'-r','LineWidth',2);
hold on
scatter(Xtrain1(:,1),Xtrain1(:,2),'filled')
title('(a). Training details with U-function')

subplot(1,2,2)
a2 = pcolor(Xmesh1,Xmesh2,Ymesh);
set(a2,'edgecolor','none','facecolor','interp');
colorbar
hold on
contour(Xmesh1,Xmesh2,Ymesh,[0,0],'-r','LineWidth',2);
hold on
scatter(Xtrain2(:,1),Xtrain2(:,2),'filled')
title('(b). Training details with EIER function')