function GPRSamp = GPRSampling(GPRmodel,Ind_Train,Xtrain,Ytrain,XSampPool,NGPR)
   Ntrain = length(Ytrain);
   Sigma0 = GPRmodel.KernelInformation.KernelParameters(end);
   Sigma1 = GPRmodel.KernelInformation.KernelParameters(1:end-1)';
   Beta0 = GPRmodel.Beta;
   kfcn = @(XN,XM) Sigma0^2*exp(-(pdist2(XN,XM,'seuclidean',Sigma1).^2)/2);%%kernel function
   K = kfcn(Xtrain,Xtrain);
   InvK=(linsolve(K,diag(ones(1,Ntrain))))';
   Meany = @(x)Beta0(1)+x*Beta0(2:end)+kfcn(x,Xtrain)*InvK*(Ytrain-[ones(Ntrain,1),Xtrain]*Beta0);

   %% define the covariance function
   corr.name = 'gauss';
   corr.c0 = Sigma1.^2;
   corr.sigma = Sigma0^2;

   %% Sample from the posterior GP using 
   [Ftrain,~] = randomfield(corr,XSampPool,'nsamples',NGPR,'filter', 0.95);
   GPRSamp = Meany(XSampPool) - kfcn(XSampPool,Xtrain)*InvK*Ftrain(Ind_Train,:)+Ftrain;
end