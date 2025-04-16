%% Pondini R., "Uncertainty and Business Cycles: New Insights Post-COVID-19", a contribution to  Angelini G., Bacchiocchi E., Caggiano G. and Fanelli L. (2018) "Uncertainty Across Volatility Regimes" 

%-----------------------------------------------------------------------
% Press the button change folder if it's the first time running the code
%-----------------------------------------------------------------------

clear
clc

addpath Final_work
addpath Functions
addpath Datasets

global NLags
global VAR_Variables_X
global VAR_Variables_Y
global T1
global T2
global T3
global T4
global Sigma_1Regime
global Sigma_2Regime
global Sigma_3Regime
global Sigma_4Regime
global Sigma_1Regime_Boot
global Sigma_2Regime_Boot
global Sigma_3Regime_Boot
global Sigma_4Regime_Boot
global StandardErrorSigma_1Regime
global StandardErrorSigma_2Regime
global StandardErrorSigma_3Regime
global StandardErrorSigma_4Regime
global CommonPI

%% General data
NLags = 4; % Number of lags of the reduced form VARs
options = optimset('MaxFunEvals',200000,'TolFun',1e-1000,'MaxIter',200000,'TolX',1e-1000);   
options2 = optimset('MaxFunEvals',200000,'TolFun',1e-200,'MaxIter',200000,'TolX',1e-200); 

LimitTEST = 1.64; %
LimitTEST_Apha = 0.1;

% Graph settings
LineWidth_IRF = 1.5;
LineWidth_IRF_Boot = 1;
FontSizeIRFGraph = 26;

% Data set
load DataSet.txt

% Break dates
TB1=284; 
TB2=569;
TB3=712;

%% Data Set
DataSet = DataSet(2:end,[1 3 2]);
AllDataSet=DataSet;
M=size(DataSet,2);

UM1=DataSet(:,1); % Macro Uncertainty variable
Y=DataSet(:,2); % Industrial Production variable
UF1=DataSet(:,3); % Financial Uncertainty variable

% Creates the data for the three regimes
DataSet_1Regime=DataSet(1:TB1,:); % First regime
DataSet_2Regime=DataSet(TB1+1-NLags:TB2,:); % Second regime
DataSet_3Regime=DataSet(TB2+1-NLags:TB3,:); % Third regime
DataSet_4Regime=DataSet(TB3+1-NLags:end,:); % Fourth regime

% Size three regimes
T1 = size(DataSet_1Regime,1)-NLags;
T2 = size(DataSet_2Regime,1)-NLags;
T3 = size(DataSet_3Regime,1)-NLags;
T4 = size(DataSet_4Regime,1)-NLags;
TAll = size(DataSet,1)-NLags;
numRegimes = (size(T1,1)+size(T2,1)+size(T3,1)+size(T4,1));

%% Reduced form estimation

T=TAll;
VAR_Variables_X=[ones(size(DataSet(NLags:end-1,:),1),1) DataSet(NLags:end-1,:) DataSet(NLags-1:end-2,:) DataSet(NLags-2:end-3,:) DataSet(NLags-3:end-4,:)];
VAR_Variables_Y=DataSet(NLags+1:end,:);

DuplicationMatrix = zeros(M^2,0.5*M*(M+1));
DuplicationMatrix(1,1)=1;
DuplicationMatrix(2,2)=1;
DuplicationMatrix(3,3)=1;
DuplicationMatrix(4,2)=1;
DuplicationMatrix(5,4)=1;
DuplicationMatrix(6,5)=1;
DuplicationMatrix(7,3)=1;
DuplicationMatrix(8,5)=1;
DuplicationMatrix(9,6)=1;
mDD=(DuplicationMatrix'*DuplicationMatrix)^(-1)*DuplicationMatrix';
mNN=DuplicationMatrix*mDD;

KommutationMatrix = zeros(M^2,M^2);
KommutationMatrix(1,1)=1;
KommutationMatrix(2,4)=1;
KommutationMatrix(3,7)=1;
KommutationMatrix(4,2)=1;
KommutationMatrix(5,5)=1;
KommutationMatrix(6,8)=1;
KommutationMatrix(7,3)=1;
KommutationMatrix(8,6)=1;
KommutationMatrix(9,9)=1;

NMatrix = 0.5*(eye(M^2)+KommutationMatrix);

Beta_OLS=(VAR_Variables_X'*VAR_Variables_X)^(-1)*VAR_Variables_X'*VAR_Variables_Y;

[Beta_LK,Log_LK,exitflag,~,grad,HESSIAN_LK] = fminunc('Likelihood_UNRESTRICTED',Beta_OLS',options2);

CommonPI=Beta_LK;
Errors=VAR_Variables_Y-VAR_Variables_X*Beta_LK';
Omega_LK=1/(T)*(Errors'*Errors);
LK_All=-Log_LK;

% Standard errors of the reduced form parameters (autoregressive parameters)
StandardErrors_BETA=reshape(sqrt(diag(kron(Omega_LK,(VAR_Variables_X'*VAR_Variables_X)^(-1)))),M*NLags+1,M);

% Standard errors of the reduced form parameters (covariance matrix)
StandardErrors_Omega=sqrt(diag(2/T*((mDD*kron(Omega_LK,Omega_LK)*(mDD)'))));
StandardErrors_Omega_M=[StandardErrors_Omega(1) StandardErrors_Omega(2) StandardErrors_Omega(3);
                        StandardErrors_Omega(2) StandardErrors_Omega(4) StandardErrors_Omega(5);
                        StandardErrors_Omega(3) StandardErrors_Omega(5) StandardErrors_Omega(6)];

Omega_All=[Omega_LK;StandardErrors_Omega_M];

% Preallocate the correlation matrix
correlationMatrix_All = zeros(size(Omega_LK));

% Calculate the correlation matrix
for i = 1:size(Omega_LK, 1)
    for j = 1:size(Omega_LK, 2)
        correlationMatrix_All(i, j) = Omega_LK(i, j) / sqrt(Omega_LK(i, i) * Omega_LK(j, j));
    end
end

% Call to Breusch-Godfrey test 
[pValueBG, LMStatBG] = BreuschGodfreyTest(Errors);


% ******************************************************************************
% First Regime
% ******************************************************************************
T=T1;
DataSet=DataSet_1Regime;
VAR_Variables_X=[ones(size(DataSet(NLags:end-1,:),1),1) DataSet(NLags:end-1,:) DataSet(NLags-1:end-2,:) DataSet(NLags-2:end-3,:) DataSet(NLags-3:end-4,:)];
VAR_Variables_Y=DataSet(NLags+1:end,:);

Beta_OLS=(VAR_Variables_X'*VAR_Variables_X)^(-1)*VAR_Variables_X'*VAR_Variables_Y;

[Beta_LK1,Log_LK,exitflag,~,grad,HESSIAN_LK] = fminunc('Likelihood_UNRESTRICTED',Beta_OLS',options2);

Errors1=VAR_Variables_Y-VAR_Variables_X*Beta_LK1';
Omega_LK1=1/(T)*(Errors1'*Errors1);
Errors_Pi_1=VAR_Variables_Y-VAR_Variables_X*Beta_LK';
Sigma_1Regime_Pi=1/(T)*(Errors_Pi_1'*Errors_Pi_1);

% Standard errors of the reduced form parameters (autoregressive parameters)
StandardErrors_BETA=reshape(sqrt(diag(kron(Omega_LK1,(VAR_Variables_X'*VAR_Variables_X)^(-1)))),M*NLags+1,M);
% Standard errors of the reduced form parameters (covariance matrix)
StandardErrors_Omega=sqrt(diag(2/T*((mDD*kron(Omega_LK1,Omega_LK1)*(mDD)'))));
StandardErrors_Omega_M=[StandardErrors_Omega(1) StandardErrors_Omega(2) StandardErrors_Omega(3);
                                StandardErrors_Omega(2) StandardErrors_Omega(4) StandardErrors_Omega(5);
                                StandardErrors_Omega(3) StandardErrors_Omega(5) StandardErrors_Omega(6)];

SE_Sigma_1Regime=StandardErrors_Omega;
StandardErrorSigma_1Regime=(2/T*((mDD*kron(Omega_LK1,Omega_LK1)*(mDD)')));

% Companion matrix and reduced form parameters
CompanionMatrix_1Regime=[Beta_LK1(:,2:end);
    eye(M*(NLags-1),M*(NLags-1)) zeros(M*(NLags-1),M)];                            
                            
Omega_1Regime=[Omega_LK1;StandardErrors_Omega_M];
Beta_1Regime=[Beta_LK1'; StandardErrors_BETA];

Sigma_1Regime=Omega_LK1;

% Calculate the correlation matrix
correlationMatrix_1Regime = zeros(size(Sigma_1Regime));
for i = 1:size(Sigma_1Regime, 1)
    for j = 1:size(Sigma_1Regime, 2)
        correlationMatrix_1Regime(i, j) = Sigma_1Regime(i, j) / sqrt(Sigma_1Regime(i, i) * Sigma_1Regime(j, j));
    end
end

% Likelihood of the VAR in the first regime
LK_1Regime=-Log_LK;
LK_Pi_1=(-0.5*T*M*(log(2*pi))-0.5*T*log(det(Sigma_1Regime_Pi))-0.5*trace((VAR_Variables_Y-VAR_Variables_X*Beta_LK')*(Sigma_1Regime_Pi)^(-1)*(VAR_Variables_Y-VAR_Variables_X*Beta_LK')'));

% Call to Breusch-Godfrey test 
[pValueBG_1regime, LMStatBG_1regime] = BreuschGodfreyTest(Errors1);


% ******************************************************************************
% Second Regime
% ******************************************************************************

T=T2;
DataSet=DataSet_2Regime;
VAR_Variables_X=[ones(size(DataSet(NLags:end-1,:),1),1) DataSet(NLags:end-1,:) DataSet(NLags-1:end-2,:) DataSet(NLags-2:end-3,:) DataSet(NLags-3:end-4,:)];
VAR_Variables_Y=DataSet(NLags+1:end,:);

Beta_OLS=(VAR_Variables_X'*VAR_Variables_X)^(-1)*VAR_Variables_X'*VAR_Variables_Y;

[Beta_LK2,Log_LK,exitflag,~,grad,HESSIAN_LK] = fminunc('Likelihood_UNRESTRICTED',Beta_OLS',options2);

Errors2=VAR_Variables_Y-VAR_Variables_X*Beta_LK2';
Errors_Pi_2=VAR_Variables_Y-VAR_Variables_X*Beta_LK';
Omega_LK2=1/(T)*(Errors2'*Errors2);
Sigma_2Regime_Pi=1/(T)*(Errors_Pi_2'*Errors_Pi_2);

% Standard errors of the reduced form parameters (autoregressive parameters)
StandardErrors_BETA=reshape(sqrt(diag(kron(Omega_LK2,(VAR_Variables_X'*VAR_Variables_X)^(-1)))),M*NLags+1,M);
% Standard errors of the reduced form parameters (Covariance matrix)
StandardErrors_Omega=sqrt(diag(2/T*((mDD*kron(Omega_LK2,Omega_LK2)*(mDD)'))));
StandardErrors_Omega_M=[StandardErrors_Omega(1) StandardErrors_Omega(2) StandardErrors_Omega(3);
                                StandardErrors_Omega(2) StandardErrors_Omega(4) StandardErrors_Omega(5);
                                StandardErrors_Omega(3) StandardErrors_Omega(5) StandardErrors_Omega(6)];

                            
SE_Sigma_2Regime=StandardErrors_Omega;
StandardErrorSigma_2Regime=(2/T*((mDD*kron(Omega_LK2,Omega_LK2)*(mDD)')));

% Companion matrix and reduced form parameters
CompanionMatrix_2Regime=[Beta_LK2(:,2:end);
    eye(M*(NLags-1),M*(NLags-1)) zeros(M*(NLags-1),M)];                            
                             
Omega_2Regime=[Omega_LK2;StandardErrors_Omega_M];
Beta_2Regime=[Beta_LK2'; StandardErrors_BETA];

Sigma_2Regime=Omega_LK2;

% Calculate the correlation matrix
correlationMatrix_2Regime = zeros(size(Sigma_2Regime));
for i = 1:size(Sigma_2Regime, 1)
    for j = 1:size(Sigma_2Regime, 2)
        correlationMatrix_2Regime(i, j) = Sigma_2Regime(i, j) / sqrt(Sigma_2Regime(i, i) * Sigma_2Regime(j, j));
    end
end

% Likelihood of the VAR in the second regime
LK_2Regime=-Log_LK;
LK_Pi_2=(-0.5*T*M*(log(2*pi))-0.5*T*log(det(Sigma_2Regime_Pi))-0.5*trace((VAR_Variables_Y-VAR_Variables_X*Beta_LK')*(Sigma_2Regime_Pi)^(-1)*(VAR_Variables_Y-VAR_Variables_X*Beta_LK')'));

% Call to Breusch-Godfrey test 
[pValueBG_2regime, LMStatBG_2regime] = BreuschGodfreyTest(Errors2);

          
% ******************************************************************************
% Third Regime
% ******************************************************************************

T=T3;
DataSet=DataSet_3Regime;
VAR_Variables_X=[ones(size(DataSet(NLags:end-1,:),1),1) DataSet(NLags:end-1,:) DataSet(NLags-1:end-2,:) DataSet(NLags-2:end-3,:) DataSet(NLags-3:end-4,:)];
VAR_Variables_Y=DataSet(NLags+1:end,:);

Beta_OLS=(VAR_Variables_X'*VAR_Variables_X)^(-1)*VAR_Variables_X'*VAR_Variables_Y;

[Beta_LK3,Log_LK,exitflag,~,grad,HESSIAN_LK] = fminunc('Likelihood_UNRESTRICTED',Beta_OLS',options2);

Errors3=VAR_Variables_Y-VAR_Variables_X*Beta_LK3';
Errors_Pi_3=VAR_Variables_Y-VAR_Variables_X*Beta_LK';
Omega_LK3=1/(T)*(Errors3'*Errors3);
Sigma_3Regime_Pi=1/(T)*(Errors_Pi_3'*Errors_Pi_3);

% Standard errors of the reduced form parameters (autoregressive parameters)
StandardErrors_BETA=reshape(sqrt(diag(kron(Omega_LK3,(VAR_Variables_X'*VAR_Variables_X)^(-1)))),M*NLags+1,M);

% Standard errors of the reduced form parameters (covariance matrix)
StandardErrors_Omega=sqrt(diag(2/T*((mDD*kron(Omega_LK3,Omega_LK3)*(mDD)'))));
StandardErrors_Omega_M=[StandardErrors_Omega(1) StandardErrors_Omega(2) StandardErrors_Omega(3);
                                StandardErrors_Omega(2) StandardErrors_Omega(4) StandardErrors_Omega(5);
                                StandardErrors_Omega(3) StandardErrors_Omega(5) StandardErrors_Omega(6)];

SE_Sigma_3Regime=StandardErrors_Omega;
StandardErrorSigma_3Regime=(2/T*((mDD*kron(Omega_LK3,Omega_LK3)*(mDD)')));

% Companion matrix and reduced form parameters
CompanionMatrix_3Regime=[Beta_LK3(:,2:end);
    eye(M*(NLags-1),M*(NLags-1)) zeros(M*(NLags-1),M)];                            
                                       
Omega_3Regime=[Omega_LK3;StandardErrors_Omega_M];
Beta_3Regime=[Beta_LK3'; StandardErrors_BETA];

Sigma_3Regime=Omega_LK3;

% Calculate the correlation matrix
correlationMatrix_3Regime = zeros(size(Sigma_3Regime));
for i = 1:size(Sigma_3Regime, 1)
    for j = 1:size(Sigma_3Regime, 2)
        correlationMatrix_3Regime(i, j) = Sigma_3Regime(i, j) / sqrt(Sigma_3Regime(i, i) * Sigma_3Regime(j, j));
    end
end

% Likelihood of the VAR in the third regime  
LK_3Regime=-Log_LK;
LK_Pi_3=(-0.5*T*M*(log(2*pi))-0.5*T*log(det(Sigma_3Regime_Pi))-0.5*trace((VAR_Variables_Y-VAR_Variables_X*Beta_LK')*(Sigma_3Regime_Pi)^(-1)*(VAR_Variables_Y-VAR_Variables_X*Beta_LK')'));

% Call to Breusch-Godfrey test
[pValueBG_3regime, LMStatBG_3regime] = BreuschGodfreyTest(Errors3);


% ******************************************************************************
% Fourth Regime
% ******************************************************************************

T=T4;
DataSet=DataSet_4Regime;
VAR_Variables_X=[ones(size(DataSet(NLags:end-1,:),1),1) DataSet(NLags:end-1,:) DataSet(NLags-1:end-2,:) DataSet(NLags-2:end-3,:) DataSet(NLags-3:end-4,:)];
VAR_Variables_Y=DataSet(NLags+1:end,:);

Beta_OLS=(VAR_Variables_X'*VAR_Variables_X)^(-1)*VAR_Variables_X'*VAR_Variables_Y;

[Beta_LK4,Log_LK,exitflag,~,grad,HESSIAN_LK] = fminunc('Likelihood_UNRESTRICTED',Beta_OLS',options2);

Errors4=VAR_Variables_Y-VAR_Variables_X*Beta_LK4';
Errors_Pi_4=VAR_Variables_Y-VAR_Variables_X*Beta_LK';
Omega_LK4=1/(T)*(Errors4'*Errors4);
Sigma_4Regime_Pi=1/(T)*(Errors_Pi_4'*Errors_Pi_4);

% Standard errors of the reduced form parameters (autoregressive parameters)
StandardErrors_BETA=reshape(sqrt(diag(kron(Omega_LK4,(VAR_Variables_X'*VAR_Variables_X)^(-1)))),M*NLags+1,M);

% Standard errors of the reduced form parameters (covariance matrix)
StandardErrors_Omega=sqrt(diag(2/T*((mDD*kron(Omega_LK4,Omega_LK4)*(mDD)'))));
StandardErrors_Omega_M=[StandardErrors_Omega(1) StandardErrors_Omega(2) StandardErrors_Omega(3);
                                StandardErrors_Omega(2) StandardErrors_Omega(4) StandardErrors_Omega(5);
                                StandardErrors_Omega(3) StandardErrors_Omega(5) StandardErrors_Omega(6)];

SE_Sigma_4Regime=StandardErrors_Omega;
StandardErrorSigma_4Regime=(2/T*((mDD*kron(Omega_LK4,Omega_LK4)*(mDD)')));

% Companion matrix and reduced form parameters
CompanionMatrix_4Regime=[Beta_LK4(:,2:end);
    eye(M*(NLags-1),M*(NLags-1)) zeros(M*(NLags-1),M)];                            
                                       
Omega_4Regime=[Omega_LK4;StandardErrors_Omega_M];
Beta_4Regime=[Beta_LK4'; StandardErrors_BETA];

Sigma_4Regime=Omega_LK4;

% Calculate the correlation matrix
correlationMatrix_4Regime = zeros(size(Sigma_4Regime));
for i = 1:size(Sigma_4Regime, 1)
    for j = 1:size(Sigma_4Regime, 2)
        correlationMatrix_4Regime(i, j) = Sigma_4Regime(i, j) / sqrt(Sigma_4Regime(i, i) * Sigma_4Regime(j, j));
    end
end

% Likelihood of the VAR in the fourth regime  
LK_4Regime=-Log_LK;
LK_Pi_4=(-0.5*T*M*(log(2*pi))-0.5*T*log(det(Sigma_4Regime_Pi))-0.5*trace((VAR_Variables_Y-VAR_Variables_X*Beta_LK')*(Sigma_4Regime_Pi)^(-1)*(VAR_Variables_Y-VAR_Variables_X*Beta_LK')'));
% Call to Breusch-Godfrey test 
[pValueBG_4regime, LMStatBG_4regime] = BreuschGodfreyTest(Errors4);


%% Chow type tests for Pi and Sigma_u variability across regimes

LR_Pi_sigma = 2 * ((LK_1Regime(1)+LK_2Regime(1)+LK_3Regime(1)+LK_4Regime(1))-LK_All(1));
df_Pi_sigma = (numRegimes-1)*((M^2)*NLags + M + (M*(M+1))/2);
PVal_LRTest_Pi_sigma = 1-chi2cdf(LR_Pi_sigma, df_Pi_sigma);

%% Chow type tests for Sigma_u variability across regimes keeping Pi fixed

%{
LK_Pi_1=(-0.5*T1*M*(log(2*pi)+1)-0.5*T1*log(det(Sigma_1Regime_Pi)));
LK_Pi_2=(-0.5*T2*M*(log(2*pi)+1)-0.5*T2*log(det(Sigma_2Regime_Pi)));
LK_Pi_3=(-0.5*T3*M*(log(2*pi)+1)-0.5*T3*log(det(Sigma_3Regime_Pi)));
LK_Pi_4=(-0.5*T4*M*(log(2*pi)+1)-0.5*T4*log(det(Sigma_4Regime_Pi)));
%}

LR_sigma = 2 * ((LK_Pi_1+LK_Pi_2+LK_Pi_3+LK_Pi_4)-LK_All(1));
df_sigma = (numRegimes-1)*((M*(M+1))/2);
PVal_LRTest_sigma = 1-chi2cdf(LR_sigma, df_sigma);


%% Rolling/Recursive window estimation of the Covariance/Variance matrix of the Var innovations

%% Rolling Window Estimation (10 years)
windowSize = 120; % Window size in months 
nData = size(AllDataSet, 1); % Total number of observations

% Preallocation of matrices for results
rollingBetas = [];
rollingOmega = [];

% Initializing the progress bar for the rolling window calculation
wbGS = waitbar(0,'Running the 10 years rolling window'); 
for start = 1:(nData - windowSize + 1)
    waitbar(start / (nData - windowSize + 1), wbGS, sprintf('Running the 10 years rolling window: Processing %d of %d', start, nData - windowSize + 1));
    endIdx = start + windowSize - 1;
    
    % Extract data for the current window
    windowData = AllDataSet(start:endIdx, :);
    
    % Construct VAR variables for the window
    VAR_Variables_X = [ones(size(windowData(NLags:end-1, :), 1), 1) windowData(NLags:end-1, :) windowData(NLags-1:end-2, :) windowData(NLags-2:end-3, :) windowData(NLags-3:end-4, :)];
    VAR_Variables_Y = windowData(NLags+1:end, :);
    
    % OLS Estimation
    Beta_OLS_Rol = (VAR_Variables_X'*VAR_Variables_X)^(-1)*VAR_Variables_X'*VAR_Variables_Y;

    % Likelihood Estimation
    [Beta_LK_Rol, Log_LK_Rol, ~, ~, HESSIAN_LK_Rol] = fminunc('Likelihood_UNRESTRICTED', Beta_OLS_Rol', optimset('MaxFunEvals', 200000, 'TolFun', 1e-200, 'MaxIter', 200000, 'TolX', 1e-200));
    Errors = VAR_Variables_Y - VAR_Variables_X * Beta_LK_Rol';
    Omega = 1 / (size(windowData, 1) - NLags) * (Errors' * Errors);
    
    % Save the results
    rollingBetas = cat(3, rollingBetas, Beta_OLS_Rol);
    rollingOmega = cat(3, rollingOmega, Omega);
end
delete(wbGS); 


%% Rolling Window Estimation (15 years)
windowSize = 180; % Window size in months
nData = size(AllDataSet, 1); % Total number of observations

% Preallocation of matrices for results
rolling2Betas = [];
rolling2Omega = [];

% Initializing the progress bar for the 15-year rolling window calculation
wbGS = waitbar(0,'Running the 15-year rolling window'); 
for start = 1:(nData - windowSize + 1)
    waitbar(start / (nData - windowSize + 1), wbGS, sprintf('Running the 15 years rolling window: Processing %d of %d', start, nData - windowSize + 1));
    endIdx = start + windowSize - 1;
    
    % Extract data for the current window
    windowData = AllDataSet(start:endIdx, :);
    
    % Construct VAR variables for the window
    VAR_Variables_X = [ones(size(windowData(NLags:end-1, :), 1), 1), windowData(NLags:end-1, :), windowData(NLags-1:end-2, :), windowData(NLags-2:end-3, :), windowData(NLags-3:end-4, :)];
    VAR_Variables_Y = windowData(NLags+1:end, :);
    
    % OLS Estimation
    Beta_OLS_Rol2 = (VAR_Variables_X' * VAR_Variables_X)^(-1) * VAR_Variables_X' * VAR_Variables_Y;

    % Likelihood Estimation
    [Beta_LK_Rol2, Log_LK_Rol2, ~, ~, HESSIAN_LK_Rol2] = fminunc('Likelihood_UNRESTRICTED', Beta_OLS_Rol2', optimset('MaxFunEvals', 200000, 'TolFun', 1e-200, 'MaxIter', 200000, 'TolX', 1e-200));
    Errors = VAR_Variables_Y - VAR_Variables_X * Beta_LK_Rol2';
    Omega2 = 1 / (size(windowData, 1) - NLags) * (Errors' * Errors);
    
    % Save the results
    rolling2Betas = cat(3, rolling2Betas, Beta_OLS_Rol2);
    rolling2Omega = cat(3, rolling2Omega, Omega2);
end
delete(wbGS); 

%% Recursive Estimation
startSize = 120; % Initial size of the data window in months

% Preallocation
recursiveBetas = [];
recursiveOmega = [];

% Initialize the progress bar for the recursive window estimation
wbGS = waitbar(0, 'Running the recursive window');
i = 0;
for endIdx = startSize:nData
    i = i + 1;
    waitbar(i / (nData - startSize + 1), wbGS, sprintf('Running the recursive window: Processing %d of %d', i, nData - startSize + 1));
    
    % Data from the beginning up to the current index
    currentData = AllDataSet(1:endIdx, :);
    
    % Construct VAR variables
    VAR_Variables_X = [ones(size(currentData(NLags:end-1, :), 1), 1), currentData(NLags:end-1, :), currentData(NLags-1:end-2, :), currentData(NLags-2:end-3, :), currentData(NLags-3:end-4, :)];
    VAR_Variables_Y = currentData(NLags+1:end, :);
    
    % OLS Estimation
    Beta_OLS_Rec = (VAR_Variables_X' * VAR_Variables_X)^(-1) * VAR_Variables_X' * VAR_Variables_Y;

    % Likelihood Estimation
    [Beta_LK_Rec, Log_LK_Rec, ~, ~, HESSIAN_LK_Rec] = fminunc('Likelihood_UNRESTRICTED', Beta_OLS_Rec', optimset('MaxFunEvals', 200000, 'TolFun', 1e-200, 'MaxIter', 200000, 'TolX', 1e-200));
    Errors = VAR_Variables_Y - VAR_Variables_X * Beta_LK_Rec';
    Omega = 1 / (endIdx - NLags) * (Errors' * Errors);
    
    % Save the results
    recursiveBetas = cat(3, recursiveBetas, Beta_OLS_Rec);
    recursiveOmega = cat(3, recursiveOmega, Omega);
end
delete(wbGS); 


%% Plotting
% Define variable names and their associated colors
variableNames = {'u_{Mt}', 'u_{Yt}', 'u_{Ft}'};
variableColors = {[0, 0, 1], [1, 0.5, 0], [1, 0, 0]};  % Colors: Blue, Orange, Red

indices = [1, 165, 450, 593];
labels = {'07/1970', '03/1984', '12/2007', '11/2019'};

figure(1);
hold on; 
for i = 1:3
    for j = 1:3
        if i <= j  % Only plot the diagonal and above
            subplot(3, 3, (i-1)*3 + j);

            % Plotting each estimate with specific colors
            plot(squeeze(rollingOmega(i, j, :)), 'Color', [1, 0, 0], 'LineWidth', 1.3);
            hold on;
            plot(squeeze(recursiveOmega(i, j, :)), 'Color', [0, 0, 1], 'LineWidth', 1.3);
            hold on;
            plot((1:size(rolling2Omega, 3)) + 59, squeeze(rolling2Omega(i, j, :)), 'Color', [1, 0.5, 0], 'LineWidth', 1.3); 

            % Add vertical dashed lines at specific indices
            xline(165, '--', 'Color', 'k', 'LineWidth', 0.5);
            xline(450, '--', 'Color', 'k', 'LineWidth', 0.5);
            xline(593, '--', 'Color', 'k', 'LineWidth', 0.5);

            if i == j
                title(sprintf('Var(%s)', variableNames{i}), 'FontSize', 14);
            else
                title(sprintf('Cov(%s, %s)', variableNames{i}, variableNames{j}), 'FontSize', 14);
            end

            %xlabel('Time (Years)');
            %ylabel('Covariance');
            set(gca, 'FontSize', 16)
            set(gcf, 'Color', 'w'); 
            set(gca, 'Color', 'w');

            xlim([1 max((1:size(rolling2Omega, 3)) + 59)]);
            xticks([indices, indices(end)+60]);
            xticklabels([labels, '03/2035']);
        else
            subplot(3, 3, (i-1)*3 + j);
            set(gca, 'Visible', 'off');  
        end
    end
end
hold off;



%% --------------- Estimation of the structural parameters (main specification in the third row of Table 2) ------------

StructuralParam=19; 
InitialValue_SVAR_Initial=0.5*ones(StructuralParam,1);

% ML function
[StructuralParam_Estiamtion_MATRIX,Likelihood_MATRIX,exitflag,output,grad,Hessian_MATRIX] = fminunc('Likelihood_SVAR_Restricted',InitialValue_SVAR_Initial',options);

StructuralParam_Estiamtion=StructuralParam_Estiamtion_MATRIX;
LK_Estimation=Likelihood_MATRIX;
Hessian_Estimation=Hessian_MATRIX;
SE_Estimation=diag(Hessian_Estimation^(-1)).^0.5;

%% Overidentification LR test
PVarl_LRTest = 1-chi2cdf(2 * ((LK_1Regime(1)+LK_2Regime(1)+LK_3Regime(1)+LK_4Regime(1))+LK_Estimation),24-StructuralParam);

%% Here below we define the matrices of the structural parameters with restrictions on the coefficients as described in the paper. SVAR_C corresponds to the B matrix in the paper, SVAR_Q2 corresponds to
% the Q2 matrix of the paper, SVAR_Q3 corresponds to the Q3 matrix of the paper and SVAR_Q4 corresponds to the Q4 matrix of the paper.

SVAR_C=[StructuralParam_Estiamtion(1) 0                             0;
        StructuralParam_Estiamtion(2) StructuralParam_Estiamtion(3) 0;
        0                             0                             StructuralParam_Estiamtion(4)];

SVAR_Q2=[StructuralParam_Estiamtion(5)  0                             StructuralParam_Estiamtion(8);
         StructuralParam_Estiamtion(6)  StructuralParam_Estiamtion(7) 0;
         0                              0                             StructuralParam_Estiamtion(9)];

SVAR_Q3=[0                              0                              StructuralParam_Estiamtion(12);
         StructuralParam_Estiamtion(10) StructuralParam_Estiamtion(11) StructuralParam_Estiamtion(13);
         0                              0                              StructuralParam_Estiamtion(14)];

SVAR_Q4=[StructuralParam_Estiamtion(15) 0                              StructuralParam_Estiamtion(17);
         0                              StructuralParam_Estiamtion(16) StructuralParam_Estiamtion(18);
         0                              0                              StructuralParam_Estiamtion(19)];  

SVAR_1Regime=SVAR_C; % B
SVAR_2Regime=SVAR_C+SVAR_Q2;   % B+Q2
SVAR_3Regime=SVAR_C+SVAR_Q2+SVAR_Q3;  % B+Q2+Q3
SVAR_4Regime=SVAR_C+SVAR_Q2+SVAR_Q3+SVAR_Q4;  % B+Q2+Q3+Q4

% Flip the sign if the parameter on the main diagonal is negative

	if SVAR_1Regime(1,1)<0
    SVAR_1Regime(:,1)=-SVAR_1Regime(:,1);
    end
    if SVAR_1Regime(2,2)<0
    SVAR_1Regime(:,2)=-SVAR_1Regime(:,2); 
    end
    if SVAR_1Regime(3,3)<0
    SVAR_1Regime(:,3)=-SVAR_1Regime(:,3);
    end
    
	if SVAR_2Regime(1,1)<0
    SVAR_2Regime(:,1)=-SVAR_2Regime(:,1);
    end
    if SVAR_2Regime(2,2)<0
    SVAR_2Regime(:,2)=-SVAR_2Regime(:,2); 
    end
    if SVAR_2Regime(3,3)<0
    SVAR_2Regime(:,3)=-SVAR_2Regime(:,3);
    end
    
    
    if SVAR_3Regime(1,1)<0
    SVAR_3Regime(:,1)=-SVAR_3Regime(:,1);
    end
    if SVAR_3Regime(2,2)<0
    SVAR_3Regime(:,2)=-SVAR_3Regime(:,2); 
    end
    if SVAR_3Regime(3,3)<0
    SVAR_3Regime(:,3)=-SVAR_3Regime(:,3);
    end
    

    if SVAR_4Regime(1,1)<0
    SVAR_4Regime(:,1)=-SVAR_4Regime(:,1);
    end
    if SVAR_4Regime(2,2)<0
    SVAR_4Regime(:,2)=-SVAR_4Regime(:,2); 
    end
    if SVAR_4Regime(3,3)<0
    SVAR_4Regime(:,3)=-SVAR_4Regime(:,3);
    end
    
     
MATRICES=[SVAR_1Regime;
          SVAR_2Regime;
          SVAR_3Regime;
          SVAR_4Regime];
   
% Calculates the analytical derivatives organized in block matrices      
V11=2*NMatrix*kron(SVAR_C,eye(M));
V21=2*NMatrix*kron(SVAR_C,eye(M))+kron(SVAR_Q2,eye(M))+kron(eye(M),SVAR_Q2)*KommutationMatrix;
V22=kron(eye(M),SVAR_C)*KommutationMatrix+kron(SVAR_C,eye(M))+2*NMatrix*kron(SVAR_Q2,eye(M));
V31=2*NMatrix*kron(SVAR_C,eye(M))+kron(SVAR_Q2,eye(M))+kron(SVAR_Q3,eye(M))+kron(eye(M),SVAR_Q2)*KommutationMatrix+kron(eye(M),SVAR_Q3)*KommutationMatrix;
V32=kron(eye(M),SVAR_C)*KommutationMatrix+kron(SVAR_C,eye(M))+2*NMatrix*kron(SVAR_Q2,eye(M))+kron(SVAR_Q3,eye(M))+kron(eye(M),SVAR_Q3)*KommutationMatrix;
V33=kron(eye(M),SVAR_C)*KommutationMatrix+kron(eye(M),SVAR_Q2)*KommutationMatrix+2*NMatrix*kron(SVAR_Q3,eye(M))+kron(SVAR_C,eye(M))+kron(SVAR_Q2,eye(M));
V41 = 2 * NMatrix * kron(SVAR_C, eye(M)) + kron(SVAR_Q2, eye(M)) + kron(SVAR_Q3, eye(M)) + kron(SVAR_Q4, eye(M)) + kron(eye(M), SVAR_Q2) * KommutationMatrix + kron(eye(M), SVAR_Q3) * KommutationMatrix + kron(eye(M), SVAR_Q4) * KommutationMatrix;
V42 = kron(eye(M), SVAR_C) * KommutationMatrix + kron(SVAR_C, eye(M)) + 2 * NMatrix * kron(SVAR_Q2, eye(M)) + kron(SVAR_Q3, eye(M)) + kron(SVAR_Q4, eye(M)) + kron(eye(M), SVAR_Q3) * KommutationMatrix + kron(eye(M), SVAR_Q4) * KommutationMatrix;
V43 = kron(eye(M), SVAR_C) * KommutationMatrix + kron(eye(M), SVAR_Q2) * KommutationMatrix + 2 * NMatrix * kron(SVAR_Q3, eye(M)) + kron(SVAR_C, eye(M)) + kron(SVAR_Q2, eye(M)) + kron(SVAR_Q4, eye(M)) + kron(eye(M), SVAR_Q4) * KommutationMatrix;
V44 = kron(eye(M), SVAR_C) * KommutationMatrix + kron(eye(M), SVAR_Q2) * KommutationMatrix + kron(eye(M), SVAR_Q3) * KommutationMatrix + 2 * NMatrix * kron(SVAR_Q4, eye(M)) + kron(SVAR_C, eye(M)) + kron(SVAR_Q2, eye(M)) + kron(SVAR_Q3, eye(M));

% Calculates the matrix for checking the rank condition (full column rank)
RankMatrix = kron(eye(4), mDD) * [V11 zeros(M^2, M^2) zeros(M^2, M^2) zeros(M^2, M^2);
                                  V21 V22 zeros(M^2, M^2) zeros(M^2, M^2);
                                  V31 V32 V33 zeros(M^2, M^2);
                                  V41 V42 V43 V44];

 
% Selection matrix for extracting the structural parameters                         
HSelection=zeros(M*M*4,StructuralParam);
HSelection(1,1)=1;
HSelection(2,2)=1;
HSelection(5,3)=1;
HSelection(9,4)=1;
HSelection(10,5)=1;
HSelection(11,6)=1;
HSelection(14,7)=1;
HSelection(16,8)=1;
HSelection(18,9)=1;
HSelection(20,10)=1;
HSelection(23,11)=1;
HSelection(25,12)=1;
HSelection(26,13)=1;
HSelection(27,14)=1;
HSelection(28, 15) = 1;
HSelection(32, 16) = 1;  
HSelection(34, 17) = 1;  
HSelection(35, 18) = 1;  
HSelection(36, 19) = 1;  


Jacobian= RankMatrix*HSelection;

% Report the rank of the matrix for checking the identification
Jacobian_rank = rank(Jacobian);

MSigma=size(StandardErrorSigma_1Regime,1);

TetaMatrix = [
    StandardErrorSigma_1Regime, zeros(MSigma, MSigma),      zeros(MSigma, MSigma),      zeros(MSigma, MSigma);
    zeros(MSigma, MSigma),      StandardErrorSigma_2Regime, zeros(MSigma, MSigma),      zeros(MSigma, MSigma);
    zeros(MSigma, MSigma),      zeros(MSigma, MSigma),      StandardErrorSigma_3Regime, zeros(MSigma, MSigma);
    zeros(MSigma, MSigma),      zeros(MSigma, MSigma),      zeros(MSigma, MSigma),      StandardErrorSigma_4Regime
];

     
% % Calculates the variance and the standard errors of the estimated coefficients        
VarTeta=  (Jacobian'* TetaMatrix^(-1)*Jacobian)^(-1);
SETetaJacobian= diag(VarTeta).^0.5;

%% Structural parameters
StructuralEstimationCorrected=[
        MATRICES(1,1);
        MATRICES(2,1);
        MATRICES(2,2);
        MATRICES(3,3);

        MATRICES(4,1)-MATRICES(1,1);
        MATRICES(5,1)-MATRICES(2,1);
        MATRICES(5,2)-MATRICES(2,2);
        MATRICES(4,3);
        MATRICES(6,3)-MATRICES(3,3);
        
        MATRICES(8,1)-MATRICES(5,1); 
        MATRICES(8,2)-MATRICES(5,2);
        MATRICES(7,3)-MATRICES(4,3);
        MATRICES(8,3);
        MATRICES(9,3)-MATRICES(6,3);
        
        MATRICES(10,1) - MATRICES(7,1);
        MATRICES(11,2) - MATRICES(8,2);
        MATRICES(10,3)- MATRICES(7,3);
        MATRICES(11,3)- MATRICES(8,3);
        MATRICES(12,3) - MATRICES(9,3);
        ];
 
OUTPUT_Table2_StructuralEstimation=[StructuralEstimationCorrected SE_Estimation SETetaJacobian];

%% Estimation of the standard errors of the parameters in B, B+Q2, B+Q2+Q3, and B+Q2+Q3+Q4 (delta method)

% Inverse of the Hessian estimation for variance estimation
VAR_Est = Hessian_Estimation^(-1);

% Define symbolic variables for delta method calculations
syms x y z w
gradient_sigma_Matrix = [];

%% Calculation for B+Q2 combination
% teta(5)
index = 1;
i = 1; j = 5;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
f = x + y;
gradient_sigma = gradient(f, [x, y]);
gradient_sigma_est = subs(gradient_sigma, [x y], [first_par second_par]);
gradient_sigma_Matrix = double(gradient_sigma_est)';
SETetaDelta(index, :) = sqrt(diag(gradient_sigma_Matrix * VAR_Est([i j], [i j]) * gradient_sigma_Matrix'));

% teta(6)
index = 2;
i = 2; j = 6;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
f = x + y;
gradient_sigma = gradient(f, [x, y]);
gradient_sigma_est = subs(gradient_sigma, [x y], [first_par second_par]);
gradient_sigma_Matrix = double(gradient_sigma_est)';
SETetaDelta(index, :) = sqrt(diag(gradient_sigma_Matrix * VAR_Est([i j], [i j]) * gradient_sigma_Matrix'));

% teta(7)
index = 3;
i = 3; j = 7;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
f = x + y;
gradient_sigma = gradient(f, [x, y]);
gradient_sigma_est = subs(gradient_sigma, [x y], [first_par second_par]);
gradient_sigma_Matrix = double(gradient_sigma_est)';
SETetaDelta(index, :) = sqrt(diag(gradient_sigma_Matrix * VAR_Est([i j], [i j]) * gradient_sigma_Matrix'));

% teta(9)
index = 4;
i = 4; j = 9;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
f = x + y;
gradient_sigma = gradient(f, [x, y]);
gradient_sigma_est = subs(gradient_sigma, [x y], [first_par second_par]);
gradient_sigma_Matrix = double(gradient_sigma_est)';
SETetaDelta(index, :) = sqrt(diag(gradient_sigma_Matrix * VAR_Est([i j], [i j]) * gradient_sigma_Matrix'));

%% Calculation for B+Q2+Q3 combination

% teta(10)
index = 5;
i = 2; j = 6; k = 10;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
third_par = OUTPUT_Table2_StructuralEstimation(k,1);
f = x + y + z;
gradient_sigma = gradient(f, [x, y, z]);
gradient_sigma_est = subs(gradient_sigma, [x y z], [first_par second_par third_par]);
gradient_sigma_Matrix = double(gradient_sigma_est)';
SETetaDelta(index, :) = sqrt(diag(gradient_sigma_Matrix * VAR_Est([i j k], [i j k]) * gradient_sigma_Matrix'));

% teta(11)
index = 6;
i = 3; j = 7; k = 11;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
third_par = OUTPUT_Table2_StructuralEstimation(k,1);
f = x + y + z;
gradient_sigma = gradient(f, [x, y, z]);
gradient_sigma_est = subs(gradient_sigma, [x y z], [first_par second_par third_par]);
gradient_sigma_Matrix = double(gradient_sigma_est)';
SETetaDelta(index, :) = sqrt(diag(gradient_sigma_Matrix * VAR_Est([i j k], [i j k]) * gradient_sigma_Matrix'));

% teta(12)
index = 7;
j = 8; k = 12;
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
third_par = OUTPUT_Table2_StructuralEstimation(k,1);
f = y + z;
gradient_sigma = gradient(f, [y, z]);
gradient_sigma_est = subs(gradient_sigma, [y z], [second_par third_par]);
gradient_sigma_Matrix = double(gradient_sigma_est)';
SETetaDelta(index, :) = sqrt(diag(gradient_sigma_Matrix * VAR_Est([j k], [j k]) * gradient_sigma_Matrix'));

% teta(14)
index = 8;
i = 4; j = 9; l = 14;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
fourth_par = OUTPUT_Table2_StructuralEstimation(l,1);
f = x + y + w;
gradient_sigma = gradient(f, [x, y, w]);
gradient_sigma_est = subs(gradient_sigma, [x y w], [first_par second_par fourth_par]);
gradient_sigma_Matrix = double(gradient_sigma_est)';
SETetaDelta(index, :) = sqrt(diag(gradient_sigma_Matrix * VAR_Est([i j l], [i j l]) * gradient_sigma_Matrix'));


%% Calculation for B+Q2+Q3+Q4 combination

% teta(15)
index = 9;
i = 1; j = 5; l = 15;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
fourth_par = OUTPUT_Table2_StructuralEstimation(l,1);
f = x + y + w;
gradient_sigma = gradient(f, [x, y, w]);
gradient_sigma_est = subs(gradient_sigma, [x y w], [first_par second_par fourth_par]);
gradient_sigma_Matrix = double(gradient_sigma_est)';
SETetaDelta(index, :) = sqrt(diag(gradient_sigma_Matrix * VAR_Est([i j l], [i j l]) * gradient_sigma_Matrix'));

% teta(16)
index = 10;
i = 3; j = 7; k = 11; l = 16;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
third_par = OUTPUT_Table2_StructuralEstimation(k,1);
fourth_par = OUTPUT_Table2_StructuralEstimation(l,1);
f = x + y + z + w;
gradient_sigma = gradient(f, [x, y, z, w]);
gradient_sigma_est = subs(gradient_sigma, [x y z w], [first_par second_par third_par fourth_par]);
gradient_sigma_Matrix = double(gradient_sigma_est)';
SETetaDelta(index, :) = sqrt(diag(gradient_sigma_Matrix * VAR_Est([i j k l], [i j k l]) * gradient_sigma_Matrix'));


% teta(17)
index = 11;
j = 8; k = 12; l = 17;
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
third_par = OUTPUT_Table2_StructuralEstimation(k,1);
fourth_par = OUTPUT_Table2_StructuralEstimation(l,1);
f = y + z + w;
gradient_sigma = gradient(f, [y, z, w]);
gradient_sigma_est = subs(gradient_sigma, [y, z w], [second_par third_par fourth_par]);
gradient_sigma_Matrix = double(gradient_sigma_est)';
SETetaDelta(index, :) = sqrt(diag(gradient_sigma_Matrix * VAR_Est([j k l], [j k l]) * gradient_sigma_Matrix'));

% teta(18)
index = 12;
k = 13; l = 18;
third_par = OUTPUT_Table2_StructuralEstimation(k,1);
fourth_par = OUTPUT_Table2_StructuralEstimation(l,1);
f = z + w;
gradient_sigma = gradient(f, [z, w]);
gradient_sigma_est = subs(gradient_sigma, [z w], [third_par fourth_par]);
gradient_sigma_Matrix = double(gradient_sigma_est)';
SETetaDelta(index, :) = sqrt(diag(gradient_sigma_Matrix * VAR_Est([k l], [k l]) * gradient_sigma_Matrix'));

% teta(19)
index = 13;
i = 4; j = 9; k = 14; l = 19;
first_par = OUTPUT_Table2_StructuralEstimation(i,1);
second_par = OUTPUT_Table2_StructuralEstimation(j,1);
third_par = OUTPUT_Table2_StructuralEstimation(k,1);
fourth_par = OUTPUT_Table2_StructuralEstimation(l,1);
f = x + y + z + w;
gradient_sigma = gradient(f, [x, y, z, w]);
gradient_sigma_est = subs(gradient_sigma, [x y z w], [first_par second_par third_par fourth_par]);
gradient_sigma_Matrix = double(gradient_sigma_est)';
SETetaDelta(index, :) = sqrt(diag(gradient_sigma_Matrix * VAR_Est([i j k l], [i j k l]) * gradient_sigma_Matrix'));

%% Concatenate all standard errors into one matrix for output
SE_ANALYTIC = [SE_Estimation; SETetaDelta];

% Organizing standard errors into a structured output matrix
OUTPUT_Table2_SE_Analytic = [
    SE_ANALYTIC(1)  0               0;
    SE_ANALYTIC(2)  SE_ANALYTIC(3)  0;  
    0               0               SE_ANALYTIC(4);               
    SE_ANALYTIC(20) 0               SE_ANALYTIC(8);
    SE_ANALYTIC(21) SE_ANALYTIC(22) 0;
    0               0               SE_ANALYTIC(23);
    SE_ANALYTIC(20) 0               SE_ANALYTIC(26);
    SE_ANALYTIC(24) SE_ANALYTIC(25) SE_ANALYTIC(13);
    0               0               SE_ANALYTIC(27);
    SE_ANALYTIC(28) 0               SE_ANALYTIC(30);
    SE_ANALYTIC(24) SE_ANALYTIC(29) SE_ANALYTIC(31);
    0               0               SE_ANALYTIC(32);
];

% Define matrices for each regime
SVAR_1Regime_SE = OUTPUT_Table2_SE_Analytic(1:3,:);
SVAR_2Regime_SE = OUTPUT_Table2_SE_Analytic(4:6,:);
SVAR_3Regime_SE = OUTPUT_Table2_SE_Analytic(7:9,:);
SVAR_4Regime_SE = OUTPUT_Table2_SE_Analytic(10:12,:);


%% -------- Plotting structural shocks and Var innovations ---------
% Computing structural shocks for each regime
Epsilon_1Regime = SVAR_1Regime^(-1) * Errors1';  
Epsilon_2Regime = SVAR_2Regime^(-1) * Errors2';  
Epsilon_3Regime = SVAR_3Regime^(-1) * Errors3';  
Epsilon_4Regime = SVAR_4Regime^(-1) * Errors4';

% Number of periods to plot for each regime
numPeriods_1 = size(Epsilon_1Regime, 2);
numPeriods_2 = size(Epsilon_2Regime, 2);
numPeriods_3 = size(Epsilon_3Regime, 2);
numPeriods_4 = size(Epsilon_4Regime, 2);

% Total number of periods
totalPeriods = numPeriods_1 + numPeriods_2 + numPeriods_3 + numPeriods_4;

% Variable names for the shocks
variableNamesEpsilon = {'ε_{Mt}', 'ε_{Yt}', 'ε_{Ft}'};
variableNamesU = {'u_{Mt}', 'u_{Yt}', 'u_{Ft}'};

% Plotting for Epsilon
figure(7);  
set(gcf, 'Position', [100, 100, 1000, 600]);  
for i = 1:3
    subplot(3, 1, i);  
    hold on;
    
    if i==1 || i==2
        % Shade recession regions with semi-transparency
    
        shadedRegions = [1 3; 109 120; 156 172; 230 236; 248 264; 356 364; 484 492; 565 583; 711 713];
        for j = 1:size(shadedRegions, 1)
            x_patch = [shadedRegions(j,1), shadedRegions(j,2), shadedRegions(j,2), shadedRegions(j,1)];
            y_patch = [-5.2, -5.2, 5, 5];  
            fill(x_patch, y_patch, [0.9 0.9 0.9], 'EdgeColor', 'none', 'FaceAlpha', 1, 'HandleVisibility', 'off');

        end
    else
        % Shade recession regions with semi-transparency
    
        shadedRegions = [1 3; 109 120; 156 172; 230 236; 248 264; 356 364; 484 492; 565 583; 711 713];
        for j = 1:size(shadedRegions, 1)
            x_patch = [shadedRegions(j,1), shadedRegions(j,2), shadedRegions(j,2), shadedRegions(j,1)];
            y_patch = [-10, -10, 5.4, 5.4];  
            fill(x_patch, y_patch, [0.9 0.9 0.9], 'EdgeColor', 'none', 'FaceAlpha', 1, 'HandleVisibility', 'off');

        end
    end
    plot(1:numPeriods_1, Epsilon_1Regime(i, :), 'b-', 'LineWidth', 1.5);
    plot(numPeriods_1+1:numPeriods_1+numPeriods_2, Epsilon_2Regime(i, :), 'r-', 'LineWidth', 1.5);
    plot(numPeriods_1+numPeriods_2+1:numPeriods_1+numPeriods_2+numPeriods_3, Epsilon_3Regime(i, :), 'g-', 'LineWidth', 1.5);
    plot(numPeriods_1+numPeriods_2+numPeriods_3+1:totalPeriods, Epsilon_4Regime(i, :), 'k-', 'LineWidth', 1.5);
    title(sprintf('%s Across Regimes', variableNamesEpsilon{i}));
    
   

    % Add vertical dashed lines at the transitions
    xline(numPeriods_1, 'k--', 'LineWidth', 3);
    xline(numPeriods_1 + numPeriods_2, 'k--', 'LineWidth', 3);
    xline(numPeriods_1 + numPeriods_2 + numPeriods_3, 'k--', 'LineWidth', 3);
    
    % Set x-ticks and labels
    xticks([1, 110, 230, numPeriods_1, 350, 470, numPeriods_1 + numPeriods_2, numPeriods_1 + numPeriods_2 + numPeriods_3, totalPeriods]);
    xticklabels({'12/1960', '1/1970', '1/1980', '03/1984', '1/1990', '1/2000', '12/2007', '11/2019', '06/2024'});
    xtickangle(45);
    ylabel('Magnitude');
    grid on;
    set(gca,'FontSize',FontSizeIRFGraph);
    set(gcf, 'Color', 'w'); 
    set(gca, 'Color', 'w');
    hold off;
end
legend({'1st Regime', '2nd Regime', '3rd Regime', '4th Regime'}, 'Location', 'best', 'Box', 'off');

% Plot for U
figure(8);  
set(gcf, 'Position', [100, 100, 1000, 600]);  
for i = 1:3
    subplot(3, 1, i);  
    hold on;

if i==1 || i==3
        % Shade recession regions with semi-transparency
    
        shadedRegions = [1 3; 109 120; 156 172; 230 236; 248 264; 356 364; 484 492; 565 583; 711 713];
        for j = 1:size(shadedRegions, 1)
            x_patch = [shadedRegions(j,1), shadedRegions(j,2), shadedRegions(j,2), shadedRegions(j,1)];
            y_patch = [-0.4, -0.4, 0.4, 0.4];  
            fill(x_patch, y_patch, [0.9 0.9 0.9], 'EdgeColor', 'none', 'FaceAlpha', 1, 'HandleVisibility', 'off');

        end
    else
        % Shade recession regions with semi-transparency
    
        shadedRegions = [1 3; 109 120; 156 172; 230 236; 248 264; 356 364; 484 492; 565 583; 711 713];
        for j = 1:size(shadedRegions, 1)
            x_patch = [shadedRegions(j,1), shadedRegions(j,2), shadedRegions(j,2), shadedRegions(j,1)];
            y_patch = [-5, -5, 5, 5];  
            fill(x_patch, y_patch, [0.9 0.9 0.9], 'EdgeColor', 'none', 'FaceAlpha', 1, 'HandleVisibility', 'off');

        end
    end

    plot(1:numPeriods_1, Errors1(:, i), 'b-', 'LineWidth', 1.5);
    plot(numPeriods_1+1:numPeriods_1+numPeriods_2, Errors2(:, i), 'r-', 'LineWidth', 1.5);
    plot(numPeriods_1+numPeriods_2+1:numPeriods_1+numPeriods_2+numPeriods_3, Errors3(:, i), 'g-', 'LineWidth', 1.5);
    plot(numPeriods_1+numPeriods_2+numPeriods_3+1:totalPeriods, Errors4(:, i), 'k-', 'LineWidth', 1.5);
    title(sprintf('%s Across Regimes', variableNamesU{i}));
     % Add vertical dashed lines at the transitions
    xline(numPeriods_1, 'k--', 'LineWidth', 3);
    xline(numPeriods_1 + numPeriods_2, 'k--', 'LineWidth', 3);
    xline(numPeriods_1 + numPeriods_2 + numPeriods_3, 'k--', 'LineWidth', 3);

    % Set x-ticks and labels
    xticks([1, 110, 230, numPeriods_1, 350, 470, numPeriods_1 + numPeriods_2, numPeriods_1 + numPeriods_2 + numPeriods_3, totalPeriods]);
    xticklabels({'12/1960', '1/1970', '1/1980', '03/1984', '1/1990', '1/2000', '12/2007', '11/2019', '06/2024'});
    xtickangle(45);
    ylabel('Magnitude');
    grid on;
    set(gca,'FontSize',FontSizeIRFGraph);
    set(gcf, 'Color', 'w'); 
    set(gca, 'Color', 'w');
    hold off;
end
legend({'1st Regime', '2nd Regime', '3rd Regime', '4th Regime'}, 'Location', 'best');

%% ------- Structural shocks passing 2 standard deviation ---------
threshold_Mt = 2;  % Positive threshold for ε_{Mt}
threshold_Ft = 2;  % Positive threshold for ε_{Ft}
threshold_Y = -2;  % Negative threshold for ε_{Y}

% Setup the figure
figure;
set(gcf, 'Position', [100, 100, 1000, 600]);

% Plot ε_{Mt} exceeding +2 standard deviations
subplot(3, 1, 1);
hold on;
% Concatenate all periods for ε_{Mt} 
totalEpsilon_Mt = [Epsilon_1Regime(1, :) Epsilon_2Regime(1, :) Epsilon_3Regime(1, :) Epsilon_4Regime(1, :)];
exceeds_Mt = totalEpsilon_Mt > threshold_Mt;
for j = 1:size(shadedRegions, 1)
            x_patch = [shadedRegions(j,1), shadedRegions(j,2), shadedRegions(j,2), shadedRegions(j,1)];
            y_patch = [0, 0, 6, 6];  
            fill(x_patch, y_patch, [0.9 0.9 0.9], 'EdgeColor', 'none', 'FaceAlpha', 1, 'HandleVisibility', 'off');

end
yline(2.5, 'LineWidth', 2)
% Add vertical dashed lines at the transitions
xline(numPeriods_1, 'k--', 'LineWidth', 3);
xline(numPeriods_1 + numPeriods_2, 'k--', 'LineWidth', 3);
xline(numPeriods_1 + numPeriods_2 + numPeriods_3, 'k--', 'LineWidth', 3);
bar(find(exceeds_Mt), totalEpsilon_Mt(exceeds_Mt), 'FaceColor', 'b', 'EdgeColor', 'none', 'BarWidth', 2);
title('Positive ε_{Mt} exceeding 2 standard deviations');
ylabel('Magnitude');
grid on;
set(gca,'FontSize',FontSizeIRFGraph);
xticks([1, 110, 230, numPeriods_1, 350, 470, numPeriods_1 + numPeriods_2, numPeriods_1 + numPeriods_2 + numPeriods_3, totalPeriods]);
xticklabels({'12/1960', '1/1970', '1/1980', '03/1984', '1/1990', '1/2000', '12/2007', '11/2019', '06/2024'});
xtickangle(45);

% Plot ε_{Y} falling below -2 standard deviations (show as positive bars)
subplot(3, 1, 2);
hold on;
% Concatenate all periods for ε_{Yt} 
totalEpsilon_Y = [Epsilon_1Regime(2, :) Epsilon_2Regime(2, :) Epsilon_3Regime(2, :) Epsilon_4Regime(2, :)];
falls_Y = totalEpsilon_Y < threshold_Y;
for j = 1:size(shadedRegions, 1)
            x_patch = [shadedRegions(j,1), shadedRegions(j,2), shadedRegions(j,2), shadedRegions(j,1)];
            y_patch = [0, 0, 6, 6];  
            fill(x_patch, y_patch, [0.9 0.9 0.9], 'EdgeColor', 'none', 'FaceAlpha', 1, 'HandleVisibility', 'off');

end
yline(2.5, 'LineWidth', 2)
% Add vertical dashed lines at the transitions
xline(numPeriods_1, 'k--', 'LineWidth', 3);
xline(numPeriods_1 + numPeriods_2, 'k--', 'LineWidth', 3);
xline(numPeriods_1 + numPeriods_2 + numPeriods_3, 'k--', 'LineWidth', 3);
% Multiply by -1 to make the negative shocks appear as upward bars
bar(find(falls_Y), -totalEpsilon_Y(falls_Y), 'FaceColor', 'b', 'EdgeColor', 'none', 'BarWidth', 2);
title('Negative ε_{Yt} exceeding 2 standard deviations (shown upward)');
ylabel('Magnitude of Shocks');
grid on;
set(gca,'FontSize',FontSizeIRFGraph);
xticks([1, 110, 230, numPeriods_1, 350, 470, numPeriods_1 + numPeriods_2, numPeriods_1 + numPeriods_2 + numPeriods_3, totalPeriods]);
xticklabels({'12/1960', '1/1970', '1/1980', '03/1984', '1/1990', '1/2000', '12/2007', '11/2019', '06/2024'});
xtickangle(45);

% Plot ε_{Ft} exceeding +2 standard deviations
subplot(3, 1, 3);
hold on;
% Concatenate all periods for ε_{Ft} 
totalEpsilon_Ft = [Epsilon_1Regime(3, :) Epsilon_2Regime(3, :) Epsilon_3Regime(3, :) Epsilon_4Regime(3, :)];
exceeds_Ft = totalEpsilon_Ft > threshold_Ft;
for j = 1:size(shadedRegions, 1)
            x_patch = [shadedRegions(j,1), shadedRegions(j,2), shadedRegions(j,2), shadedRegions(j,1)];
            y_patch = [0, 0, 6, 6];  
            fill(x_patch, y_patch, [0.9 0.9 0.9], 'EdgeColor', 'none', 'FaceAlpha', 1, 'HandleVisibility', 'off');

end
yline(2.5, 'LineWidth', 2)
% Add vertical dashed lines at the transitions
xline(numPeriods_1, 'k--', 'LineWidth', 3);
xline(numPeriods_1 + numPeriods_2, 'k--', 'LineWidth', 3);
xline(numPeriods_1 + numPeriods_2 + numPeriods_3, 'k--', 'LineWidth', 3);
bar(find(exceeds_Ft), totalEpsilon_Ft(exceeds_Ft), 'FaceColor', 'b', 'EdgeColor', 'none', 'BarWidth', 2);
title('Positive ε_{Ft} exceeding 2 standard deviations');
ylabel('Magnitude');
grid on;
set(gca,'FontSize',FontSizeIRFGraph);
xticks([1, 110, 230, numPeriods_1, 350, 470, numPeriods_1 + numPeriods_2, numPeriods_1 + numPeriods_2 + numPeriods_3, totalPeriods]);
xticklabels({'12/1960', '1/1970', '1/1980', '03/1984', '1/1990', '1/2000', '12/2007', '11/2019', '06/2024'});
xtickangle(45);
set(gcf, 'Color', 'w'); 
set(gca, 'Color', 'w');
hold off;

%% ------- Plot for X_t --------
figure(9);
set(gcf, 'Position', [100, 100, 1000, 600]);

% Plot for Macro Uncertainty
subplot(2, 1, 1);
std_UM1 = std(UM1);  % Calculate standard deviation of UMF
mean_UM1 = mean(UM1);
threshold_UM1 = mean(UM1) + 1.65 * std_UM1;  % Calculate threshold
shadedRegions = [1 7; 113 124; 160 176; 234 240; 252 268; 360 368; 488 496; 569 587; 715 717];
hold on;  
for j = 1:size(shadedRegions, 1)
    x_patch = [shadedRegions(j,1), shadedRegions(j,2), shadedRegions(j,2), shadedRegions(j,1)];
    
    y_patch = [0.4, 0.4, 1.4, 1.4];
    fill(x_patch, y_patch, [0.9 0.9 0.9], 'EdgeColor', 'none', 'FaceAlpha', 1, 'HandleVisibility', 'off');
end
plot(UM1, 'b-', 'LineWidth', 2);
hline = refline([0 threshold_UM1]); % Plot horizontal line at threshold
hline.Color = 'k';
hline.LineStyle = '--';
% Find and plot points above threshold
aboveThreshold_UM1 = UM1 > threshold_UM1;
plot(find(aboveThreshold_UM1), UM1(aboveThreshold_UM1), 'ko', 'MarkerFaceColor', 'k');
title('Macro Uncertainty (UM1)');
ylabel('UM1 Values');
grid on;
set(gca,'FontSize',FontSizeIRFGraph);
xticks([1, 113, 233, 284, 353, 473, 569, 712, 767]);
% Add vertical dashed lines at the transitions
xline(284, 'k--', 'LineWidth', 3);
xline(569, 'k--', 'LineWidth', 3);
xline(712, 'k--', 'LineWidth', 3);
xticklabels({'08/1960', '1/1970', '1/1980', '03/1984', '1/1990', '1/2000', '12/2007', '11/2019', '06/2024'});
xtickangle(45);
set(gcf, 'Color', 'w'); 
set(gca, 'Color', 'w');
hold off;

% Plot for Financial Uncertainty
subplot(2, 1, 2);
std_UF1 = std(UF1);  % Calculate standard deviation of UF1
mean_UF1 = mean(UF1);
threshold_UF1 = mean(UF1) + 1.65 * std_UF1;  % Calculate threshold
hold on;
for j = 1:size(shadedRegions, 1)
    x_patch = [shadedRegions(j,1), shadedRegions(j,2), shadedRegions(j,2), shadedRegions(j,1)];
    y_patch = [0.4, 0.4, 2, 2];
    fill(x_patch, y_patch, [0.9 0.9 0.9], 'EdgeColor', 'none', 'FaceAlpha', 1, 'HandleVisibility', 'off');
end
plot(UF1, 'r-', 'LineWidth', 2);
hline = refline([0 threshold_UF1]); % Plot horizontal line at threshold
hline.Color = 'k';
hline.LineStyle = '--';
% Find and plot points above threshold
aboveThreshold_UF1 = UF1 > threshold_UF1;
plot(find(aboveThreshold_UF1), UF1(aboveThreshold_UF1), 'ko', 'MarkerFaceColor', 'k');
% Add vertical dashed lines at the transitions
xline(284, 'k--', 'LineWidth', 3);
xline(569, 'k--', 'LineWidth', 3);
xline(712, 'k--', 'LineWidth', 3);
title('Financial Uncertainty (UF1)');
ylabel('UF1 Values');
grid on;
set(gca,'FontSize',FontSizeIRFGraph);
set(gcf, 'Color', 'w'); 
set(gca, 'Color', 'w');
xticks([1, 113, 233, 284, 353, 473, 569, 712, 767]);
xticklabels({'08/1960', '1/1970', '1/1980', '03/1984', '1/1990', '1/2000', '12/2007', '11/2019', '06/2024'});
xtickangle(45);
hold off;

%% Plot for Uncertainty together with real economic activity
figure(34);
set(gcf, 'Position', [100, 100, 1000, 600]);

% Shade the recession periods
hold on;
for j = 1:size(shadedRegions, 1)
    x_patch = [shadedRegions(j,1), shadedRegions(j,2), shadedRegions(j,2), shadedRegions(j,1)];
    y_patch = [0.4, 0.4, 2, 2];
    fill(x_patch, y_patch, [0.9 0.9 0.9], 'EdgeColor', 'none', 'FaceAlpha', 1, 'HandleVisibility', 'off');
end

% Plot Uncertainty on left y-axis
yyaxis left;
plot(UF1, 'r-', 'LineWidth', 2);
plot(UM1, 'b-', 'LineWidth', 2);
ylabel('Uncertainty');

% Plot Economic Activity on right y-axis
yyaxis right;
plot(Y, 'k-', 'LineWidth', 2);
ylabel('Economic Activity');

% Add vertical dashed lines at the transitions
xline(284, 'k--', 'LineWidth', 3);
xline(569, 'k--', 'LineWidth', 3);
xline(712, 'k--', 'LineWidth', 3);

grid on;
set(gca,'FontSize',FontSizeIRFGraph); 
set(gcf, 'Color', 'w'); 
set(gca, 'Color', 'w');
xticks([1, 113, 233, 284, 353, 473, 569, 712, 767]);
xticklabels({'08/1960', '1/1970', '1/1980', '03/1984', '1/1990', '1/2000', '12/2007', '11/2019', '06/2024'});
xtickangle(45);

%% ********************** BOOTSTRAP **********************


BootstrapIterations = 500;
HorizonIRF = 60;  
quant = [5,95]; % quantile bootstrap to build 90% confidence intervals

Const = Beta_LK(:, 1);
mP1 = Beta_LK(:, [2 3 4]);
mP2 = Beta_LK(:, [5 6 7]);
mP3 = Beta_LK(:, [8 9 10]);
mP4 = Beta_LK(:, [11 12 13]);

Const_1 = Beta_LK1(:, 1);
mP1_1 = Beta_LK1(:, [2 3 4]);
mP2_1 = Beta_LK1(:, [5 6 7]);
mP3_1 = Beta_LK1(:, [8 9 10]);
mP4_1 = Beta_LK1(:, [11 12 13]);

Const_2 = Beta_LK2(:, 1);
mP1_2 = Beta_LK2(:, [2 3 4]);
mP2_2 = Beta_LK2(:, [5 6 7]);
mP3_2 = Beta_LK2(:, [8 9 10]);
mP4_2 = Beta_LK2(:, [11 12 13]);

Const_3 = Beta_LK3(:, 1);
mP1_3 = Beta_LK3(:, [2 3 4]);
mP2_3 = Beta_LK3(:, [5 6 7]);
mP3_3 = Beta_LK3(:, [8 9 10]);
mP4_3 = Beta_LK3(:, [11 12 13]);

Const_4 = Beta_LK4(:, 1);
mP1_4 = Beta_LK4(:, [2 3 4]);
mP2_4 = Beta_LK4(:, [5 6 7]);
mP3_4 = Beta_LK4(:, [8 9 10]);
mP4_4 = Beta_LK4(:, [11 12 13]);

% Initialize matrixes for the computation of bootstrapped s.e.
SVAR_1_Boot = zeros(M, M, BootstrapIterations);
SVAR_2_Boot = zeros(M, M, BootstrapIterations);
SVAR_3_Boot = zeros(M, M, BootstrapIterations);
SVAR_4_Boot = zeros(M, M, BootstrapIterations);

wbGS = waitbar(0,'Running the bootstrap');
for boot = 1 : BootstrapIterations
    waitbar(boot/BootstrapIterations, wbGS, sprintf('Running the bootstrap :  Processing %d of %d', boot,BootstrapIterations)) 
    
    %{
    %  **** iid bootstrap ****
    %  **** All ***************
    TBoot = datasample(1:TAll,TAll); 
    Residuals_Boot=Errors(TBoot,:); % bootstrap errors 

    DataSet_Bootstrap=zeros(TAll+ NLags,M);
    DataSet_Bootstrap(1:NLags,:)=AllDataSet(1:NLags,:); % set the first NLags elements equal to the original sample values

        for t = 1+NLags : TAll
        DataSet_Bootstrap(t,:)=Const + mP1 * DataSet_Bootstrap(t-1,:)' +...
                                       mP2 * DataSet_Bootstrap(t-2,:)' + ...
                                       mP3 * DataSet_Bootstrap(t-3,:)' + ...
                                       mP4 * DataSet_Bootstrap(t-4,:)' + ...
                                       Residuals_Boot(t-NLags,:)';
        end

    DataSet=DataSet_Bootstrap(end-length(AllDataSet)+1:end,:);

    VAR_Variables_X=[ones(size(DataSet(NLags:end-1,:),1),1) DataSet(NLags:end-1,:) DataSet(NLags-1:end-2,:) DataSet(NLags-2:end-3,:) DataSet(NLags-3:end-4,:)];
    VAR_Variables_Y=DataSet(NLags+1:end,:);

    Beta_OLS=(VAR_Variables_X'*VAR_Variables_X)^(-1)*VAR_Variables_X'*VAR_Variables_Y;
    % try
    [Beta_LK_Boot,Log_LK_Boot,exitflag,output,grad,HESSIAN_LK_Boot] = fminunc('Likelihood_UNRESTRICTED',Beta_OLS',options2);
  
    Beta_LK_Boot;
    mP1_Boot = Beta_LK_Boot(:, [2 3 4]);
    mP2_Boot = Beta_LK_Boot(:, [5 6 7]);
    mP3_Boot = Beta_LK_Boot(:, [8 9 10]);
    mP4_Boot = Beta_LK_Boot(:, [11 12 13]);
    LK_Regime_Boot = -Log_LK;

    Errors_Boot=VAR_Variables_Y-VAR_Variables_X*Beta_LK_Boot';
    Sigma_Regime_Boot=1/(T)*Errors_Boot'*Errors_Boot;
    
    %}
     %  **** iid bootstrap ****
    %  **** First regime ***************

    TBoot = datasample(1:T1,T1); 
    Residuals_Boot=Errors1(TBoot,:); % bootstrap errors 

    DataSet_Bootstrap=zeros(T1+ NLags,M);
    DataSet_Bootstrap(1:NLags,:)=AllDataSet(1:NLags,:); % set the first NLags elements equal to the original sample values

        for t = 1+NLags : T1+NLags
        DataSet_Bootstrap(t,:)=Const_1 + mP1_1 * DataSet_Bootstrap(t-1,:)' +...
                                       mP2_1 * DataSet_Bootstrap(t-2,:)' + ...
                                       mP3_1 * DataSet_Bootstrap(t-3,:)' + ...
                                       mP4_1 * DataSet_Bootstrap(t-4,:)' + ...
                                       Residuals_Boot(t-NLags,:)';
        end

    DataSet=DataSet_Bootstrap;

    VAR_Variables_X=[ones(size(DataSet(NLags:end-1,:),1),1) DataSet(NLags:end-1,:) DataSet(NLags-1:end-2,:) DataSet(NLags-2:end-3,:) DataSet(NLags-3:end-4,:)];
    VAR_Variables_Y=DataSet(NLags+1:end,:);
    T=T1;
    Beta_OLS=(VAR_Variables_X'*VAR_Variables_X)^(-1)*VAR_Variables_X'*VAR_Variables_Y;
    
    [Beta_LK1_Boot,Log_LK_Boot,exitflag,output,grad,HESSIAN_LK_Boot] = fminunc('Likelihood_UNRESTRICTED',Beta_OLS',options2);
  
    Beta_LK1_Boot;
    mP1_Boot_1 = Beta_LK1_Boot(:, [2 3 4]);
    mP2_Boot_1 = Beta_LK1_Boot(:, [5 6 7]);
    mP3_Boot_1 = Beta_LK1_Boot(:, [8 9 10]);
    mP4_Boot_1 = Beta_LK1_Boot(:, [11 12 13]);
    LK_1Regime_Boot = -Log_LK_Boot;

    Errors1_Boot=VAR_Variables_Y-VAR_Variables_X*Beta_LK1_Boot';
    Sigma_1Regime_Boot=1/(T1)*(Errors1_Boot'*Errors1_Boot);


     %  **** iid bootstrap ****
    %  **** Second regime ***************
    
    
    TBoot = datasample(1:T2,T2); 
    Residuals_Boot=Errors2(TBoot,:); % bootstrap errors 

    DataSet_Bootstrap=zeros(T2+ NLags,M);
    DataSet_Bootstrap(1:NLags,:)=AllDataSet(1:NLags,:); % set the first NLags elements equal to the original sample values

        for t = 1+NLags : T2+NLags
        DataSet_Bootstrap(t,:)=Const_2 + mP1_2 * DataSet_Bootstrap(t-1,:)' +...
                                       mP2_2 * DataSet_Bootstrap(t-2,:)' + ...
                                       mP3_2 * DataSet_Bootstrap(t-3,:)' + ...
                                       mP4_2 * DataSet_Bootstrap(t-4,:)' + ...
                                       Residuals_Boot(t-NLags,:)';
        end

    DataSet=DataSet_Bootstrap;

    VAR_Variables_X=[ones(size(DataSet(NLags:end-1,:),1),1) DataSet(NLags:end-1,:) DataSet(NLags-1:end-2,:) DataSet(NLags-2:end-3,:) DataSet(NLags-3:end-4,:)];
    VAR_Variables_Y=DataSet(NLags+1:end,:);
    
    T=T2;
    Beta_OLS=(VAR_Variables_X'*VAR_Variables_X)^(-1)*VAR_Variables_X'*VAR_Variables_Y;
    
    [Beta_LK2_Boot,Log_LK_Boot,exitflag,output,grad,HESSIAN_LK_Boot] = fminunc('Likelihood_UNRESTRICTED',Beta_OLS',options2);
  
    Beta_LK2_Boot;
    mP1_Boot_2 = Beta_LK2_Boot(:, [2 3 4]);
    mP2_Boot_2 = Beta_LK2_Boot(:, [5 6 7]);
    mP3_Boot_2 = Beta_LK2_Boot(:, [8 9 10]);
    mP4_Boot_2 = Beta_LK2_Boot(:, [11 12 13]);
    LK_2Regime_Boot = -Log_LK_Boot;

    Errors2_Boot=VAR_Variables_Y-VAR_Variables_X*Beta_LK2_Boot';
    Sigma_2Regime_Boot=1/(T2)*(Errors2_Boot'*Errors2_Boot);


         %  **** iid bootstrap ****
    %  **** Third regime ***************
    
    
    TBoot = datasample(1:T3,T3); 
    Residuals_Boot=Errors3(TBoot,:); % bootstrap errors 

    DataSet_Bootstrap=zeros(T3+ NLags,M);
    DataSet_Bootstrap(1:NLags,:)=AllDataSet(1:NLags,:); % set the first NLags elements equal to the original sample values

        for t = 1+NLags : T3+NLags
        DataSet_Bootstrap(t,:)=Const_3 + mP1_3 * DataSet_Bootstrap(t-1,:)' +...
                                       mP2_3 * DataSet_Bootstrap(t-2,:)' + ...
                                       mP3_3 * DataSet_Bootstrap(t-3,:)' + ...
                                       mP4_3 * DataSet_Bootstrap(t-4,:)' + ...
                                       Residuals_Boot(t-NLags,:)';
        end

    DataSet=DataSet_Bootstrap;

    VAR_Variables_X=[ones(size(DataSet(NLags:end-1,:),1),1) DataSet(NLags:end-1,:) DataSet(NLags-1:end-2,:) DataSet(NLags-2:end-3,:) DataSet(NLags-3:end-4,:)];
    VAR_Variables_Y=DataSet(NLags+1:end,:);
    T=T3;
    Beta_OLS=(VAR_Variables_X'*VAR_Variables_X)^(-1)*VAR_Variables_X'*VAR_Variables_Y;
    
    [Beta_LK3_Boot,Log_LK_Boot,exitflag,output,grad,HESSIAN_LK_Boot] = fminunc('Likelihood_UNRESTRICTED',Beta_OLS',options2);
  
    Beta_LK3_Boot;
    mP1_Boot_3 = Beta_LK3_Boot(:, [2 3 4]);
    mP2_Boot_3 = Beta_LK3_Boot(:, [5 6 7]);
    mP3_Boot_3 = Beta_LK3_Boot(:, [8 9 10]);
    mP4_Boot_3 = Beta_LK3_Boot(:, [11 12 13]);
    LK_3Regime_Boot = -Log_LK_Boot;

    Errors3_Boot=VAR_Variables_Y-VAR_Variables_X*Beta_LK3_Boot';
    Sigma_3Regime_Boot=1/(T3)*(Errors3_Boot'*Errors3_Boot);

    
    
    %  **** iid bootstrap ****
    %  **** Fourth regime ***************
    
    
    TBoot = datasample(1:T4,T4); 
    Residuals_Boot=Errors4(TBoot,:); % bootstrap errors 

    DataSet_Bootstrap=zeros(T4+ NLags,M);
    DataSet_Bootstrap(1:NLags,:)=AllDataSet(1:NLags,:); % set the first NLags elements equal to the original sample values

        for t = 1+NLags : T4+NLags
        DataSet_Bootstrap(t,:)=Const_4 + mP1_4 * DataSet_Bootstrap(t-1,:)' +...
                                       mP2_4 * DataSet_Bootstrap(t-2,:)' + ...
                                       mP3_4 * DataSet_Bootstrap(t-3,:)' + ...
                                       mP4_4 * DataSet_Bootstrap(t-4,:)' + ...
                                       Residuals_Boot(t-NLags,:)';
        end

    DataSet=DataSet_Bootstrap;

    VAR_Variables_X=[ones(size(DataSet(NLags:end-1,:),1),1) DataSet(NLags:end-1,:) DataSet(NLags-1:end-2,:) DataSet(NLags-2:end-3,:) DataSet(NLags-3:end-4,:)];
    VAR_Variables_Y=DataSet(NLags+1:end,:);
    T=T4;
    Beta_OLS=(VAR_Variables_X'*VAR_Variables_X)^(-1)*VAR_Variables_X'*VAR_Variables_Y;
    
    [Beta_LK4_Boot,Log_LK_Boot,exitflag,output,grad,HESSIAN_LK_Boot] = fminunc('Likelihood_UNRESTRICTED',Beta_OLS',options2);
  
    Beta_LK4_Boot;
    mP1_Boot_4 = Beta_LK4_Boot(:, [2 3 4]);
    mP2_Boot_4 = Beta_LK4_Boot(:, [5 6 7]);
    mP3_Boot_4 = Beta_LK4_Boot(:, [8 9 10]);
    mP4_Boot_4 = Beta_LK4_Boot(:, [11 12 13]);
    LK_4Regime_Boot = -Log_LK_Boot;

    Errors4_Boot=VAR_Variables_Y-VAR_Variables_X*Beta_LK4_Boot';
    Sigma_4Regime_Boot=1/(T4)*(Errors4_Boot'*Errors4_Boot);


    % ********* estimating bootstrapped IRFs *********
                                                                                                                                
    [StructuralParam_Estimation_Boot,Likelihood_SVAR_Boot,exitflag,output,grad,Hessian_MATRIX_Boot] = fminunc('Likelihood_SVAR_Restricted_Boot',StructuralEstimationCorrected',options);
 

    SVAR_C_Boot=[StructuralParam_Estimation_Boot(1) 0                                  0;
                 StructuralParam_Estimation_Boot(2) StructuralParam_Estimation_Boot(3) 0;
                 0                                  0                                  StructuralParam_Estimation_Boot(4)];

    SVAR_Q2_Boot=[StructuralParam_Estimation_Boot(5)  0                                  StructuralParam_Estimation_Boot(8);
                  StructuralParam_Estimation_Boot(6)  StructuralParam_Estimation_Boot(7) 0;
                  0                                   0                                  StructuralParam_Estimation_Boot(9)];

    SVAR_Q3_Boot=[0                                   0                                   StructuralParam_Estimation_Boot(12);
                  StructuralParam_Estimation_Boot(10) StructuralParam_Estimation_Boot(11) StructuralParam_Estimation_Boot(13);
                  0                                   0                                   StructuralParam_Estimation_Boot(14)];

    SVAR_Q4_Boot=[StructuralParam_Estimation_Boot(15) 0                                   StructuralParam_Estimation_Boot(17);
                  0                                   StructuralParam_Estimation_Boot(16) StructuralParam_Estimation_Boot(18);
                  0                                   0                                   StructuralParam_Estimation_Boot(19)];  

    SVAR_1Regime_Boot=SVAR_C_Boot; % B
    SVAR_2Regime_Boot=SVAR_C_Boot+SVAR_Q2_Boot;   % B+Q2
    SVAR_3Regime_Boot=SVAR_C_Boot+SVAR_Q2_Boot+SVAR_Q3_Boot;  % B+Q2+Q3
    SVAR_4Regime_Boot=SVAR_C_Boot+SVAR_Q2_Boot+SVAR_Q3_Boot+SVAR_Q4_Boot;  % B+Q2+Q3+Q4

    % Flip the sign if the parameter on the main diagonal is negative

	if SVAR_1Regime_Boot(1,1)<0
    SVAR_1Regime_Boot(:,1)=-SVAR_1Regime_Boot(:,1);
    end
    if SVAR_1Regime_Boot(2,2)<0
    SVAR_1Regime_Boot(:,2)=-SVAR_1Regime_Boot(:,2); 
    end
    if SVAR_1Regime_Boot(3,3)<0
    SVAR_1Regime_Boot(:,3)=-SVAR_1Regime_Boot(:,3);
    end
    
	if SVAR_2Regime_Boot(1,1)<0
    SVAR_2Regime_Boot(:,1)=-SVAR_2Regime_Boot(:,1);
    end
    if SVAR_2Regime_Boot(2,2)<0
    SVAR_2Regime_Boot(:,2)=-SVAR_2Regime_Boot(:,2); 
    end
    if SVAR_2Regime_Boot(3,3)<0
    SVAR_2Regime_Boot(:,3)=-SVAR_2Regime_Boot(:,3);
     end
    
    
    if SVAR_3Regime_Boot(1,1)<0
    SVAR_3Regime_Boot(:,1)=-SVAR_3Regime_Boot(:,1);
    end
    if SVAR_3Regime_Boot(2,2)<0
    SVAR_3Regime_Boot(:,2)=-SVAR_3Regime_Boot(:,2); 
    end
    if SVAR_3Regime_Boot(3,3)<0
    SVAR_3Regime_Boot(:,3)=-SVAR_3Regime_Boot(:,3);
    end
    

    if SVAR_4Regime_Boot(1,1)<0
    SVAR_4Regime_Boot(:,1)=-SVAR_4Regime_Boot(:,1);
    end
    if SVAR_4Regime_Boot(2,2)<0
    SVAR_4Regime_Boot(:,2)=-SVAR_4Regime_Boot(:,2); 
    end
    if SVAR_4Regime_Boot(3,3)<0
    SVAR_4Regime_Boot(:,3)=-SVAR_4Regime_Boot(:,3);
    end
    
    SVAR_1_Boot(:, :, boot) = SVAR_1Regime_Boot;
    SVAR_2_Boot(:, :, boot) = SVAR_2Regime_Boot;
    SVAR_3_Boot(:, :, boot) = SVAR_3Regime_Boot;
    SVAR_4_Boot(:, :, boot) = SVAR_4Regime_Boot;
    
    J=[eye(M) zeros(M,M*(NLags-1))]; 

    CompanionMatrix_Boot_1=[Beta_LK1_Boot(:,2:end);
    eye(M*(NLags-1),M*(NLags-1)) zeros(M*(NLags-1),M)]; 
    
    CompanionMatrix_Boot_2=[Beta_LK2_Boot(:,2:end);
    eye(M*(NLags-1),M*(NLags-1)) zeros(M*(NLags-1),M)]; 

    CompanionMatrix_Boot_3=[Beta_LK3_Boot(:,2:end);
    eye(M*(NLags-1),M*(NLags-1)) zeros(M*(NLags-1),M)]; 

    CompanionMatrix_Boot_4=[Beta_LK4_Boot(:,2:end);
    eye(M*(NLags-1),M*(NLags-1)) zeros(M*(NLags-1),M)]; 

    for h = 0 : HorizonIRF
    TETA_Boot1(:,:,h+1,boot)=J*CompanionMatrix_Boot_1^h*J'*SVAR_1Regime_Boot;
    end    

    for h = 0 : HorizonIRF
    TETA_Boot2(:,:,h+1,boot)=J*CompanionMatrix_Boot_2^h*J'*SVAR_2Regime_Boot;
    end 

    for h = 0 : HorizonIRF
    TETA_Boot3(:,:,h+1,boot)=J*CompanionMatrix_Boot_3^h*J'*SVAR_3Regime_Boot;
    end 

    for h = 0 : HorizonIRF
    TETA_Boot4(:,:,h+1,boot)=J*CompanionMatrix_Boot_4^h*J'*SVAR_4Regime_Boot;
    end 


end 
delete(wbGS);

% Initialization of the standard error matrices
SE_SVAR_1 = zeros(M, M);
SE_SVAR_2 = zeros(M, M);
SE_SVAR_3 = zeros(M, M);
SE_SVAR_4 = zeros(M, M);

% Calculation of the standard errors for each parameter
for i = 1:M
    for j = 1:M
        SE_SVAR_1(i, j) = std(reshape(SVAR_1_Boot(i, j, :), [], 1));
        SE_SVAR_2(i, j) = std(reshape(SVAR_2_Boot(i, j, :), [], 1));
        SE_SVAR_3(i, j) = std(reshape(SVAR_3_Boot(i, j, :), [], 1));
        SE_SVAR_4(i, j) = std(reshape(SVAR_4_Boot(i, j, :), [], 1));
    end
end

IRF_Inf_Boot1 = prctile(TETA_Boot1,quant(1),4);
IRF_Sup_Boot1 = prctile(TETA_Boot1,quant(2),4);

IRF_Inf_Boot2 = prctile(TETA_Boot2,quant(1),4);
IRF_Sup_Boot2 = prctile(TETA_Boot2,quant(2),4);

IRF_Inf_Boot3 = prctile(TETA_Boot3,quant(1),4);
IRF_Sup_Boot3 = prctile(TETA_Boot3,quant(2),4);

IRF_Inf_Boot4 = prctile(TETA_Boot4,quant(1),4);
IRF_Sup_Boot4 = prctile(TETA_Boot4,quant(2),4);

%% --------------- Endogenous UM and reverse causality from UM to UF (first row of Table 2) ---------------

StructuralParam_endog=21; 
InitialValue_SVAR_Initial_endog=[
0.5;
-0.5;
0;
0.5;
0.5;
0;
0.5;
0;
0;
0.5;
0.5;
-0.5;
0.7;
0.5;
-0.5;
-0.5;
0.5;
0.5;
0.7;
0.5;
0.5]';

% ML function
[StructuralParam_Estiamtion_MATRIX_endog,Likelihood_MATRIX_endog,exitflag,output,grad,Hessian_MATRIX_endog] = fminunc('Likelihood_SVAR_Restricted_endog',InitialValue_SVAR_Initial_endog',options2);

StructuralParam_Estiamtion_endog=StructuralParam_Estiamtion_MATRIX_endog;
LK_Estimation_endog=Likelihood_MATRIX_endog;
Hessian_Estimation_endog=Hessian_MATRIX_endog;
SE_Estimation_endog=diag(Hessian_Estimation_endog^(-1)).^0.5;

%% Overidentification LR test
PVarl_LRTest_endog = 1-chi2cdf(2 * ((LK_1Regime(1)+LK_2Regime(1)+LK_3Regime(1)+LK_4Regime(1))+LK_Estimation_endog),24-StructuralParam_endog);

%% Here below we define the matrices of the structural parameters with restrictions on the coefficients as described in the paper. SVAR_C corresponds to the B matrix in the paper, SVAR_Q2 corresponds to
% the Q2 matrix of the paper, SVAR_Q3 corresponds to the Q3 matrix of the paper and SVAR_Q4 corresponds to the Q4 matrix of the paper.

SVAR_C_endog=[StructuralParam_Estiamtion_endog(1) StructuralParam_Estiamtion_endog(3) 0;
              StructuralParam_Estiamtion_endog(2) StructuralParam_Estiamtion_endog(4) 0;
              0                                   0                                   StructuralParam_Estiamtion_endog(5)];

SVAR_Q2_endog=[StructuralParam_Estiamtion_endog(6)  0                                   StructuralParam_Estiamtion_endog(10);
               StructuralParam_Estiamtion_endog(7)  StructuralParam_Estiamtion_endog(9) 0;
               StructuralParam_Estiamtion_endog(8)  0                                   StructuralParam_Estiamtion_endog(11)];

SVAR_Q3_endog=[0                                    0                                    StructuralParam_Estiamtion_endog(14);
               StructuralParam_Estiamtion_endog(12) StructuralParam_Estiamtion_endog(13) StructuralParam_Estiamtion_endog(15);
               0                                    0                                    StructuralParam_Estiamtion_endog(16)];

SVAR_Q4_endog=[StructuralParam_Estiamtion_endog(17) 0                                    StructuralParam_Estiamtion_endog(19);
               0                                    StructuralParam_Estiamtion_endog(18) StructuralParam_Estiamtion_endog(20);
               0                                    0                                    StructuralParam_Estiamtion_endog(21)];  

SVAR_1Regime_endog=SVAR_C_endog; % B
SVAR_2Regime_endog=SVAR_C_endog+SVAR_Q2_endog;   % B+Q2
SVAR_3Regime_endog=SVAR_C_endog+SVAR_Q2_endog+SVAR_Q3_endog;  % B+Q2+Q3
SVAR_4Regime_endog=SVAR_C_endog+SVAR_Q2_endog+SVAR_Q3_endog+SVAR_Q4_endog;  % B+Q2+Q3+Q4

% Flip the sign if the parameter on the main diagonal is negative

	if SVAR_1Regime_endog(1,1)<0
    SVAR_1Regime_endog(:,1)=-SVAR_1Regime_endog(:,1);
    end
    if SVAR_1Regime_endog(2,2)<0
    SVAR_1Regime_endog(:,2)=-SVAR_1Regime_endog(:,2); 
    end
    if SVAR_1Regime_endog(3,3)<0
    SVAR_1Regime_endog(:,3)=-SVAR_1Regime_endog(:,3);
    end
    
	if SVAR_2Regime_endog(1,1)<0
    SVAR_2Regime_endog(:,1)=-SVAR_2Regime_endog(:,1);
    end
    if SVAR_2Regime_endog(2,2)<0
    SVAR_2Regime_endog(:,2)=-SVAR_2Regime_endog(:,2); 
    end
    if SVAR_2Regime_endog(3,3)<0
    SVAR_2Regime_endog(:,3)=-SVAR_2Regime_endog(:,3);
     end
    
    
    if SVAR_3Regime_endog(1,1)<0
    SVAR_3Regime_endog(:,1)=-SVAR_3Regime_endog(:,1);
    end
    if SVAR_3Regime_endog(2,2)<0
    SVAR_3Regime_endog(:,2)=-SVAR_3Regime_endog(:,2); 
    end
    if SVAR_3Regime_endog(3,3)<0
    SVAR_3Regime_endog(:,3)=-SVAR_3Regime_endog(:,3);
    end
    

    if SVAR_4Regime_endog(1,1)<0
    SVAR_4Regime_endog(:,1)=-SVAR_4Regime_endog(:,1);
    end
    if SVAR_4Regime_endog(2,2)<0
    SVAR_4Regime_endog(:,2)=-SVAR_4Regime_endog(:,2); 
    end
    if SVAR_4Regime_endog(3,3)<0
    SVAR_4Regime_endog(:,3)=-SVAR_4Regime_endog(:,3);
    end
    
     
MATRICES_endog=[SVAR_1Regime_endog;
                SVAR_2Regime_endog;
                SVAR_3Regime_endog;
                SVAR_4Regime_endog];
   
% Calculates the analytical derivatives organized in block matrices      
V11=2*NMatrix*kron(SVAR_C_endog,eye(M));
V21=2*NMatrix*kron(SVAR_C_endog,eye(M))+kron(SVAR_Q2_endog,eye(M))+kron(eye(M),SVAR_Q2_endog)*KommutationMatrix;
V22=kron(eye(M),SVAR_C_endog)*KommutationMatrix+kron(SVAR_C_endog,eye(M))+2*NMatrix*kron(SVAR_Q2_endog,eye(M));
V31=2*NMatrix*kron(SVAR_C_endog,eye(M))+kron(SVAR_Q2_endog,eye(M))+kron(SVAR_Q3_endog,eye(M))+kron(eye(M),SVAR_Q2_endog)*KommutationMatrix+kron(eye(M),SVAR_Q3_endog)*KommutationMatrix;
V32=kron(eye(M),SVAR_C_endog)*KommutationMatrix+kron(SVAR_C_endog,eye(M))+2*NMatrix*kron(SVAR_Q2_endog,eye(M))+kron(SVAR_Q3_endog,eye(M))+kron(eye(M),SVAR_Q3_endog)*KommutationMatrix;
V33=kron(eye(M),SVAR_C_endog)*KommutationMatrix+kron(eye(M),SVAR_Q2_endog)*KommutationMatrix+2*NMatrix*kron(SVAR_Q3_endog,eye(M))+kron(SVAR_C_endog,eye(M))+kron(SVAR_Q2_endog,eye(M));
V41 = 2 * NMatrix * kron(SVAR_C_endog, eye(M)) + kron(SVAR_Q2_endog, eye(M)) + kron(SVAR_Q3_endog, eye(M)) + kron(SVAR_Q4_endog, eye(M)) + kron(eye(M), SVAR_Q2_endog) * KommutationMatrix + kron(eye(M), SVAR_Q3_endog) * KommutationMatrix + kron(eye(M), SVAR_Q4_endog) * KommutationMatrix;
V42 = kron(eye(M), SVAR_C_endog) * KommutationMatrix + kron(SVAR_C_endog, eye(M)) + 2 * NMatrix * kron(SVAR_Q2_endog, eye(M)) + kron(SVAR_Q3_endog, eye(M)) + kron(SVAR_Q4_endog, eye(M)) + kron(eye(M), SVAR_Q3_endog) * KommutationMatrix + kron(eye(M), SVAR_Q4_endog) * KommutationMatrix;
V43 = kron(eye(M), SVAR_C_endog) * KommutationMatrix + kron(eye(M), SVAR_Q2_endog) * KommutationMatrix + 2 * NMatrix * kron(SVAR_Q3_endog, eye(M)) + kron(SVAR_C_endog, eye(M)) + kron(SVAR_Q2_endog, eye(M)) + kron(SVAR_Q4_endog, eye(M)) + kron(eye(M), SVAR_Q4_endog) * KommutationMatrix;
V44 = kron(eye(M), SVAR_C_endog) * KommutationMatrix + kron(eye(M), SVAR_Q2_endog) * KommutationMatrix + kron(eye(M), SVAR_Q3_endog) * KommutationMatrix + 2 * NMatrix * kron(SVAR_Q4_endog, eye(M)) + kron(SVAR_C_endog, eye(M)) + kron(SVAR_Q2_endog, eye(M)) + kron(SVAR_Q3_endog, eye(M));

% Calculates the matrix for checking the rank condition (full column rank)
RankMatrix = kron(eye(4), mDD) * [V11 zeros(M^2, M^2) zeros(M^2, M^2) zeros(M^2, M^2);
                                  V21 V22 zeros(M^2, M^2) zeros(M^2, M^2);
                                  V31 V32 V33 zeros(M^2, M^2);
                                  V41 V42 V43 V44];

 
% Selection matrix for extracting the structural parameters                         
HSelection=zeros(M*M*4,StructuralParam_endog);
HSelection(1,1)=1;
HSelection(2,2)=1;
HSelection(4,3)=1;
HSelection(5,4)=1;
HSelection(9,5)=1;
HSelection(10,6)=1;
HSelection(11,7)=1;
HSelection(12,8)=1;
HSelection(14,9)=1;
HSelection(16,10)=1;
HSelection(18,11)=1;
HSelection(20,12)=1;
HSelection(23,13)=1;
HSelection(25,14)=1;
HSelection(26, 15) = 1;
HSelection(27, 16) = 1;  
HSelection(28, 17) = 1;  
HSelection(32, 18) = 1;  
HSelection(34, 19) = 1;  
HSelection(35, 20) = 1;
HSelection(36, 21) = 1;

 
Jacobian_endog= RankMatrix*HSelection;

% Report the rank of the matrix for checking the identification
Jacobian_rank_endog = rank(Jacobian_endog);

MSigma=size(StandardErrorSigma_1Regime,1);

TetaMatrix = [
    StandardErrorSigma_1Regime, zeros(MSigma, MSigma),      zeros(MSigma, MSigma),      zeros(MSigma, MSigma);
    zeros(MSigma, MSigma),      StandardErrorSigma_2Regime, zeros(MSigma, MSigma),      zeros(MSigma, MSigma);
    zeros(MSigma, MSigma),      zeros(MSigma, MSigma),      StandardErrorSigma_3Regime, zeros(MSigma, MSigma);
    zeros(MSigma, MSigma),      zeros(MSigma, MSigma),      zeros(MSigma, MSigma),      StandardErrorSigma_4Regime
];

     
% Calculates the variance and the standard errors of the estimated coefficients        
VarTeta=  (Jacobian_endog'* TetaMatrix^(-1)*Jacobian_endog)^(-1);
SETetaJacobian_endog= diag(VarTeta).^0.5;

%% Structural parameters
StructuralEstimationCorrected_endog=[
        MATRICES_endog(1,1);
        MATRICES_endog(2,1);
        MATRICES_endog(1,2);
        MATRICES_endog(2,2);
        MATRICES_endog(3,3);

        MATRICES_endog(4,1)-MATRICES_endog(1,1);
        MATRICES_endog(5,1)-MATRICES_endog(2,1);
        MATRICES_endog(6,1);
        MATRICES_endog(5,2)-MATRICES_endog(2,2);
        MATRICES_endog(4,3);
        MATRICES_endog(6,3)-MATRICES_endog(3,3);
        
        MATRICES_endog(8,1)-MATRICES_endog(5,1); 
        MATRICES_endog(8,2)-MATRICES_endog(5,2); 
        MATRICES_endog(7,3)-MATRICES_endog(4,3);
        MATRICES_endog(8,3);
        MATRICES_endog(9,3)-MATRICES_endog(6,3);
        
        MATRICES_endog(10,1) - MATRICES_endog(7,1);
        MATRICES_endog(11,2) - MATRICES_endog(8,2);
        MATRICES_endog(10,3)- MATRICES_endog(7,3);
        MATRICES_endog(11,3)- MATRICES_endog(8,3);
        MATRICES_endog(12,3) - MATRICES_endog(9,3);
        ];
 
OUTPUT_Table2_StructuralEstimation_endog=[StructuralEstimationCorrected_endog SE_Estimation_endog SETetaJacobian_endog];

%% Estimation of the standard errors of the parameters in B, B+Q2, B+Q2+Q3, and B+Q2+Q3+Q4 (delta method)

% Inverse of the Hessian estimation for variance estimation
VAR_Est_endog = Hessian_Estimation_endog^(-1);

% Define symbolic variables for delta method calculations
syms x y z w
gradient_sigma_Matrix = [];

%% Calculation for B+Q2 combination
% teta(6)
index = 1;
i = 1; j = 6;
first_par = OUTPUT_Table2_StructuralEstimation_endog(i,1);
second_par = OUTPUT_Table2_StructuralEstimation_endog(j,1);
f = x + y;
gradient_sigma = gradient(f, [x, y]);
gradient_sigma_est = subs(gradient_sigma, [x y], [first_par second_par]);
gradient_sigma_Matrix = double(gradient_sigma_est)';
SETetaDelta_endog(index, :) = sqrt(diag(gradient_sigma_Matrix * VAR_Est_endog([i j], [i j]) * gradient_sigma_Matrix'));

% teta(7)
index = 2;
i = 2; j = 7;
first_par = OUTPUT_Table2_StructuralEstimation_endog(i,1);
second_par = OUTPUT_Table2_StructuralEstimation_endog(j,1);
f = x + y;
gradient_sigma = gradient(f, [x, y]);
gradient_sigma_est = subs(gradient_sigma, [x y], [first_par second_par]);
gradient_sigma_Matrix = double(gradient_sigma_est)';
SETetaDelta_endog(index, :) = sqrt(diag(gradient_sigma_Matrix * VAR_Est_endog([i j], [i j]) * gradient_sigma_Matrix'));

% teta(9)
index = 3;
i = 4; j = 9;
first_par = OUTPUT_Table2_StructuralEstimation_endog(i,1);
second_par = OUTPUT_Table2_StructuralEstimation_endog(j,1);
f = x + y;
gradient_sigma = gradient(f, [x, y]);
gradient_sigma_est = subs(gradient_sigma, [x y], [first_par second_par]);
gradient_sigma_Matrix = double(gradient_sigma_est)';
SETetaDelta_endog(index, :) = sqrt(diag(gradient_sigma_Matrix * VAR_Est_endog([i j], [i j]) * gradient_sigma_Matrix'));

% teta(11)
index = 4;
i = 5; j = 11;
first_par = OUTPUT_Table2_StructuralEstimation_endog(i,1);
second_par = OUTPUT_Table2_StructuralEstimation_endog(j,1);
f = x + y;
gradient_sigma = gradient(f, [x, y]);
gradient_sigma_est = subs(gradient_sigma, [x y], [first_par second_par]);
gradient_sigma_Matrix = double(gradient_sigma_est)';
SETetaDelta_endog(index, :) = sqrt(diag(gradient_sigma_Matrix * VAR_Est_endog([i j], [i j]) * gradient_sigma_Matrix'));

%% Calculation for B+Q2+Q3 combination

% teta(12)
index = 5;
i = 2; j = 7; k = 12;
first_par = OUTPUT_Table2_StructuralEstimation_endog(i,1);
second_par = OUTPUT_Table2_StructuralEstimation_endog(j,1);
third_par = OUTPUT_Table2_StructuralEstimation_endog(k,1);
f = x + y + z;
gradient_sigma = gradient(f, [x, y, z]);
gradient_sigma_est = subs(gradient_sigma, [x y z], [first_par second_par third_par]);
gradient_sigma_Matrix = double(gradient_sigma_est)';
SETetaDelta_endog(index, :) = sqrt(diag(gradient_sigma_Matrix * VAR_Est_endog([i j k], [i j k]) * gradient_sigma_Matrix'));

% teta(13)
index = 6;
i = 4; j = 9; k = 13;
first_par = OUTPUT_Table2_StructuralEstimation_endog(i,1);
second_par = OUTPUT_Table2_StructuralEstimation_endog(j,1);
third_par = OUTPUT_Table2_StructuralEstimation_endog(k,1);
f = x + y + z;
gradient_sigma = gradient(f, [x, y, z]);
gradient_sigma_est = subs(gradient_sigma, [x y z], [first_par second_par third_par]);
gradient_sigma_Matrix = double(gradient_sigma_est)';
SETetaDelta_endog(index, :) = sqrt(diag(gradient_sigma_Matrix * VAR_Est_endog([i j k], [i j k]) * gradient_sigma_Matrix'));

% teta(14)
index = 7;
j = 10; k = 14;
second_par = OUTPUT_Table2_StructuralEstimation_endog(j,1);
third_par = OUTPUT_Table2_StructuralEstimation_endog(k,1);
f = y + z;
gradient_sigma = gradient(f, [y, z]);
gradient_sigma_est = subs(gradient_sigma, [y z], [second_par third_par]);
gradient_sigma_Matrix = double(gradient_sigma_est)';
SETetaDelta_endog(index, :) = sqrt(diag(gradient_sigma_Matrix * VAR_Est_endog([j k], [j k]) * gradient_sigma_Matrix'));

% teta(16)
index = 8;
i = 5; j = 11; l = 16;
first_par = OUTPUT_Table2_StructuralEstimation_endog(i,1);
second_par = OUTPUT_Table2_StructuralEstimation_endog(j,1);
fourth_par = OUTPUT_Table2_StructuralEstimation_endog(l,1);
f = x + y + w;
gradient_sigma = gradient(f, [x, y, w]);
gradient_sigma_est = subs(gradient_sigma, [x y w], [first_par second_par fourth_par]);
gradient_sigma_Matrix = double(gradient_sigma_est)';
SETetaDelta_endog(index, :) = sqrt(diag(gradient_sigma_Matrix * VAR_Est_endog([i j l], [i j l]) * gradient_sigma_Matrix'));

%% Calculation for B+Q2+Q3+Q4 combination

% teta(17)
index = 9;
i = 1; j = 6; l = 17;
first_par = OUTPUT_Table2_StructuralEstimation_endog(i,1);
second_par = OUTPUT_Table2_StructuralEstimation_endog(j,1);
fourth_par = OUTPUT_Table2_StructuralEstimation_endog(l,1);
f = x + y + w;
gradient_sigma = gradient(f, [x, y, w]);
gradient_sigma_est = subs(gradient_sigma, [x y w], [first_par second_par fourth_par]);
gradient_sigma_Matrix = double(gradient_sigma_est)';
SETetaDelta_endog(index, :) = sqrt(diag(gradient_sigma_Matrix * VAR_Est_endog([i j l], [i j l]) * gradient_sigma_Matrix'));

% teta(18)
index = 10;
i = 4; j = 9; k = 13; l = 18;
first_par = OUTPUT_Table2_StructuralEstimation_endog(i,1);
second_par = OUTPUT_Table2_StructuralEstimation_endog(j,1);
third_par = OUTPUT_Table2_StructuralEstimation_endog(k,1);
fourth_par = OUTPUT_Table2_StructuralEstimation_endog(l,1);
f = x + y + z + w;
gradient_sigma = gradient(f, [x, y, z, w]);
gradient_sigma_est = subs(gradient_sigma, [x y z w], [first_par second_par third_par fourth_par]);
gradient_sigma_Matrix = double(gradient_sigma_est)';
SETetaDelta_endog(index, :) = sqrt(diag(gradient_sigma_Matrix * VAR_Est_endog([i j k l], [i j k l]) * gradient_sigma_Matrix'));

% teta(19)
index = 11;
j = 10; k = 14; l = 19;
second_par = OUTPUT_Table2_StructuralEstimation_endog(j,1);
third_par = OUTPUT_Table2_StructuralEstimation_endog(k,1);
fourth_par = OUTPUT_Table2_StructuralEstimation_endog(l,1);
f = y + z + w;
gradient_sigma = gradient(f, [y, z, w]);
gradient_sigma_est = subs(gradient_sigma, [y z w], [second_par third_par fourth_par]);
gradient_sigma_Matrix = double(gradient_sigma_est)';
SETetaDelta_endog(index, :) = sqrt(diag(gradient_sigma_Matrix * VAR_Est_endog([j k l], [j k l]) * gradient_sigma_Matrix'));

% teta(20)
index = 12;
k = 15; l = 20;
third_par = OUTPUT_Table2_StructuralEstimation_endog(k,1);
fourth_par = OUTPUT_Table2_StructuralEstimation_endog(l,1);
f = z + w;
gradient_sigma = gradient(f, [z, w]);
gradient_sigma_est = subs(gradient_sigma, [z w], [third_par fourth_par]);
gradient_sigma_Matrix = double(gradient_sigma_est)';
SETetaDelta_endog(index, :) = sqrt(diag(gradient_sigma_Matrix * VAR_Est_endog([k l], [k l]) * gradient_sigma_Matrix'));

% teta(21)
index = 13;
i = 5; j = 11; k = 16; l = 21;
first_par = OUTPUT_Table2_StructuralEstimation_endog(i,1);
second_par = OUTPUT_Table2_StructuralEstimation_endog(j,1);
third_par = OUTPUT_Table2_StructuralEstimation_endog(k,1);
fourth_par = OUTPUT_Table2_StructuralEstimation_endog(l,1);
f = x + y + z + w;
gradient_sigma = gradient(f, [x, y, z, w]);
gradient_sigma_est = subs(gradient_sigma, [x y z w], [first_par second_par third_par fourth_par]);
gradient_sigma_Matrix = double(gradient_sigma_est)';
SETetaDelta_endog(index, :) = sqrt(diag(gradient_sigma_Matrix * VAR_Est_endog([i j k l], [i j k l]) * gradient_sigma_Matrix'));

%% Concatenate all standard errors into one matrix for output
SE_ANALYTIC_endog = [SE_Estimation_endog; SETetaDelta_endog];

% Organizing standard errors into a structured output matrix
OUTPUT_Table2_SE_Analytic_endog = [
    SE_ANALYTIC_endog(1)  SE_ANALYTIC_endog(3)  0;
    SE_ANALYTIC_endog(2)  SE_ANALYTIC_endog(4)  0;  
    0                     0                     SE_ANALYTIC_endog(5); 
    %
    SE_ANALYTIC_endog(22) SE_ANALYTIC_endog(3)  SE_ANALYTIC_endog(10);
    SE_ANALYTIC_endog(23) SE_ANALYTIC_endog(24) 0;
    SE_ANALYTIC_endog(8)  0                     SE_ANALYTIC_endog(25);
    %
    SE_ANALYTIC_endog(22) SE_ANALYTIC_endog(3)  SE_ANALYTIC_endog(28);
    SE_ANALYTIC_endog(26) SE_ANALYTIC_endog(27) SE_ANALYTIC_endog(15);
    SE_ANALYTIC_endog(8)  0                     SE_ANALYTIC_endog(29);
    %
    SE_ANALYTIC_endog(30) SE_ANALYTIC_endog(3)  SE_ANALYTIC_endog(32);
    SE_ANALYTIC_endog(26) SE_ANALYTIC_endog(31) SE_ANALYTIC_endog(33);
    SE_ANALYTIC_endog(8)  0                     SE_ANALYTIC_endog(34);
];

% Define matrices for each regime
SVAR_1Regime_SE_endog = OUTPUT_Table2_SE_Analytic_endog(1:3,:);
SVAR_2Regime_SE_endog = OUTPUT_Table2_SE_Analytic_endog(4:6,:);
SVAR_3Regime_SE_endog = OUTPUT_Table2_SE_Analytic_endog(7:9,:);
SVAR_4Regime_SE_endog = OUTPUT_Table2_SE_Analytic_endog(10:12,:);

PVarl_LRTest_ex_end = 1-chi2cdf(-2 * (LK_Estimation_endog-LK_Estimation), (StructuralParam_endog - StructuralParam));

%% --------------- Endogenous UM and reverse causality from UM to UF (Covid), (second row of Table 2) ---------------

StructuralParam_endog2=21; 
InitialValue_SVAR_Initial_endog2 = 0.5*ones(StructuralParam_endog2,1);

% ML function
[StructuralParam_Estiamtion_MATRIX_endog2,Likelihood_MATRIX_endog2,exitflag,output,grad,Hessian_MATRIX_endog2] = fminunc('Likelihood_SVAR_Restricted_endog2',InitialValue_SVAR_Initial_endog2',options2);

StructuralParam_Estiamtion_endog2=StructuralParam_Estiamtion_MATRIX_endog2;
LK_Estimation_endog2=Likelihood_MATRIX_endog2;
Hessian_Estimation_endog2=Hessian_MATRIX_endog2;
SE_Estimation_endog2=diag(Hessian_Estimation_endog2^(-1)).^0.5;

%% Overidentification LR test
PVarl_LRTest_endog2 = 1-chi2cdf(2 * ((LK_1Regime(1)+LK_2Regime(1)+LK_3Regime(1)+LK_4Regime(1))+LK_Estimation_endog2),24-StructuralParam_endog2);

%% Here below we define the matrices of the structural parameters with restrictions on the coefficients as described in the paper. SVAR_C corresponds to the B matrix in the paper, SVAR_Q2 corresponds to
% the Q2 matrix of the paper, SVAR_Q3 corresponds to the Q3 matrix of the paper and SVAR_Q4 corresponds to the Q4 matrix of the paper.

SVAR_C_endog2=[StructuralParam_Estiamtion_endog2(1) 0                                    0;
               StructuralParam_Estiamtion_endog2(2) StructuralParam_Estiamtion_endog2(3) 0;
               0                                    0                                    StructuralParam_Estiamtion_endog2(4)];

SVAR_Q2_endog2=[StructuralParam_Estiamtion_endog2(5) 0                                    StructuralParam_Estiamtion_endog2(8);
                StructuralParam_Estiamtion_endog2(6) StructuralParam_Estiamtion_endog2(7) 0;
                0                                    0                                    StructuralParam_Estiamtion_endog2(9)];

SVAR_Q3_endog2=[0                                     0                                     StructuralParam_Estiamtion_endog2(13);
                StructuralParam_Estiamtion_endog2(10) StructuralParam_Estiamtion_endog2(12) StructuralParam_Estiamtion_endog2(14);
                StructuralParam_Estiamtion_endog2(11) 0                                     StructuralParam_Estiamtion_endog2(15)];

SVAR_Q4_endog2=[StructuralParam_Estiamtion_endog2(16) StructuralParam_Estiamtion_endog2(17) StructuralParam_Estiamtion_endog2(19);
                0                                     StructuralParam_Estiamtion_endog2(18) StructuralParam_Estiamtion_endog2(20);
                0                                     0                                     StructuralParam_Estiamtion_endog2(21)];  

SVAR_1Regime_endog2=SVAR_C_endog2; % B
SVAR_2Regime_endog2=SVAR_C_endog2+SVAR_Q2_endog2;   % B+Q2
SVAR_3Regime_endog2=SVAR_C_endog2+SVAR_Q2_endog2+SVAR_Q3_endog2;  % B+Q2+Q3
SVAR_4Regime_endog2=SVAR_C_endog2+SVAR_Q2_endog2+SVAR_Q3_endog2+SVAR_Q4_endog2;  % B+Q2+Q3+Q4

% Flip the sign if the parameter on the main diagonal is negative

	if SVAR_1Regime_endog2(1,1)<0
    SVAR_1Regime_endog2(:,1)=-SVAR_1Regime_endog2(:,1);
    end
    if SVAR_1Regime_endog2(2,2)<0
    SVAR_1Regime_endog2(:,2)=-SVAR_1Regime_endog2(:,2); 
    end
    if SVAR_1Regime_endog2(3,3)<0
    SVAR_1Regime_endog2(:,3)=-SVAR_1Regime_endog2(:,3);
    end
    
	if SVAR_2Regime_endog2(1,1)<0
    SVAR_2Regime_endog2(:,1)=-SVAR_2Regime_endog2(:,1);
    end
    if SVAR_2Regime_endog2(2,2)<0
    SVAR_2Regime_endog2(:,2)=-SVAR_2Regime_endog2(:,2); 
    end
    if SVAR_2Regime_endog2(3,3)<0
    SVAR_2Regime_endog2(:,3)=-SVAR_2Regime_endog2(:,3);
     end
    
    
    if SVAR_3Regime_endog2(1,1)<0
    SVAR_3Regime_endog2(:,1)=-SVAR_3Regime_endog2(:,1);
    end
    if SVAR_3Regime_endog2(2,2)<0
    SVAR_3Regime_endog2(:,2)=-SVAR_3Regime_endog2(:,2); 
    end
    if SVAR_3Regime_endog2(3,3)<0
    SVAR_3Regime_endog2(:,3)=-SVAR_3Regime_endog2(:,3);
    end
    

    if SVAR_4Regime_endog2(1,1)<0
    SVAR_4Regime_endog2(:,1)=-SVAR_4Regime_endog2(:,1);
    end
    if SVAR_4Regime_endog2(2,2)<0
    SVAR_4Regime_endog2(:,2)=-SVAR_4Regime_endog2(:,2); 
    end
    if SVAR_4Regime_endog2(3,3)<0
    SVAR_4Regime_endog2(:,3)=-SVAR_4Regime_endog2(:,3);
    end
    
     
MATRICES_endog2=[SVAR_1Regime_endog2;
                 SVAR_2Regime_endog2;
                 SVAR_3Regime_endog2;
                 SVAR_4Regime_endog2];
   
% Calculates the analytical derivatives organized in block matrices      
V11=2*NMatrix*kron(SVAR_C_endog2,eye(M));
V21=2*NMatrix*kron(SVAR_C_endog2,eye(M))+kron(SVAR_Q2_endog2,eye(M))+kron(eye(M),SVAR_Q2_endog2)*KommutationMatrix;
V22=kron(eye(M),SVAR_C_endog2)*KommutationMatrix+kron(SVAR_C_endog2,eye(M))+2*NMatrix*kron(SVAR_Q2_endog2,eye(M));
V31=2*NMatrix*kron(SVAR_C_endog2,eye(M))+kron(SVAR_Q2_endog2,eye(M))+kron(SVAR_Q3_endog2,eye(M))+kron(eye(M),SVAR_Q2_endog2)*KommutationMatrix+kron(eye(M),SVAR_Q3_endog2)*KommutationMatrix;
V32=kron(eye(M),SVAR_C_endog2)*KommutationMatrix+kron(SVAR_C_endog2,eye(M))+2*NMatrix*kron(SVAR_Q2_endog2,eye(M))+kron(SVAR_Q3_endog2,eye(M))+kron(eye(M),SVAR_Q3_endog2)*KommutationMatrix;
V33=kron(eye(M),SVAR_C_endog2)*KommutationMatrix+kron(eye(M),SVAR_Q2_endog2)*KommutationMatrix+2*NMatrix*kron(SVAR_Q3_endog2,eye(M))+kron(SVAR_C_endog2,eye(M))+kron(SVAR_Q2_endog2,eye(M));
V41 = 2 * NMatrix * kron(SVAR_C_endog2, eye(M)) + kron(SVAR_Q2_endog2, eye(M)) + kron(SVAR_Q3_endog2, eye(M)) + kron(SVAR_Q4_endog2, eye(M)) + kron(eye(M), SVAR_Q2_endog2) * KommutationMatrix + kron(eye(M), SVAR_Q3_endog2) * KommutationMatrix + kron(eye(M), SVAR_Q4_endog2) * KommutationMatrix;
V42 = kron(eye(M), SVAR_C_endog2) * KommutationMatrix + kron(SVAR_C_endog2, eye(M)) + 2 * NMatrix * kron(SVAR_Q2_endog2, eye(M)) + kron(SVAR_Q3_endog2, eye(M)) + kron(SVAR_Q4_endog2, eye(M)) + kron(eye(M), SVAR_Q3_endog2) * KommutationMatrix + kron(eye(M), SVAR_Q4_endog2) * KommutationMatrix;
V43 = kron(eye(M), SVAR_C_endog2) * KommutationMatrix + kron(eye(M), SVAR_Q2_endog2) * KommutationMatrix + 2 * NMatrix * kron(SVAR_Q3_endog2, eye(M)) + kron(SVAR_C_endog2, eye(M)) + kron(SVAR_Q2_endog2, eye(M)) + kron(SVAR_Q4_endog2, eye(M)) + kron(eye(M), SVAR_Q4_endog2) * KommutationMatrix;
V44 = kron(eye(M), SVAR_C_endog2) * KommutationMatrix + kron(eye(M), SVAR_Q2_endog2) * KommutationMatrix + kron(eye(M), SVAR_Q3_endog2) * KommutationMatrix + 2 * NMatrix * kron(SVAR_Q4_endog2, eye(M)) + kron(SVAR_C_endog2, eye(M)) + kron(SVAR_Q2_endog2, eye(M)) + kron(SVAR_Q3_endog2, eye(M));

% Calculates the matrix for checking the rank condition (full column rank)
RankMatrix = kron(eye(4), mDD) * [V11 zeros(M^2, M^2) zeros(M^2, M^2) zeros(M^2, M^2);
                                  V21 V22 zeros(M^2, M^2) zeros(M^2, M^2);
                                  V31 V32 V33 zeros(M^2, M^2);
                                  V41 V42 V43 V44];

 
% Selection matrix for extracting the structural parameters                         
HSelection=zeros(M*M*4,StructuralParam_endog2);
HSelection(1,1)=1;
HSelection(2,2)=1;
HSelection(5,3)=1;
HSelection(9,4)=1;
HSelection(10,5)=1;
HSelection(11,6)=1;
HSelection(14,7)=1;
HSelection(16,8)=1;
HSelection(18,9)=1;
HSelection(20,10)=1;
HSelection(21,11)=1;
HSelection(23,12)=1;
HSelection(25,13)=1;
HSelection(26,14)=1;
HSelection(27,15) = 1;
HSelection(28,16) = 1;  
HSelection(31,17) = 1;  
HSelection(32,18) = 1;  
HSelection(34,19) = 1;  
HSelection(35,20) = 1;
HSelection(36,21) = 1;
 
Jacobian_endog2= RankMatrix*HSelection;

% Report the rank of the matrix for checking the identification
Jacobian_rank_endog2 = rank(Jacobian_endog2);

MSigma=size(StandardErrorSigma_1Regime,1);

TetaMatrix = [
    StandardErrorSigma_1Regime, zeros(MSigma, MSigma),      zeros(MSigma, MSigma),      zeros(MSigma, MSigma);
    zeros(MSigma, MSigma),      StandardErrorSigma_2Regime, zeros(MSigma, MSigma),      zeros(MSigma, MSigma);
    zeros(MSigma, MSigma),      zeros(MSigma, MSigma),      StandardErrorSigma_3Regime, zeros(MSigma, MSigma);
    zeros(MSigma, MSigma),      zeros(MSigma, MSigma),      zeros(MSigma, MSigma),      StandardErrorSigma_4Regime
];

    
% Calculates the variance and the standard errors of the estimated coefficients        
VarTeta=  (Jacobian_endog2'* TetaMatrix^(-1)*Jacobian_endog2)^(-1);
SETetaJacobian_endog2= diag(VarTeta).^0.5;

%% Structural parameters
StructuralEstimationCorrected_endog2=[
        MATRICES_endog2(1,1);
        MATRICES_endog2(2,1);
        MATRICES_endog2(2,2);
        MATRICES_endog2(3,3);

        MATRICES_endog2(4,1)-MATRICES_endog2(1,1);
        MATRICES_endog2(5,1)-MATRICES_endog2(2,1);
        MATRICES_endog2(5,2)-MATRICES_endog2(2,2);
        MATRICES_endog2(4,3);
        MATRICES_endog2(6,3)-MATRICES_endog2(3,3);
        
        MATRICES_endog2(8,1)-MATRICES_endog2(5,1);
        MATRICES_endog2(9,1);
        MATRICES_endog2(8,2)-MATRICES_endog2(5,2);
        MATRICES_endog2(7,3)-MATRICES_endog2(4,3);
        MATRICES_endog2(8,3);
        MATRICES_endog2(9,3)-MATRICES_endog2(6,3);
        
        MATRICES_endog2(10,1) - MATRICES_endog2(7,1);
        MATRICES_endog2(10,2);
        MATRICES_endog2(11,2) - MATRICES_endog2(8,2);
        MATRICES_endog2(10,3)- MATRICES_endog2(7,3);
        MATRICES_endog2(11,3)- MATRICES_endog2(8,3);
        MATRICES_endog2(12,3) - MATRICES_endog2(9,3);
        ];

OUTPUT_Table2_StructuralEstimation_endog2=[StructuralEstimationCorrected_endog2 SE_Estimation_endog2 SETetaJacobian_endog2];

%% Estimation of the standard errors of the parameters in B, B+Q2, B+Q2+Q3, and B+Q2+Q3+Q4 (delta method)

% Inverse of the Hessian estimation for variance estimation
VAR_Est_endog2 = Hessian_Estimation_endog2^(-1);

% Define symbolic variables for delta method calculations
syms x y z w
gradient_sigma_Matrix = [];

%% Calculation for B+Q2 combination
% teta(5)
index = 1;
i = 1; j = 5;
first_par = OUTPUT_Table2_StructuralEstimation_endog2(i,1);
second_par = OUTPUT_Table2_StructuralEstimation_endog2(j,1);
f = x + y;
gradient_sigma = gradient(f, [x, y]);
gradient_sigma_est = subs(gradient_sigma, [x y], [first_par second_par]);
gradient_sigma_Matrix = double(gradient_sigma_est)';
SETetaDelta_endog2(index, :) = sqrt(diag(gradient_sigma_Matrix * VAR_Est_endog2([i j], [i j]) * gradient_sigma_Matrix'));

% teta(6)
index = 2;
i = 2; j = 6;
first_par = OUTPUT_Table2_StructuralEstimation_endog2(i,1);
second_par = OUTPUT_Table2_StructuralEstimation_endog2(j,1);
f = x + y;
gradient_sigma = gradient(f, [x, y]);
gradient_sigma_est = subs(gradient_sigma, [x y], [first_par second_par]);
gradient_sigma_Matrix = double(gradient_sigma_est)';
SETetaDelta_endog2(index, :) = sqrt(diag(gradient_sigma_Matrix * VAR_Est_endog2([i j], [i j]) * gradient_sigma_Matrix'));

% teta(7)
index = 3;
i = 3; j = 7;
first_par = OUTPUT_Table2_StructuralEstimation_endog2(i,1);
second_par = OUTPUT_Table2_StructuralEstimation_endog2(j,1);
f = x + y;
gradient_sigma = gradient(f, [x, y]);
gradient_sigma_est = subs(gradient_sigma, [x y], [first_par second_par]);
gradient_sigma_Matrix = double(gradient_sigma_est)';
SETetaDelta_endog2(index, :) = sqrt(diag(gradient_sigma_Matrix * VAR_Est_endog2([i j], [i j]) * gradient_sigma_Matrix'));

% teta(9)
index = 4;
i = 4; j = 9;
first_par = OUTPUT_Table2_StructuralEstimation_endog2(i,1);
second_par = OUTPUT_Table2_StructuralEstimation_endog2(j,1);
f = x + y;
gradient_sigma = gradient(f, [x, y]);
gradient_sigma_est = subs(gradient_sigma, [x y], [first_par second_par]);
gradient_sigma_Matrix = double(gradient_sigma_est)';
SETetaDelta_endog2(index, :) = sqrt(diag(gradient_sigma_Matrix * VAR_Est_endog2([i j], [i j]) * gradient_sigma_Matrix'));


%% Calculation for B+Q2+Q3 combination

% teta(10)
index = 5;
i = 2; j = 6; k = 10;
first_par = OUTPUT_Table2_StructuralEstimation_endog2(i,1);
second_par = OUTPUT_Table2_StructuralEstimation_endog2(j,1);
third_par = OUTPUT_Table2_StructuralEstimation_endog2(k,1);
f = x + y + z;
gradient_sigma = gradient(f, [x, y, z]);
gradient_sigma_est = subs(gradient_sigma, [x y z], [first_par second_par third_par]);
gradient_sigma_Matrix = double(gradient_sigma_est)';
SETetaDelta_endog2(index, :) = sqrt(diag(gradient_sigma_Matrix * VAR_Est_endog2([i j k], [i j k]) * gradient_sigma_Matrix'));

% teta(12)
index = 6;
i = 3; j = 7; k = 12;
first_par = OUTPUT_Table2_StructuralEstimation_endog2(i,1);
second_par = OUTPUT_Table2_StructuralEstimation_endog2(j,1);
third_par = OUTPUT_Table2_StructuralEstimation_endog2(k,1);
f = x + y + z;
gradient_sigma = gradient(f, [x, y, z]);
gradient_sigma_est = subs(gradient_sigma, [x y z], [first_par second_par third_par]);
gradient_sigma_Matrix = double(gradient_sigma_est)';
SETetaDelta_endog2(index, :) = sqrt(diag(gradient_sigma_Matrix * VAR_Est_endog2([i j k], [i j k]) * gradient_sigma_Matrix'));

% teta(13)
index = 7;
j = 8; k = 13;
second_par = OUTPUT_Table2_StructuralEstimation_endog2(j,1);
third_par = OUTPUT_Table2_StructuralEstimation_endog2(k,1);
f = y + z;
gradient_sigma = gradient(f, [y, z]);
gradient_sigma_est = subs(gradient_sigma, [y z], [second_par third_par]);
gradient_sigma_Matrix = double(gradient_sigma_est)';
SETetaDelta_endog2(index, :) = sqrt(diag(gradient_sigma_Matrix * VAR_Est_endog2([j k], [j k]) * gradient_sigma_Matrix'));

% teta(15)
index = 8;
i = 4; j = 9; l = 15;
first_par = OUTPUT_Table2_StructuralEstimation_endog2(i,1);
second_par = OUTPUT_Table2_StructuralEstimation_endog2(j,1);
fourth_par = OUTPUT_Table2_StructuralEstimation_endog2(l,1);
f = x + y + w;
gradient_sigma = gradient(f, [x, y, w]);
gradient_sigma_est = subs(gradient_sigma, [x y w], [first_par second_par fourth_par]);
gradient_sigma_Matrix = double(gradient_sigma_est)';
SETetaDelta_endog2(index, :) = sqrt(diag(gradient_sigma_Matrix * VAR_Est_endog2([i j l], [i j l]) * gradient_sigma_Matrix'));

%% Calculation for B+Q2+Q3+Q4 combination

% teta(16)
index = 9;
i = 1; j = 5; l = 16;
first_par = OUTPUT_Table2_StructuralEstimation_endog2(i,1);
second_par = OUTPUT_Table2_StructuralEstimation_endog2(j,1);
fourth_par = OUTPUT_Table2_StructuralEstimation_endog2(l,1);
f = x + y + w;
gradient_sigma = gradient(f, [x, y, w]);
gradient_sigma_est = subs(gradient_sigma, [x y w], [first_par second_par fourth_par]);
gradient_sigma_Matrix = double(gradient_sigma_est)';
SETetaDelta_endog2(index, :) = sqrt(diag(gradient_sigma_Matrix * VAR_Est_endog2([i j l], [i j l]) * gradient_sigma_Matrix'));

% teta(18)
index = 10;
i = 3; j = 7; k = 12; l = 18;
first_par = OUTPUT_Table2_StructuralEstimation_endog2(i,1);
second_par = OUTPUT_Table2_StructuralEstimation_endog2(j,1);
third_par = OUTPUT_Table2_StructuralEstimation_endog2(k,1);
fourth_par = OUTPUT_Table2_StructuralEstimation_endog2(l,1);
f = x + y + z + w;
gradient_sigma = gradient(f, [x, y, z, w]);
gradient_sigma_est = subs(gradient_sigma, [x y z w], [first_par second_par third_par fourth_par]);
gradient_sigma_Matrix = double(gradient_sigma_est)';
SETetaDelta_endog2(index, :) = sqrt(diag(gradient_sigma_Matrix * VAR_Est_endog2([i j k l], [i j k l]) * gradient_sigma_Matrix'));

% teta(19)
index = 11;
j = 8; k = 13; l = 19;
second_par = OUTPUT_Table2_StructuralEstimation_endog2(j,1);
third_par = OUTPUT_Table2_StructuralEstimation_endog2(k,1);
fourth_par = OUTPUT_Table2_StructuralEstimation_endog2(l,1);
f = y + z + w;
gradient_sigma = gradient(f, [y, z, w]);
gradient_sigma_est = subs(gradient_sigma, [y z w], [second_par third_par fourth_par]);
gradient_sigma_Matrix = double(gradient_sigma_est)';
SETetaDelta_endog2(index, :) = sqrt(diag(gradient_sigma_Matrix * VAR_Est_endog2([j k l], [j k l]) * gradient_sigma_Matrix'));

% teta(20)
index = 12;
k = 14; l = 20;
third_par = OUTPUT_Table2_StructuralEstimation_endog2(k,1);
fourth_par = OUTPUT_Table2_StructuralEstimation_endog2(l,1);
f = z + w;
gradient_sigma = gradient(f, [z, w]);
gradient_sigma_est = subs(gradient_sigma, [z w], [third_par fourth_par]);
gradient_sigma_Matrix = double(gradient_sigma_est)';
SETetaDelta_endog2(index, :) = sqrt(diag(gradient_sigma_Matrix * VAR_Est_endog2([k l], [k l]) * gradient_sigma_Matrix'));

% teta(21)
index = 13;
i = 4; j = 9; k = 15; l = 21;
first_par = OUTPUT_Table2_StructuralEstimation_endog2(i,1);
second_par = OUTPUT_Table2_StructuralEstimation_endog2(j,1);
third_par = OUTPUT_Table2_StructuralEstimation_endog2(k,1);
fourth_par = OUTPUT_Table2_StructuralEstimation_endog2(l,1);
f = x + y + z + w;
gradient_sigma = gradient(f, [x, y, z, w]);
gradient_sigma_est = subs(gradient_sigma, [x y z w], [first_par second_par third_par fourth_par]);
gradient_sigma_Matrix = double(gradient_sigma_est)';
SETetaDelta_endog2(index, :) = sqrt(diag(gradient_sigma_Matrix * VAR_Est_endog2([i j k l], [i j k l]) * gradient_sigma_Matrix'));

%% Concatenate all standard errors into one matrix for output
SE_ANALYTIC_endog2 = [SE_Estimation_endog2; SETetaDelta_endog2];

% Organizing standard errors into a structured output matrix
OUTPUT_Table2_SE_Analytic_endog2 = [
    SE_ANALYTIC_endog2(1)  0                      0;
    SE_ANALYTIC_endog2(2)  SE_ANALYTIC_endog2(3)  0;  
    0                      0                      SE_ANALYTIC_endog2(4); 
    %
    SE_ANALYTIC_endog2(22) 0                      SE_ANALYTIC_endog2(8);
    SE_ANALYTIC_endog2(23) SE_ANALYTIC_endog2(24) 0;
    0                      0                      SE_ANALYTIC_endog2(25);
    %
    SE_ANALYTIC_endog2(22) 0                      SE_ANALYTIC_endog2(28);
    SE_ANALYTIC_endog2(26) SE_ANALYTIC_endog2(27) SE_ANALYTIC_endog2(14);
    SE_ANALYTIC_endog2(11) 0                      SE_ANALYTIC_endog2(29);
    %
    SE_ANALYTIC_endog2(30) SE_ANALYTIC_endog2(17) SE_ANALYTIC_endog2(32);
    SE_ANALYTIC_endog2(26) SE_ANALYTIC_endog2(31) SE_ANALYTIC_endog2(33);
    SE_ANALYTIC_endog2(11) 0                      SE_ANALYTIC_endog2(34);
];

% Define matrices for each regime
SVAR_1Regime_SE_endog2 = OUTPUT_Table2_SE_Analytic_endog2(1:3,:);
SVAR_2Regime_SE_endog2 = OUTPUT_Table2_SE_Analytic_endog2(4:6,:);
SVAR_3Regime_SE_endog2 = OUTPUT_Table2_SE_Analytic_endog2(7:9,:);
SVAR_4Regime_SE_endog2 = OUTPUT_Table2_SE_Analytic_endog2(10:12,:);

PVarl_LRTest_ex_end2 = 1-chi2cdf(-2 * (LK_Estimation_endog2-LK_Estimation), (StructuralParam_endog2 - StructuralParam));

%% --------------- UM12 and UF12 -------------------

% Data set
load DataSet12.txt

%% Data Set
DataSet = DataSet12(2:end,[1 3 2]);
AllDataSet_12=DataSet;
M=size(DataSet,2);

UM12=DataSet(:,1); % Macro Uncertainty variable
Y=DataSet(:,2); % Industrial Production variable
UF12=DataSet(:,3); % Financial Uncertainty variable

% Creates the data for the three regimes
DataSet_1Regime_12=DataSet(1:TB1,:); % First regime
DataSet_2Regime_12=DataSet(TB1+1-NLags:TB2,:); % Second regime
DataSet_3Regime_12=DataSet(TB2+1-NLags:TB3,:); % Third regime
DataSet_4Regime_12=DataSet(TB3+1-NLags:end,:); % Fourth regime

% Size three regimes
T1 = size(DataSet_1Regime_12,1)-NLags;
T2 = size(DataSet_2Regime_12,1)-NLags;
T3 = size(DataSet_3Regime_12,1)-NLags;
T4 = size(DataSet_4Regime_12,1)-NLags;
TAll = size(DataSet,1)-NLags;

%% Reduced form estimation

T=TAll;
VAR_Variables_X=[ones(size(DataSet(NLags:end-1,:),1),1) DataSet(NLags:end-1,:) DataSet(NLags-1:end-2,:) DataSet(NLags-2:end-3,:) DataSet(NLags-3:end-4,:)];
VAR_Variables_Y=DataSet(NLags+1:end,:);

Beta_OLS=(VAR_Variables_X'*VAR_Variables_X)^(-1)*VAR_Variables_X'*VAR_Variables_Y;

[Beta_LK_12,Log_LK,exitflag,~,grad,HESSIAN_LK] = fminunc('Likelihood_UNRESTRICTED',Beta_OLS',options2);

CommonPI_12=Beta_LK_12;
Errors_12=VAR_Variables_Y-VAR_Variables_X*Beta_LK_12';
Omega_LK_12=1/(T)*(Errors_12'*Errors_12);
LK_All_12=[-Log_LK];

% ******************************************************************************
% First Regime
% ******************************************************************************
T=T1;
DataSet=DataSet_1Regime_12;
VAR_Variables_X=[ones(size(DataSet(NLags:end-1,:),1),1) DataSet(NLags:end-1,:) DataSet(NLags-1:end-2,:) DataSet(NLags-2:end-3,:) DataSet(NLags-3:end-4,:)];
VAR_Variables_Y=DataSet(NLags+1:end,:);

Beta_OLS=(VAR_Variables_X'*VAR_Variables_X)^(-1)*VAR_Variables_X'*VAR_Variables_Y;

[Beta_LK1_12,Log_LK,exitflag,~,grad,HESSIAN_LK] = fminunc('Likelihood_UNRESTRICTED',Beta_OLS',options2);

Errors1_12=VAR_Variables_Y-VAR_Variables_X*Beta_LK1_12';
Omega_LK1_12=1/(T)*(Errors1_12'*Errors1_12);

% Companion matrix
CompanionMatrix_1Regime_12=[Beta_LK1_12(:,2:end);
    eye(M*(NLags-1),M*(NLags-1)) zeros(M*(NLags-1),M)];                            
                           

Sigma_1Regime=Omega_LK1_12;

% Likelihood of the VAR in the first regime
LK_1Regime_12=[-Log_LK];

% ******************************************************************************
% Second Regime
% ******************************************************************************

T=T2;
DataSet=DataSet_2Regime_12;
VAR_Variables_X=[ones(size(DataSet(NLags:end-1,:),1),1) DataSet(NLags:end-1,:) DataSet(NLags-1:end-2,:) DataSet(NLags-2:end-3,:) DataSet(NLags-3:end-4,:)];
VAR_Variables_Y=DataSet(NLags+1:end,:);

Beta_OLS=(VAR_Variables_X'*VAR_Variables_X)^(-1)*VAR_Variables_X'*VAR_Variables_Y;

[Beta_LK2_12,Log_LK,exitflag,~,grad,HESSIAN_LK] = fminunc('Likelihood_UNRESTRICTED',Beta_OLS',options2);

Errors2_12=VAR_Variables_Y-VAR_Variables_X*Beta_LK2_12';
Omega_LK2_12=1/(T)*(Errors2_12'*Errors2_12);

% Companion matrix
CompanionMatrix_2Regime_12=[Beta_LK2_12(:,2:end);
    eye(M*(NLags-1),M*(NLags-1)) zeros(M*(NLags-1),M)];                            

Sigma_2Regime=Omega_LK2_12;

% Likelihood of the VAR in the second regime
LK_2Regime_12=[-Log_LK];
          
% ******************************************************************************
% Third Regime
% ******************************************************************************

T=T3;
DataSet=DataSet_3Regime_12;
VAR_Variables_X=[ones(size(DataSet(NLags:end-1,:),1),1) DataSet(NLags:end-1,:) DataSet(NLags-1:end-2,:) DataSet(NLags-2:end-3,:) DataSet(NLags-3:end-4,:)];
VAR_Variables_Y=DataSet(NLags+1:end,:);

Beta_OLS=(VAR_Variables_X'*VAR_Variables_X)^(-1)*VAR_Variables_X'*VAR_Variables_Y;

[Beta_LK3_12,Log_LK,exitflag,~,grad,HESSIAN_LK] = fminunc('Likelihood_UNRESTRICTED',Beta_OLS',options2);

Errors3_12=VAR_Variables_Y-VAR_Variables_X*Beta_LK3_12';
Omega_LK3_12=1/(T)*(Errors3_12'*Errors3_12);

% Companion matrix
CompanionMatrix_3Regime_12=[Beta_LK3_12(:,2:end);
    eye(M*(NLags-1),M*(NLags-1)) zeros(M*(NLags-1),M)];                            

Sigma_3Regime=Omega_LK3_12;

% Likelihood of the VAR in the third regime  
LK_3Regime_12=[-Log_LK];

% ******************************************************************************
% Fourth Regime
% ******************************************************************************

T=T4;
DataSet=DataSet_4Regime_12;
VAR_Variables_X=[ones(size(DataSet(NLags:end-1,:),1),1) DataSet(NLags:end-1,:) DataSet(NLags-1:end-2,:) DataSet(NLags-2:end-3,:) DataSet(NLags-3:end-4,:)];
VAR_Variables_Y=DataSet(NLags+1:end,:);

Beta_OLS=(VAR_Variables_X'*VAR_Variables_X)^(-1)*VAR_Variables_X'*VAR_Variables_Y;

[Beta_LK4_12,Log_LK,exitflag,~,grad,HESSIAN_LK] = fminunc('Likelihood_UNRESTRICTED',Beta_OLS',options2);

Errors4_12=VAR_Variables_Y-VAR_Variables_X*Beta_LK4_12';

Omega_LK4_12=1/(T)*(Errors4_12'*Errors4_12);

% Companion matrix
CompanionMatrix_4Regime_12=[Beta_LK4_12(:,2:end);
    eye(M*(NLags-1),M*(NLags-1)) zeros(M*(NLags-1),M)];                            
                                       
Sigma_4Regime=Omega_LK4_12;

% Likelihood of the VAR in the fourth regime  
LK_4Regime_12=[-Log_LK];

%% Estimation of the structural parameters

% ML function
[StructuralParam_Estiamtion_MATRIX,Likelihood_MATRIX,exitflag,output,grad,Hessian_MATRIX] = fminunc('Likelihood_SVAR_Restricted',InitialValue_SVAR_Initial',options);

StructuralParam_Estiamtion_12=StructuralParam_Estiamtion_MATRIX;
LK_Estimation_12=Likelihood_MATRIX;
Hessian_Estimation_12=Hessian_MATRIX;
SE_Estimation_12=diag(Hessian_Estimation^(-1)).^0.5;

%% Overidentification LR test
PVarl_LRTest_12 = 1-chi2cdf(2 * ((LK_1Regime_12(1)+LK_2Regime_12(1)+LK_3Regime_12(1)+LK_4Regime_12(1))+LK_Estimation_12),24-StructuralParam);

%% Here below we define the matrices of the structural parameters with restrictions on the coefficients as described in the paper. SVAR_C corresponds to the B matrix in the paper, SVAR_Q2 corresponds to
% the Q2 matrix of the paper, SVAR_Q3 corresponds to the Q3 matrix of the paper and SVAR_Q4 corresponds to the Q4 matrix of the paper.

SVAR_C_12=[StructuralParam_Estiamtion_12(1) 0                                0;
           StructuralParam_Estiamtion_12(2) StructuralParam_Estiamtion_12(3) 0;
           0                                0                                StructuralParam_Estiamtion_12(4)];

SVAR_Q2_12=[StructuralParam_Estiamtion_12(5)  0                                StructuralParam_Estiamtion_12(8);
            StructuralParam_Estiamtion_12(6)  StructuralParam_Estiamtion_12(7) 0;
            0                                 0                                StructuralParam_Estiamtion_12(9)];

SVAR_Q3_12=[0                                 0                                 StructuralParam_Estiamtion_12(12);
            StructuralParam_Estiamtion_12(10) StructuralParam_Estiamtion_12(11) StructuralParam_Estiamtion_12(13);
            0                                 0                                 StructuralParam_Estiamtion_12(14)];

SVAR_Q4_12=[StructuralParam_Estiamtion_12(15) 0                                 StructuralParam_Estiamtion_12(17);
            0                                 StructuralParam_Estiamtion_12(16) StructuralParam_Estiamtion_12(18);
            0                                 0                                 StructuralParam_Estiamtion_12(19)];  

SVAR_1Regime_12=SVAR_C_12; % B
SVAR_2Regime_12=SVAR_C_12+SVAR_Q2_12;   % B+Q2
SVAR_3Regime_12=SVAR_C_12+SVAR_Q2_12+SVAR_Q3_12;  % B+Q2+Q3
SVAR_4Regime_12=SVAR_C_12+SVAR_Q2_12+SVAR_Q3_12+SVAR_Q4_12;  % B+Q2+Q3+Q4

% Flip the sign if the parameter on the main diagonal is negative

	if SVAR_1Regime_12(1,1)<0
    SVAR_1Regime_12(:,1)=-SVAR_1Regime_12(:,1);
    end
    if SVAR_1Regime_12(2,2)<0
    SVAR_1Regime_12(:,2)=-SVAR_1Regime_12(:,2); 
    end
    if SVAR_1Regime_12(3,3)<0
    SVAR_1Regime_12(:,3)=-SVAR_1Regime_12(:,3);
    end
    
	if SVAR_2Regime_12(1,1)<0
    SVAR_2Regime_12(:,1)=-SVAR_2Regime_12(:,1);
    end
    if SVAR_2Regime_12(2,2)<0
    SVAR_2Regime_12(:,2)=-SVAR_2Regime_12(:,2); 
    end
    if SVAR_2Regime_12(3,3)<0
    SVAR_2Regime_12(:,3)=-SVAR_2Regime_12(:,3);
     end
    
    
    if SVAR_3Regime_12(1,1)<0
    SVAR_3Regime_12(:,1)=-SVAR_3Regime_12(:,1);
    end
    if SVAR_3Regime_12(2,2)<0
    SVAR_3Regime_12(:,2)=-SVAR_3Regime_12(:,2); 
    end
    if SVAR_3Regime_12(3,3)<0
    SVAR_3Regime_12(:,3)=-SVAR_3Regime_12(:,3);
    end
    

    if SVAR_4Regime_12(1,1)<0
    SVAR_4Regime_12(:,1)=-SVAR_4Regime_12(:,1);
    end
    if SVAR_4Regime_12(2,2)<0
    SVAR_4Regime_12(:,2)=-SVAR_4Regime_12(:,2); 
    end
    if SVAR_4Regime_12(3,3)<0
    SVAR_4Regime_12(:,3)=-SVAR_4Regime_12(:,3);
    end
    
     
MATRICES=[SVAR_1Regime_12;
          SVAR_2Regime_12;
          SVAR_3Regime_12;
          SVAR_4Regime_12];
   
% Calculates the analytical derivatives organized in block matrices      
V11=2*NMatrix*kron(SVAR_C_12,eye(M));
V21=2*NMatrix*kron(SVAR_C_12,eye(M))+kron(SVAR_Q2_12,eye(M))+kron(eye(M),SVAR_Q2_12)*KommutationMatrix;
V22=kron(eye(M),SVAR_C_12)*KommutationMatrix+kron(SVAR_C_12,eye(M))+2*NMatrix*kron(SVAR_Q2_12,eye(M));
V31=2*NMatrix*kron(SVAR_C_12,eye(M))+kron(SVAR_Q2_12,eye(M))+kron(SVAR_Q3_12,eye(M))+kron(eye(M),SVAR_Q2_12)*KommutationMatrix+kron(eye(M),SVAR_Q3_12)*KommutationMatrix;
V32=kron(eye(M),SVAR_C_12)*KommutationMatrix+kron(SVAR_C_12,eye(M))+2*NMatrix*kron(SVAR_Q2_12,eye(M))+kron(SVAR_Q3_12,eye(M))+kron(eye(M),SVAR_Q3_12)*KommutationMatrix;
V33=kron(eye(M),SVAR_C_12)*KommutationMatrix+kron(eye(M),SVAR_Q2_12)*KommutationMatrix+2*NMatrix*kron(SVAR_Q3_12,eye(M))+kron(SVAR_C_12,eye(M))+kron(SVAR_Q2_12,eye(M));
V41 = 2 * NMatrix * kron(SVAR_C_12, eye(M)) + kron(SVAR_Q2_12, eye(M)) + kron(SVAR_Q3_12, eye(M)) + kron(SVAR_Q4_12, eye(M)) + kron(eye(M), SVAR_Q2_12) * KommutationMatrix + kron(eye(M), SVAR_Q3_12) * KommutationMatrix + kron(eye(M), SVAR_Q4_12) * KommutationMatrix;
V42 = kron(eye(M), SVAR_C_12) * KommutationMatrix + kron(SVAR_C_12, eye(M)) + 2 * NMatrix * kron(SVAR_Q2_12, eye(M)) + kron(SVAR_Q3_12, eye(M)) + kron(SVAR_Q4_12, eye(M)) + kron(eye(M), SVAR_Q3_12) * KommutationMatrix + kron(eye(M), SVAR_Q4_12) * KommutationMatrix;
V43 = kron(eye(M), SVAR_C_12) * KommutationMatrix + kron(eye(M), SVAR_Q2_12) * KommutationMatrix + 2 * NMatrix * kron(SVAR_Q3_12, eye(M)) + kron(SVAR_C_12, eye(M)) + kron(SVAR_Q2_12, eye(M)) + kron(SVAR_Q4_12, eye(M)) + kron(eye(M), SVAR_Q4_12) * KommutationMatrix;
V44 = kron(eye(M), SVAR_C_12) * KommutationMatrix + kron(eye(M), SVAR_Q2_12) * KommutationMatrix + kron(eye(M), SVAR_Q3_12) * KommutationMatrix + 2 * NMatrix * kron(SVAR_Q4_12, eye(M)) + kron(SVAR_C_12, eye(M)) + kron(SVAR_Q2_12, eye(M)) + kron(SVAR_Q3_12, eye(M));

% Calculates the matrix for checking the rank condition (full column rank)
RankMatrix = kron(eye(4), mDD) * [V11 zeros(M^2, M^2) zeros(M^2, M^2) zeros(M^2, M^2);
                                  V21 V22 zeros(M^2, M^2) zeros(M^2, M^2);
                                  V31 V32 V33 zeros(M^2, M^2);
                                  V41 V42 V43 V44];

 
% Selection matrix for extracting the structural parameters                         
HSelection=zeros(M*M*4,StructuralParam);
HSelection(1,1)=1;
HSelection(2,2)=1;
HSelection(5,3)=1;
HSelection(9,4)=1;
HSelection(10,5)=1;
HSelection(11,6)=1;
HSelection(14,7)=1;
HSelection(16,8)=1;
HSelection(18,9)=1;
HSelection(20,10)=1;
HSelection(23,11)=1;
HSelection(25,12)=1;
HSelection(26,13)=1;
HSelection(27,14)=1;
HSelection(28, 15) = 1;
HSelection(32, 16) = 1;  
HSelection(34, 17) = 1;  
HSelection(35, 18) = 1;  
HSelection(36, 19) = 1;  


Jacobian= RankMatrix*HSelection;

% Report the rank of the matrix for checking the identification
Jacobian_rank_12 = rank(Jacobian);

%% ********************** BOOTSTRAP **********************

Const_12 = Beta_LK_12(:, 1);
mP1_12 = Beta_LK_12(:, [2 3 4]);
mP2_12 = Beta_LK_12(:, [5 6 7]);
mP3_12 = Beta_LK_12(:, [8 9 10]);
mP4_12 = Beta_LK_12(:, [11 12 13]);

Const_1_12 = Beta_LK1_12(:, 1);
mP1_1_12 = Beta_LK1_12(:, [2 3 4]);
mP2_1_12 = Beta_LK1_12(:, [5 6 7]);
mP3_1_12 = Beta_LK1_12(:, [8 9 10]);
mP4_1_12 = Beta_LK1_12(:, [11 12 13]);

Const_2_12 = Beta_LK2_12(:, 1);
mP1_2_12 = Beta_LK2_12(:, [2 3 4]);
mP2_2_12 = Beta_LK2_12(:, [5 6 7]);
mP3_2_12 = Beta_LK2_12(:, [8 9 10]);
mP4_2_12 = Beta_LK2_12(:, [11 12 13]);

Const_3_12 = Beta_LK3_12(:, 1);
mP1_3_12 = Beta_LK3_12(:, [2 3 4]);
mP2_3_12 = Beta_LK3_12(:, [5 6 7]);
mP3_3_12 = Beta_LK3_12(:, [8 9 10]);
mP4_3_12 = Beta_LK3_12(:, [11 12 13]);

Const_4_12 = Beta_LK4_12(:, 1);
mP1_4_12 = Beta_LK4_12(:, [2 3 4]);
mP2_4_12 = Beta_LK4_12(:, [5 6 7]);
mP3_4_12 = Beta_LK4_12(:, [8 9 10]);
mP4_4_12 = Beta_LK4_12(:, [11 12 13]);

wbGS = waitbar(0,'Running the bootstrap for 12 months uncertainties');
for boot = 1 : BootstrapIterations
    waitbar(boot/BootstrapIterations, wbGS, sprintf('Running the bootstrap for 12 months uncertainties :  Processing %d of %d', boot,BootstrapIterations)) 
    
    
     %  **** iid bootstrap ****
    %  **** First regime ***************

    TBoot = datasample(1:T1,T1); 
    Residuals_Boot=Errors1_12(TBoot,:); % bootstrap errors 

    DataSet_Bootstrap=zeros(T1+ NLags,M);
    DataSet_Bootstrap(1:NLags,:)=AllDataSet_12(1:NLags,:); % set the first NLags elements equal to the original sample values

        for t = 1+NLags : T1+NLags
        DataSet_Bootstrap(t,:)=Const_1_12 + mP1_1_12 * DataSet_Bootstrap(t-1,:)' +...
                                       mP2_1_12 * DataSet_Bootstrap(t-2,:)' + ...
                                       mP3_1_12 * DataSet_Bootstrap(t-3,:)' + ...
                                       mP4_1_12 * DataSet_Bootstrap(t-4,:)' + ...
                                       Residuals_Boot(t-NLags,:)';
        end

    DataSet=DataSet_Bootstrap;

    VAR_Variables_X=[ones(size(DataSet(NLags:end-1,:),1),1) DataSet(NLags:end-1,:) DataSet(NLags-1:end-2,:) DataSet(NLags-2:end-3,:) DataSet(NLags-3:end-4,:)];
    VAR_Variables_Y=DataSet(NLags+1:end,:);
    T=T1;
    Beta_OLS=(VAR_Variables_X'*VAR_Variables_X)^(-1)*VAR_Variables_X'*VAR_Variables_Y;
    
    [Beta_LK1_Boot,Log_LK_Boot,exitflag,output,grad,HESSIAN_LK_Boot] = fminunc('Likelihood_UNRESTRICTED',Beta_OLS',options2);
  
    Beta_LK1_Boot;
    mP1_Boot_1 = Beta_LK1_Boot(:, [2 3 4]);
    mP2_Boot_1 = Beta_LK1_Boot(:, [5 6 7]);
    mP3_Boot_1 = Beta_LK1_Boot(:, [8 9 10]);
    mP4_Boot_1 = Beta_LK1_Boot(:, [11 12 13]);
    LK_1Regime_Boot = -Log_LK_Boot;

    Errors1_Boot=VAR_Variables_Y-VAR_Variables_X*Beta_LK1_Boot';
    Sigma_1Regime_Boot=1/(T1)*(Errors1_Boot'*Errors1_Boot);


     %  **** iid bootstrap ****
    %  **** Second regime ***************
    
    
    TBoot = datasample(1:T2,T2); 
    Residuals_Boot=Errors2_12(TBoot,:); % bootstrap errors 

    DataSet_Bootstrap=zeros(T2+ NLags,M);
    DataSet_Bootstrap(1:NLags,:)=AllDataSet_12(1:NLags,:); % set the first NLags elements equal to the original sample values

        for t = 1+NLags : T2+NLags
        DataSet_Bootstrap(t,:)=Const_2_12 + mP1_2_12 * DataSet_Bootstrap(t-1,:)' +...
                                       mP2_2_12 * DataSet_Bootstrap(t-2,:)' + ...
                                       mP3_2_12 * DataSet_Bootstrap(t-3,:)' + ...
                                       mP4_2_12 * DataSet_Bootstrap(t-4,:)' + ...
                                       Residuals_Boot(t-NLags,:)';
        end

    DataSet=DataSet_Bootstrap;

    VAR_Variables_X=[ones(size(DataSet(NLags:end-1,:),1),1) DataSet(NLags:end-1,:) DataSet(NLags-1:end-2,:) DataSet(NLags-2:end-3,:) DataSet(NLags-3:end-4,:)];
    VAR_Variables_Y=DataSet(NLags+1:end,:);
    
    T=T2;
    Beta_OLS=(VAR_Variables_X'*VAR_Variables_X)^(-1)*VAR_Variables_X'*VAR_Variables_Y;
    
    [Beta_LK2_Boot,Log_LK_Boot,exitflag,output,grad,HESSIAN_LK_Boot] = fminunc('Likelihood_UNRESTRICTED',Beta_OLS',options2);
  
    Beta_LK2_Boot;
    mP1_Boot_2 = Beta_LK2_Boot(:, [2 3 4]);
    mP2_Boot_2 = Beta_LK2_Boot(:, [5 6 7]);
    mP3_Boot_2 = Beta_LK2_Boot(:, [8 9 10]);
    mP4_Boot_2 = Beta_LK2_Boot(:, [11 12 13]);
    LK_2Regime_Boot = -Log_LK_Boot;

    Errors2_Boot=VAR_Variables_Y-VAR_Variables_X*Beta_LK2_Boot';
    Sigma_2Regime_Boot=1/(T2)*(Errors2_Boot'*Errors2_Boot);


         %  **** iid bootstrap ****
    %  **** Third regime ***************
    
    
    TBoot = datasample(1:T3,T3); 
    Residuals_Boot=Errors3_12(TBoot,:); % bootstrap errors 

    DataSet_Bootstrap=zeros(T3+ NLags,M);
    DataSet_Bootstrap(1:NLags,:)=AllDataSet_12(1:NLags,:); % set the first NLags elements equal to the original sample values

        for t = 1+NLags : T3+NLags
        DataSet_Bootstrap(t,:)=Const_3_12 + mP1_3_12 * DataSet_Bootstrap(t-1,:)' +...
                                       mP2_3_12 * DataSet_Bootstrap(t-2,:)' + ...
                                       mP3_3_12 * DataSet_Bootstrap(t-3,:)' + ...
                                       mP4_3_12 * DataSet_Bootstrap(t-4,:)' + ...
                                       Residuals_Boot(t-NLags,:)';
        end

    DataSet=DataSet_Bootstrap;

    VAR_Variables_X=[ones(size(DataSet(NLags:end-1,:),1),1) DataSet(NLags:end-1,:) DataSet(NLags-1:end-2,:) DataSet(NLags-2:end-3,:) DataSet(NLags-3:end-4,:)];
    VAR_Variables_Y=DataSet(NLags+1:end,:);
    T=T3;
    Beta_OLS=(VAR_Variables_X'*VAR_Variables_X)^(-1)*VAR_Variables_X'*VAR_Variables_Y;
    
    [Beta_LK3_Boot,Log_LK_Boot,exitflag,output,grad,HESSIAN_LK_Boot] = fminunc('Likelihood_UNRESTRICTED',Beta_OLS',options2);
  
    Beta_LK3_Boot;
    mP1_Boot_3 = Beta_LK3_Boot(:, [2 3 4]);
    mP2_Boot_3 = Beta_LK3_Boot(:, [5 6 7]);
    mP3_Boot_3 = Beta_LK3_Boot(:, [8 9 10]);
    mP4_Boot_3 = Beta_LK3_Boot(:, [11 12 13]);
    LK_3Regime_Boot = -Log_LK_Boot;

    Errors3_Boot=VAR_Variables_Y-VAR_Variables_X*Beta_LK3_Boot';
    Sigma_3Regime_Boot=1/(T3)*(Errors3_Boot'*Errors3_Boot);

    
    
    %  **** iid bootstrap ****
    %  **** Fourth regime ***************
    
    
    TBoot = datasample(1:T4,T4); 
    Residuals_Boot=Errors4_12(TBoot,:); % bootstrap errors 

    DataSet_Bootstrap=zeros(T4+ NLags,M);
    DataSet_Bootstrap(1:NLags,:)=AllDataSet_12(1:NLags,:); % set the first NLags elements equal to the original sample values

        for t = 1+NLags : T4+NLags
        DataSet_Bootstrap(t,:)=Const_4_12 + mP1_4_12 * DataSet_Bootstrap(t-1,:)' +...
                                       mP2_4_12 * DataSet_Bootstrap(t-2,:)' + ...
                                       mP3_4_12 * DataSet_Bootstrap(t-3,:)' + ...
                                       mP4_4_12 * DataSet_Bootstrap(t-4,:)' + ...
                                       Residuals_Boot(t-NLags,:)';
        end

    DataSet=DataSet_Bootstrap;

    VAR_Variables_X=[ones(size(DataSet(NLags:end-1,:),1),1) DataSet(NLags:end-1,:) DataSet(NLags-1:end-2,:) DataSet(NLags-2:end-3,:) DataSet(NLags-3:end-4,:)];
    VAR_Variables_Y=DataSet(NLags+1:end,:);
    T=T4;
    Beta_OLS=(VAR_Variables_X'*VAR_Variables_X)^(-1)*VAR_Variables_X'*VAR_Variables_Y;
    
    [Beta_LK4_Boot,Log_LK_Boot,exitflag,output,grad,HESSIAN_LK_Boot] = fminunc('Likelihood_UNRESTRICTED',Beta_OLS',options2);
  
    Beta_LK4_Boot;
    mP1_Boot_4 = Beta_LK4_Boot(:, [2 3 4]);
    mP2_Boot_4 = Beta_LK4_Boot(:, [5 6 7]);
    mP3_Boot_4 = Beta_LK4_Boot(:, [8 9 10]);
    mP4_Boot_4 = Beta_LK4_Boot(:, [11 12 13]);
    LK_4Regime_Boot = -Log_LK_Boot;

    Errors4_Boot=VAR_Variables_Y-VAR_Variables_X*Beta_LK4_Boot';
    Sigma_4Regime_Boot=1/(T4)*(Errors4_Boot'*Errors4_Boot);


    % ********* estimating bootstrapped IRFs *********
                                                                                                                                
    [StructuralParam_Estimation_Boot,Likelihood_SVAR_Boot,exitflag,output,grad,Hessian_MATRIX_Boot] = fminunc('Likelihood_SVAR_Restricted_Boot',StructuralEstimationCorrected',options);
 

    SVAR_C_Boot=[StructuralParam_Estimation_Boot(1) 0                                  0;
                 StructuralParam_Estimation_Boot(2) StructuralParam_Estimation_Boot(3) 0;
                 0                                  0                                  StructuralParam_Estimation_Boot(4)];

    SVAR_Q2_Boot=[StructuralParam_Estimation_Boot(5)  0                                  StructuralParam_Estimation_Boot(8);
                  StructuralParam_Estimation_Boot(6)  StructuralParam_Estimation_Boot(7) 0;
                  0                                   0                                  StructuralParam_Estimation_Boot(9)];

    SVAR_Q3_Boot=[0                                   0                                   StructuralParam_Estimation_Boot(12);
                  StructuralParam_Estimation_Boot(10) StructuralParam_Estimation_Boot(11) StructuralParam_Estimation_Boot(13);
                  0                                   0                                   StructuralParam_Estimation_Boot(14)];

    SVAR_Q4_Boot=[StructuralParam_Estimation_Boot(15) 0                                   StructuralParam_Estimation_Boot(17);
                  0                                   StructuralParam_Estimation_Boot(16) StructuralParam_Estimation_Boot(18);
                  0                                   0                                   StructuralParam_Estimation_Boot(19)];  

    SVAR_1Regime_Boot=SVAR_C_Boot; % B
    SVAR_2Regime_Boot=SVAR_C_Boot+SVAR_Q2_Boot;   % B+Q2
    SVAR_3Regime_Boot=SVAR_C_Boot+SVAR_Q2_Boot+SVAR_Q3_Boot;  % B+Q2+Q3
    SVAR_4Regime_Boot=SVAR_C_Boot+SVAR_Q2_Boot+SVAR_Q3_Boot+SVAR_Q4_Boot;  % B+Q2+Q3+Q4

    % Flip the sign if the parameter on the main diagonal is negative

	if SVAR_1Regime_Boot(1,1)<0
    SVAR_1Regime_Boot(:,1)=-SVAR_1Regime_Boot(:,1);
    end
    if SVAR_1Regime_Boot(2,2)<0
    SVAR_1Regime_Boot(:,2)=-SVAR_1Regime_Boot(:,2); 
    end
    if SVAR_1Regime_Boot(3,3)<0
    SVAR_1Regime_Boot(:,3)=-SVAR_1Regime_Boot(:,3);
    end
    
	if SVAR_2Regime_Boot(1,1)<0
    SVAR_2Regime_Boot(:,1)=-SVAR_2Regime_Boot(:,1);
    end
    if SVAR_2Regime_Boot(2,2)<0
    SVAR_2Regime_Boot(:,2)=-SVAR_2Regime_Boot(:,2); 
    end
    if SVAR_2Regime_Boot(3,3)<0
    SVAR_2Regime_Boot(:,3)=-SVAR_2Regime_Boot(:,3);
     end
    
    
    if SVAR_3Regime_Boot(1,1)<0
    SVAR_3Regime_Boot(:,1)=-SVAR_3Regime_Boot(:,1);
    end
    if SVAR_3Regime_Boot(2,2)<0
    SVAR_3Regime_Boot(:,2)=-SVAR_3Regime_Boot(:,2); 
    end
    if SVAR_3Regime_Boot(3,3)<0
    SVAR_3Regime_Boot(:,3)=-SVAR_3Regime_Boot(:,3);
    end
    

    if SVAR_4Regime_Boot(1,1)<0
    SVAR_4Regime_Boot(:,1)=-SVAR_4Regime_Boot(:,1);
    end
    if SVAR_4Regime_Boot(2,2)<0
    SVAR_4Regime_Boot(:,2)=-SVAR_4Regime_Boot(:,2); 
    end
    if SVAR_4Regime_Boot(3,3)<0
    SVAR_4Regime_Boot(:,3)=-SVAR_4Regime_Boot(:,3);
    end
    
    
    J=[eye(M) zeros(M,M*(NLags-1))]; 

    CompanionMatrix_Boot_1=[Beta_LK1_Boot(:,2:end);
    eye(M*(NLags-1),M*(NLags-1)) zeros(M*(NLags-1),M)]; 
    
    CompanionMatrix_Boot_2=[Beta_LK2_Boot(:,2:end);
    eye(M*(NLags-1),M*(NLags-1)) zeros(M*(NLags-1),M)]; 

    CompanionMatrix_Boot_3=[Beta_LK3_Boot(:,2:end);
    eye(M*(NLags-1),M*(NLags-1)) zeros(M*(NLags-1),M)]; 

    CompanionMatrix_Boot_4=[Beta_LK4_Boot(:,2:end);
    eye(M*(NLags-1),M*(NLags-1)) zeros(M*(NLags-1),M)]; 

    for h = 0 : HorizonIRF
    TETA_Boot1(:,:,h+1,boot)=J*CompanionMatrix_Boot_1^h*J'*SVAR_1Regime_Boot;
    end    

    for h = 0 : HorizonIRF
    TETA_Boot2(:,:,h+1,boot)=J*CompanionMatrix_Boot_2^h*J'*SVAR_2Regime_Boot;
    end 

    for h = 0 : HorizonIRF
    TETA_Boot3(:,:,h+1,boot)=J*CompanionMatrix_Boot_3^h*J'*SVAR_3Regime_Boot;
    end 

    for h = 0 : HorizonIRF
    TETA_Boot4(:,:,h+1,boot)=J*CompanionMatrix_Boot_4^h*J'*SVAR_4Regime_Boot;
    end 


end 
delete(wbGS);

IRF_Inf_Boot1_12 = prctile(TETA_Boot1,quant(1),4);
IRF_Sup_Boot1_12 = prctile(TETA_Boot1,quant(2),4);

IRF_Inf_Boot2_12 = prctile(TETA_Boot2,quant(1),4);
IRF_Sup_Boot2_12 = prctile(TETA_Boot2,quant(2),4);

IRF_Inf_Boot3_12 = prctile(TETA_Boot3,quant(1),4);
IRF_Sup_Boot3_12 = prctile(TETA_Boot3,quant(2),4);

IRF_Inf_Boot4_12 = prctile(TETA_Boot4,quant(1),4);
IRF_Sup_Boot4_12 = prctile(TETA_Boot4,quant(2),4);

%% IRF Q1

C_IRF = SVAR_1Regime;  % instantaneous impact at h=0
C_IRF_12 = SVAR_1Regime_12;

% Compute impulse responses over the horizon
HorizonIRF = 60;  % Define the response horizon
for h = 0 : HorizonIRF
    TETA_1(:,:,h+1) = J * CompanionMatrix_1Regime^h * J' * C_IRF;
end

for h = 0 : HorizonIRF
    TETA_1_12(:,:,h+1) = J * CompanionMatrix_1Regime_12^h * J' * C_IRF_12;
end

% Define titles for each shock scenario
Titles = cell(1,3);
Titles{1,1} = '$$\varepsilon_{UM_t}$$ Shock';
Titles{1,2} = '$$\varepsilon_{y_t}$$ Shock';
Titles{1,3} = '$$\varepsilon_{UF_t}$$ Shock';

% Define Y-axis labels for each variable
YLabel = cell(3,1);
YLabel{1,1} = '$$UM_t$$';
YLabel{2,1} = '$$y_t$$';
YLabel{3,1} = '$$UF_t$$';

M_IRF = 3;  
figure(11);
index = 1;Shock_1=[1 1 1];

% Loop through each variable and shock for plotting
for jr = 1 : M_IRF
    for jc = 1 : M_IRF
        TETA_Iter_Sample = squeeze(TETA_1(jr,jc,:));
        TETA_Iter_Sample_12 = squeeze(TETA_1_12(jr,jc,:));
        TETA_Iter_Boot_Inf = squeeze(IRF_Inf_Boot1(jr,jc,:));
        TETA_Iter_Boot_Sup = squeeze(IRF_Sup_Boot1(jr,jc,:));
        TETA_Iter_Boot_Inf_12 = squeeze(IRF_Inf_Boot1_12(jr,jc,:));
        TETA_Iter_Boot_Sup_12 = squeeze(IRF_Sup_Boot1_12(jr,jc,:));
        subplot(M_IRF, M_IRF, index);
        x = 1:1:HorizonIRF+1;  
        

        % Plot the IRF
        y = TETA_Iter_Sample' * Shock_1(jr);  
        y12 = TETA_Iter_Sample_12' * Shock_1(jr);  
        plot(x, y, 'Color', [0 0.4470 0.7410], 'LineWidth', LineWidth_IRF);
        hold on;
        plot(x, y12, 'Color', [0.8500, 0.3250, 0.0980], 'LineWidth', LineWidth_IRF);

        % Add shaded confidence interval 
        fill([x fliplr(x)], [TETA_Iter_Boot_Inf' fliplr(TETA_Iter_Boot_Sup')], ...
        [0 0.4470 0.7410], 'EdgeColor', 'none', 'FaceAlpha', 0.2);  
        fill([x fliplr(x)], [TETA_Iter_Boot_Inf_12' fliplr(TETA_Iter_Boot_Sup_12')], ...
        [0.8500, 0.3250, 0.0980], 'EdgeColor', 'none', 'FaceAlpha', 0.2);

        % Add a zero line for reference
        plot(x, zeros(HorizonIRF+1, 1), 'k', 'LineWidth', 1);

        % Set labels and titles
        ylabel(YLabel{jr, 1}, 'interpreter', 'latex');
        title(Titles{1, jc}, 'interpreter', 'latex');
        set(gca, 'FontSize', FontSizeIRFGraph);
        set(gcf, 'Color', 'w'); 
        set(gca, 'Color', 'w');
        axis tight;  
        index = index + 1;
    end
end


%% IRF Q2

C_IRF = SVAR_2Regime;  % instantaneous impact at h=0
C_IRF_12 = SVAR_2Regime_12;

% Compute impulse responses over the horizon
HorizonIRF = 60;  % Define the response horizon
for h = 0 : HorizonIRF
    TETA_2(:,:,h+1) = J * CompanionMatrix_2Regime^h * J' * C_IRF;
end

for h = 0 : HorizonIRF
    TETA_2_12(:,:,h+1) = J * CompanionMatrix_2Regime_12^h * J' * C_IRF_12;
end

% Define titles for each shock scenario
Titles = cell(1,3);
Titles{1,1} = '$$\varepsilon_{UM_t}$$ Shock';
Titles{1,2} = '$$\varepsilon_{y_t}$$ Shock';
Titles{1,3} = '$$\varepsilon_{UF_t}$$ Shock';

% Define Y-axis labels for each variable
YLabel = cell(3,1);
YLabel{1,1} = '$$UM_t$$';
YLabel{2,1} = '$$y_t$$';
YLabel{3,1} = '$$UF_t$$';

M_IRF = 3;  
figure(12);
index = 1;
Shock_1=[1 1 1];

% Loop through each variable and shock for plotting
for jr = 1 : M_IRF
    for jc = 1 : M_IRF
        TETA_Iter_Sample = squeeze(TETA_2(jr,jc,:));
        TETA_Iter_Sample_12 = squeeze(TETA_2_12(jr,jc,:));
        TETA_Iter_Boot_Inf = squeeze(IRF_Inf_Boot2(jr,jc,:));
        TETA_Iter_Boot_Sup = squeeze(IRF_Sup_Boot2(jr,jc,:));
        TETA_Iter_Boot_Inf_12 = squeeze(IRF_Inf_Boot2_12(jr,jc,:));
        TETA_Iter_Boot_Sup_12 = squeeze(IRF_Sup_Boot2_12(jr,jc,:));
        subplot(M_IRF, M_IRF, index);
        x = 1:1:HorizonIRF+1;  

        % Plot the IRF
        y = TETA_Iter_Sample' * Shock_1(jr);  
        y12 = TETA_Iter_Sample_12' * Shock_1(jr);  
        plot(x, y, 'Color', [0 0.4470 0.7410], 'LineWidth', LineWidth_IRF);
        hold on;
        plot(x, y12, 'Color', [0.8500, 0.3250, 0.0980], 'LineWidth', LineWidth_IRF);

        % Add shaded confidence interval
        fill([x fliplr(x)], [TETA_Iter_Boot_Inf' fliplr(TETA_Iter_Boot_Sup')], ...
        [0 0.4470 0.7410], 'EdgeColor', 'none', 'FaceAlpha', 0.2);  
        fill([x fliplr(x)], [TETA_Iter_Boot_Inf_12' fliplr(TETA_Iter_Boot_Sup_12')], ...
        [0.8500, 0.3250, 0.0980], 'EdgeColor', 'none', 'FaceAlpha', 0.2);

        % Add a zero line for reference
        plot(x, zeros(HorizonIRF+1, 1), 'k', 'LineWidth', 1);

        % Set labels and titles
        ylabel(YLabel{jr, 1}, 'interpreter', 'latex');
        title(Titles{1, jc}, 'interpreter', 'latex');
        set(gca, 'FontSize', FontSizeIRFGraph);
        set(gcf, 'Color', 'w'); 
        set(gca, 'Color', 'w');
        axis tight;  
        index = index + 1;
    end
end


%% IRF Q3

C_IRF = SVAR_3Regime;  % instantaneous impact at h=0
C_IRF_12 = SVAR_3Regime_12;

% Compute impulse responses over the horizon
HorizonIRF = 60;  % Define the response horizon
for h = 0 : HorizonIRF
    TETA_3(:,:,h+1) = J * CompanionMatrix_3Regime^h * J' * C_IRF;
end

for h = 0 : HorizonIRF
    TETA_3_12(:,:,h+1) = J * CompanionMatrix_3Regime_12^h * J' * C_IRF_12;
end

% Define titles for each shock scenario
Titles = cell(1,3);
Titles{1,1} = '$$\varepsilon_{UM_t}$$ Shock';
Titles{1,2} = '$$\varepsilon_{y_t}$$ Shock';
Titles{1,3} = '$$\varepsilon_{UF_t}$$ Shock';

% Define Y-axis labels for each variable
YLabel = cell(3,1);
YLabel{1,1} = '$$UM_t$$';
YLabel{2,1} = '$$y_t$$';
YLabel{3,1} = '$$UF_t$$';

M_IRF = 3;  
figure(13);
index = 1;
Shock_1=[1 1 1];

% Loop through each variable and shock for plotting
for jr = 1 : M_IRF
    for jc = 1 : M_IRF
        TETA_Iter_Sample = squeeze(TETA_3(jr,jc,:));
        TETA_Iter_Sample_12 = squeeze(TETA_3_12(jr,jc,:));
        TETA_Iter_Boot_Inf = squeeze(IRF_Inf_Boot3(jr,jc,:));
        TETA_Iter_Boot_Sup = squeeze(IRF_Sup_Boot3(jr,jc,:));
        TETA_Iter_Boot_Inf_12 = squeeze(IRF_Inf_Boot3_12(jr,jc,:));
        TETA_Iter_Boot_Sup_12 = squeeze(IRF_Sup_Boot3_12(jr,jc,:));
        subplot(M_IRF, M_IRF, index);
        x = 1:1:HorizonIRF+1;  

        % Plot the IRF
        y = TETA_Iter_Sample' * Shock_1(jr);  
        y12 = TETA_Iter_Sample_12' * Shock_1(jr); 
        plot(x, y, 'Color', [0 0.4470 0.7410], 'LineWidth', LineWidth_IRF);
        hold on;
        plot(x, y12, 'Color', [0.8500, 0.3250, 0.0980], 'LineWidth', LineWidth_IRF);

        % Add shaded confidence interval 
        fill([x fliplr(x)], [TETA_Iter_Boot_Inf' fliplr(TETA_Iter_Boot_Sup')], ...
        [0 0.4470 0.7410], 'EdgeColor', 'none', 'FaceAlpha', 0.2);  
        fill([x fliplr(x)], [TETA_Iter_Boot_Inf_12' fliplr(TETA_Iter_Boot_Sup_12')], ...
        [0.8500, 0.3250, 0.0980], 'EdgeColor', 'none', 'FaceAlpha', 0.2);

        % Add a zero line for reference
        plot(x, zeros(HorizonIRF+1, 1), 'k', 'LineWidth', 1);

        % Set labels and titles
        ylabel(YLabel{jr, 1}, 'interpreter', 'latex');
        title(Titles{1, jc}, 'interpreter', 'latex');
        set(gca, 'FontSize', FontSizeIRFGraph);
        set(gcf, 'Color', 'w'); 
        set(gca, 'Color', 'w');
        axis tight;  
        index = index + 1;
    end
end


%% IRF Q4

% Initialize storage for variance decompositions
PSI_var = zeros(M, M, HorizonIRF+1);
cum_PSI_var = zeros(M, M, HorizonIRF+1);
Var_dem_y = zeros(M, HorizonIRF+1);
Var_dem_y_12 = zeros(M, HorizonIRF+1);

C_IRF = SVAR_4Regime;  % instantaneous impact at h=0
C_IRF_12 = SVAR_4Regime_12;

% Compute impulse responses over the horizon
HorizonIRF = 60;  % Define the response horizon
for h = 0 : HorizonIRF
    TETA_4(:,:,h+1) = J * CompanionMatrix_4Regime^h * J' * C_IRF;

     % Squared impulse responses for variance decomposition
    PSI_var(:,:,h+1) = TETA_4(:,:,h+1).^2;
    cum_PSI_var(:,:,h+1) = sum(PSI_var(:,:,1:h+1), 3);
    
    % Compute normalized cumulative variance contributions for Y_t
    Var_dem_y(:,h+1) = cum_PSI_var(2,:,h+1) ./ sum(cum_PSI_var(2,:,h+1));
    
end

for h = 0 : HorizonIRF
    TETA_4_12(:,:,h+1) = J * CompanionMatrix_4Regime_12^h * J' * C_IRF_12;

    % Squared impulse responses for variance decomposition
    PSI_var(:,:,h+1) = TETA_4_12(:,:,h+1).^2;
    cum_PSI_var(:,:,h+1) = sum(PSI_var(:,:,1:h+1), 3);
    
    % Compute normalized cumulative variance contributions for Y_t
    Var_dem_y_12(:,h+1) = cum_PSI_var(2,:,h+1) ./ sum(cum_PSI_var(2,:,h+1));
end

% Define titles for each shock scenario
Titles = cell(1,3);
Titles{1,1} = '$$\varepsilon_{UM_t}$$ Shock';
Titles{1,2} = '$$\varepsilon_{y_t}$$ Shock';
Titles{1,3} = '$$\varepsilon_{UF_t}$$ Shock';

% Define Y-axis labels for each variable
YLabel = cell(3,1);
YLabel{1,1} = '$$UM_t$$';
YLabel{2,1} = '$$y_t$$';
YLabel{3,1} = '$$UF_t$$';

M_IRF = 3;  
figure(14);
index = 1;
Shock_1=[1 1 1];

% Loop through each variable and shock for plotting
for jr = 1 : M_IRF
    for jc = 1 : M_IRF
        TETA_Iter_Sample = squeeze(TETA_4(jr,jc,:));
        TETA_Iter_Sample_12 = squeeze(TETA_4_12(jr,jc,:));
        TETA_Iter_Boot_Inf = squeeze(IRF_Inf_Boot4(jr,jc,:));
        TETA_Iter_Boot_Sup = squeeze(IRF_Sup_Boot4(jr,jc,:));
        TETA_Iter_Boot_Inf_12 = squeeze(IRF_Inf_Boot4_12(jr,jc,:));
        TETA_Iter_Boot_Sup_12 = squeeze(IRF_Sup_Boot4_12(jr,jc,:));
        subplot(M_IRF, M_IRF, index);
        x = 1:1:HorizonIRF+1;  

        % Plot the IRF
        y = TETA_Iter_Sample' * Shock_1(jr); 
        y12 = TETA_Iter_Sample_12' * Shock_1(jr);  
        plot(x, y, 'Color', [0 0.4470 0.7410], 'LineWidth', LineWidth_IRF);
        hold on;
        plot(x, y12, 'Color', [0.8500, 0.3250, 0.0980], 'LineWidth', LineWidth_IRF);

        % Add shaded confidence interval 
        fill([x fliplr(x)], [TETA_Iter_Boot_Inf' fliplr(TETA_Iter_Boot_Sup')], ...
        [0 0.4470 0.7410], 'EdgeColor', 'none', 'FaceAlpha', 0.2);  
        fill([x fliplr(x)], [TETA_Iter_Boot_Inf_12' fliplr(TETA_Iter_Boot_Sup_12')], ...
        [0.8500, 0.3250, 0.0980], 'EdgeColor', 'none', 'FaceAlpha', 0.2);

        % Add a zero line for reference
        plot(x, zeros(HorizonIRF+1, 1), 'k', 'LineWidth', 1);

        % Set labels and titles
        ylabel(YLabel{jr, 1}, 'interpreter', 'latex');
        title(Titles{1, jc}, 'interpreter', 'latex');
        set(gca, 'FontSize', FontSizeIRFGraph);
        set(gcf, 'Color', 'w'); 
        set(gca, 'Color', 'w');
        axis tight;  
        index = index + 1;
    end
end

%{
%% IRF Q1

C_IRF = SVAR_1Regime;     % instantaneous impact at h=0

                       
    for h = 0 : HorizonIRF
    TETA_1(:,:,h+1)=J*CompanionMatrix_1Regime^h*J'*C_IRF;
    end
    
Titles=cell(1,3);
Titles{1,1}='$$\varepsilon_{UM}$$ $$Shock$$';
Titles{1,2}='$$\varepsilon_{y}$$ $$Shock$$';
Titles{1,3}='$$\varepsilon_{UF}$$ $$Shock$$';

YLabel=cell(3,1);
YLabel{1,1}='$$UM$$';
YLabel{2,1}='$$y$$';
YLabel{3,1}='$$UF$$';
index = 1;
Shock_1=[1 1 1];

M_IRF = 3;

figure(2)
for jr = 1 : M_IRF
    for jc = 1 : M_IRF
    TETA_Iter_Sample = squeeze(TETA_1(jr,jc,:));
    TETA_Iter_Boot_Inf = squeeze(IRF_Inf_Boot1(jr,jc,:));
    TETA_Iter_Boot_Sup = squeeze(IRF_Sup_Boot1(jr,jc,:));
    subplot(M_IRF,M_IRF,index)  
    x = 1:1:HorizonIRF+1;
    y= TETA_Iter_Sample'*Shock_1(jr);
    plot(y,'Color',[0 0.4470 0.7410], 'LineWidth',LineWidth_IRF);
    hold on
    plot(TETA_Iter_Boot_Inf,'--r', 'LineWidth',LineWidth_IRF_Boot);
    plot(TETA_Iter_Boot_Sup,'--r', 'LineWidth',LineWidth_IRF_Boot);
    plot(zeros(HorizonIRF+1,1),'k','LineWidth',1);
    ylabel(YLabel{jr,1},'interpreter','latex');
    title(Titles{1,jc},'interpreter','latex');
    set(gca,'FontSize',FontSizeIRFGraph);
    axis tight
    index=index+1;
    
    end
end

%% IRF Q2

C_IRF = SVAR_2Regime;     % instantaneous impact at h=0
                       
    for h = 0 : HorizonIRF
    TETA_2(:,:,h+1)=J*CompanionMatrix_2Regime^h*J'*C_IRF;
    end
    
Titles=cell(1,3);
Titles{1,1}='$$\varepsilon_{UM}$$ $$Shock$$';
Titles{1,2}='$$\varepsilon_{y}$$ $$Shock$$';
Titles{1,3}='$$\varepsilon_{UF}$$ $$Shock$$';

YLabel=cell(3,1);
YLabel{1,1}='$$UM$$';
YLabel{2,1}='$$y$$';
YLabel{3,1}='$$UF$$';
index = 1;
Shock_1=[1 1 1];

M_IRF = 3;

figure(3)
for jr = 1 : M_IRF
    for jc = 1 : M_IRF
    TETA_Iter_Sample = squeeze(TETA_2(jr,jc,:));
    TETA_Iter_Boot_Inf = squeeze(IRF_Inf_Boot2(jr,jc,:));
    TETA_Iter_Boot_Sup = squeeze(IRF_Sup_Boot2(jr,jc,:));
    subplot(M_IRF,M_IRF,index)  
    x = 1:1:HorizonIRF+1;
    y= TETA_Iter_Sample'*Shock_1(jr);
    plot(y,'Color',[0 0.4470 0.7410], 'LineWidth',LineWidth_IRF);
    hold on
    plot(TETA_Iter_Boot_Inf,'--r', 'LineWidth',LineWidth_IRF_Boot);
    plot(TETA_Iter_Boot_Sup,'--r', 'LineWidth',LineWidth_IRF_Boot);
    plot(zeros(HorizonIRF+1,1),'k','LineWidth',1);
    ylabel(YLabel{jr,1},'interpreter','latex');
    title(Titles{1,jc},'interpreter','latex');
    set(gca,'FontSize',FontSizeIRFGraph);
    axis tight
    index=index+1;
    
    end
end

%% IRF Q3

C_IRF = SVAR_3Regime;     % instantaneous impact at h=0
                       
    for h = 0 : HorizonIRF
    TETA_3(:,:,h+1)=J*CompanionMatrix_3Regime^h*J'*C_IRF;
    end
    
Titles=cell(1,3);
Titles{1,1}='$$\varepsilon_{UM}$$ $$Shock$$';
Titles{1,2}='$$\varepsilon_{y}$$ $$Shock$$';
Titles{1,3}='$$\varepsilon_{UF}$$ $$Shock$$';

YLabel=cell(3,1);
YLabel{1,1}='$$UM$$';
YLabel{2,1}='$$y$$';
YLabel{3,1}='$$UF$$';
index = 1;
Shock_1=[1 1 1];

M_IRF = 3;

figure(4)
for jr = 1 : M_IRF
    for jc = 1 : M_IRF
    TETA_Iter_Sample = squeeze(TETA_3(jr,jc,:));
    TETA_Iter_Boot_Inf = squeeze(IRF_Inf_Boot3(jr,jc,:));
    TETA_Iter_Boot_Sup = squeeze(IRF_Sup_Boot3(jr,jc,:));
    subplot(M_IRF,M_IRF,index)  
    x = 1:1:HorizonIRF+1;
    y= TETA_Iter_Sample'*Shock_1(jr);
    plot(y,'Color',[0 0.4470 0.7410], 'LineWidth',LineWidth_IRF);
    hold on
    plot(TETA_Iter_Boot_Inf,'--r', 'LineWidth',LineWidth_IRF_Boot);
    plot(TETA_Iter_Boot_Sup,'--r', 'LineWidth',LineWidth_IRF_Boot);
    plot(zeros(HorizonIRF+1,1),'k','LineWidth',1);
    ylabel(YLabel{jr,1},'interpreter','latex');
    title(Titles{1,jc},'interpreter','latex');
    set(gca,'FontSize',FontSizeIRFGraph);
    axis tight
    index=index+1;
    
    end
end

%% IRF Q4

C_IRF = SVAR_4Regime;     % instantaneous impact at h=0

    for h = 0 : HorizonIRF
    TETA_4(:,:,h+1)=J*CompanionMatrix_4Regime^h*J'*C_IRF;
    end
    
Titles=cell(1,3);
Titles{1,1}='$$\varepsilon_{UM}$$ $$Shock$$';
Titles{1,2}='$$\varepsilon_{y}$$ $$Shock$$';
Titles{1,3}='$$\varepsilon_{UF}$$ $$Shock$$';

YLabel=cell(3,1);
YLabel{1,1}='$$UM$$';
YLabel{2,1}='$$y$$';
YLabel{3,1}='$$UF$$';
index = 1;
Shock_1=[1 1 1];

M_IRF = 3;

figure(5)
for jr = 1 : M_IRF
    for jc = 1 : M_IRF
    TETA_Iter_Sample = squeeze(TETA_4(jr,jc,:));
    TETA_Iter_Boot_Inf = squeeze(IRF_Inf_Boot4(jr,jc,:));
    TETA_Iter_Boot_Sup = squeeze(IRF_Sup_Boot4(jr,jc,:));
    subplot(M_IRF,M_IRF,index)  
    x = 1:1:HorizonIRF+1;
    y= TETA_Iter_Sample'*Shock_1(jr);
    plot(y,'Color',[0 0.4470 0.7410], 'LineWidth',LineWidth_IRF);
    hold on
    plot(TETA_Iter_Boot_Inf,'--r', 'LineWidth',LineWidth_IRF_Boot);
    plot(TETA_Iter_Boot_Sup,'--r', 'LineWidth',LineWidth_IRF_Boot);
    plot(zeros(HorizonIRF+1,1),'k','LineWidth',1);
    ylabel(YLabel{jr,1},'interpreter','latex');
    title(Titles{1,jc},'interpreter','latex');
    set(gca,'FontSize',FontSizeIRFGraph);
    axis tight
    index=index+1;
    
    end
end

%}

%% Variance decomposition for Y_t in the fourth regime

figure(17);

% Plotting the variance decomposition for Y_t 
subplot(2,1,1); 
plot(0:HorizonIRF, Var_dem_y(1, :), 'LineWidth', 3, 'Color', 'b'); 
hold on; 
plot(0:HorizonIRF, Var_dem_y(2, :), 'LineWidth', 3, 'Color', 'k'); 
plot(0:HorizonIRF, Var_dem_y(3, :), 'LineWidth', 3, 'Color', 'r'); 
hold off; 
%xlabel('Time Horizon');
ylabel('Variance Contribution');
legend('ε_{Mt}', 'ε_{Yt}', 'ε_{Ft}', 'Location', 'best'); 
title('VD using uncertainty at h=1');
set(gca, 'FontSize', FontSizeIRFGraph);
set(gcf, 'Color', 'w'); 
set(gca, 'Color', 'w');
grid on;

% Using C_IRF_12
subplot(2,1,2); 
plot(0:HorizonIRF, Var_dem_y_12(1, :), 'LineWidth', 3, 'Color', 'b'); 
hold on; 
plot(0:HorizonIRF, Var_dem_y_12(2, :), 'LineWidth', 3, 'Color', 'k'); 
plot(0:HorizonIRF, Var_dem_y_12(3, :), 'LineWidth', 3, 'Color', 'r'); 
hold off; 
%xlabel('Time Horizon');
ylabel('Variance Contribution');
legend('ε_{Mt}', 'ε_{Yt}', 'ε_{Ft}', 'Location', 'best'); 
title('VD using uncertainty at h=12');
set(gca, 'FontSize', FontSizeIRFGraph);
set(gcf, 'Color', 'w'); 
set(gca, 'Color', 'w');
grid on;

%% IRF Combined Plot Without Bootstrap Bands

figure(6);
hold on;

% Setting the colors for each regime
regimeColors = lines(numRegimes);  

% Array to keep track of plot handles for the legend
hPlots = zeros(numRegimes, 1);  

for r = 1:numRegimes
    eval(sprintf('currentIRF = TETA_%d;', r));

    for jr = 1:M_IRF
        for jc = 1:M_IRF
            subplot(M_IRF, M_IRF, (jr-1)*3 + jc); 
            hold on;

            % Extract and plot the time series for the current IRF
            y = squeeze(currentIRF(jr, jc, :))';
            hPlots(r) = plot(1:HorizonIRF+1, y, 'Color', regimeColors(r, :), 'LineWidth', LineWidth_IRF);  % Save the handle of the plot
            plot(xlim, [0 0], 'k', 'LineWidth', 1);  
            if r == numRegimes  
                ylabel(YLabel{jr,1}, 'interpreter', 'latex');
                title(Titles{1,jc}, 'interpreter', 'latex');
                set(gca, 'FontSize', FontSizeIRFGraph);
                set(gcf, 'Color', 'w'); 
                set(gca, 'Color', 'w');
                axis tight;
            end
        end
    end
end

legend(hPlots, arrayfun(@(x) sprintf('Regime %d', x), 1:numRegimes, 'UniformOutput', false), 'Location', 'BestOutside');

hold off;


clc


%% Output 

disp('-------------------- Preliminary LR tests ----------------------')
disp('----------------------------------------------------------------')
disp('-------- H0: Pi1 = Pi2 = Pi3 = Pi4 ∧ Σ1 = Σ2 = Σ3 = Σ4 ---------')
disp('----------------------------------------------------------------')

disp('Test statistics:')
disp(LR_Pi_sigma)
disp('P-value:')
disp(PVal_LRTest_Pi_sigma)

disp(' ')
disp('----------------------------------------------------------------')
disp('------------------- H0: Σ1 = Σ2 = Σ3 = Σ4  ---------------------')
disp('----------------------------------------------------------------')
disp('Test statistics:')
disp(LR_sigma)
disp('P-value:')
disp(PVal_LRTest_sigma)

disp(' ')
disp('----------------------------------------------------------------')
disp('------ Doornik-Hansen Omnibus Multivariate Normality Test ------')
disp('----------------------- All the sample -------------------------')
disp(' ')
DoornikHansenTest(Errors);
disp(' ')
disp('----------------------------------------------------------------')
disp('------ Doornik-Hansen Omnibus Multivariate Normality Test ------')
disp('------------------------- First regime -------------------------')
disp(' ')
DoornikHansenTest(Errors1);
disp(' ')
disp('----------------------------------------------------------------')
disp('------ Doornik-Hansen Omnibus Multivariate Normality Test ------')
disp('------------------------- Second regime ------------------------')
disp(' ')
DoornikHansenTest(Errors2);
disp(' ')
disp('----------------------------------------------------------------')
disp('------ Doornik-Hansen Omnibus Multivariate Normality Test ------')
disp('------------------------- Third regime -------------------------')
disp(' ')
DoornikHansenTest(Errors3);
disp(' ')
disp('----------------------------------------------------------------')
disp('------ Doornik-Hansen Omnibus Multivariate Normality Test ------')
disp('------------------------- Fourth regime ------------------------')
disp(' ')
DoornikHansenTest(Errors4);
disp(' ')



disp('----------------------------------------------------------------')
disp('--------------------- Main specification -----------------------')
disp('------------------ (Third row of Table 2) ----------------------')
disp('----------------------- Coefficients ---------------------------')

disp('B=')
disp(SVAR_1Regime)
disp('B+Q2=')
disp(SVAR_2Regime)
disp('B+Q2+Q3=')
disp(SVAR_3Regime)
disp('B+Q2+Q3+Q4=')
disp(SVAR_4Regime)

disp('---------------------- Standard Errors -------------------------')

disp('B=')
disp(SVAR_1Regime_SE)
disp('B+Q2=')
disp(SVAR_2Regime_SE)
disp('B+Q2+Q3=')
disp(SVAR_3Regime_SE)
disp('B+Q2+Q3+Q4=')
disp(SVAR_4Regime_SE)

disp('----------------------- Bootstrap S.E. -------------------------')

disp('B=')
disp(SE_SVAR_1)
disp('B+Q2=')
disp(SE_SVAR_2)
disp('B+Q2+Q3=')
disp(SE_SVAR_3)
disp('B+Q2+Q3+Q4=')
disp(SE_SVAR_4)

disp(' ')
disp('----------------------------------------------------------------')
disp('----- Necessary and suffient condition for identification ------')

disp(' ')
if Jacobian_rank == StructuralParam
    disp('The Jacobian matrix is full rank')
else
    disp('The Jacobian matrix is not full rank')
end

disp('----------------------------------------------------------------')
disp('------------------- Likelihood ratio test ----------------------')
disp(' ')
disp('Test statistics:')
disp(2 * ((LK_1Regime(1)+LK_2Regime(1)+LK_3Regime(1)+LK_4Regime(1))+LK_Estimation))
disp('P-value:')
disp(PVarl_LRTest)

%% ------ Negative significant peaks of the IRFs for h=1---------------
disp(' ')
disp('------------ Significant peaks of the IRFs for h=1 -------------')
desiredCombos = [1 2; 3 2; 2 1; 2 3];

% Regime 1
disp(' ')
disp('----------------------------------------------------------------')
disp('------- Significant negative peaks for the first regime --------')
disp('----------------------------------------------------------------')
for index = 1:size(desiredCombos, 1)
    jr = desiredCombos(index, 1);  
    jc = desiredCombos(index, 2);  
        [peakValue, peakTime] = findNegativePeaks(squeeze(TETA_1(jr,jc,:)), ...
                                                  squeeze(IRF_Inf_Boot1(jr,jc,:)), ...
                                                  squeeze(IRF_Sup_Boot1(jr,jc,:)));
        fprintf('Shock %d -> Variable %d: Peak Value = %.4f at Time = %d\n', jc, jr, peakValue, peakTime);
end

% Regime 2
disp('----------------------------------------------------------------')
disp('------- Significant negative peaks for the second regime -------')
disp('----------------------------------------------------------------')

for index = 1:size(desiredCombos, 1)
    jr = desiredCombos(index, 1);  
    jc = desiredCombos(index, 2);
    [peakValue, peakTime] = findNegativePeaks(squeeze(TETA_2(jr,jc,:)), ...
                                              squeeze(IRF_Inf_Boot2(jr,jc,:)), ...
                                              squeeze(IRF_Sup_Boot2(jr,jc,:)));
    fprintf('Shock %d -> Variable %d: Peak Value = %.4f at Time = %d\n', jc, jr, peakValue, peakTime);
end

% Regime 3
disp('----------------------------------------------------------------')
disp('------- Significant negative peaks for the third regime --------')
disp('----------------------------------------------------------------')

for index = 1:size(desiredCombos, 1)
    jr = desiredCombos(index, 1);  
    jc = desiredCombos(index, 2);
        [peakValue, peakTime] = findNegativePeaks(squeeze(TETA_3(jr,jc,:)), ...
                                                  squeeze(IRF_Inf_Boot3(jr,jc,:)), ...
                                                  squeeze(IRF_Sup_Boot3(jr,jc,:)));
        fprintf('Shock %d -> Variable %d: Peak Value = %.4f at Time = %d\n', jc, jr, peakValue, peakTime);
end

% Regime 4
disp('----------------------------------------------------------------')
disp('------- Significant negative peaks for the fourth regime -------')
disp('----------------------------------------------------------------')

for index = 1:size(desiredCombos, 1)
    jr = desiredCombos(index, 1);  
    jc = desiredCombos(index, 2);
        [peakValue, peakTime] = findNegativePeaks(squeeze(TETA_4(jr,jc,:)), ...
                                                  squeeze(IRF_Inf_Boot4(jr,jc,:)), ...
                                                  squeeze(IRF_Sup_Boot4(jr,jc,:)));
        fprintf('Shock %d -> Variable %d: Peak Value = %.4f at Time = %d\n', jc, jr, peakValue, peakTime);
end


disp(' ')


%% ------ Negative significant peaks of the IRFs for h=2---------------
disp(' ')
disp('------------ Significant peaks of the IRFs for h=12 ------------')
% Regime 1
disp(' ')
disp('----------------------------------------------------------------')
disp('------- Significant negative peaks for the first regime --------')
disp('----------------------------------------------------------------')
for index = 1:size(desiredCombos, 1)
    jr = desiredCombos(index, 1);  
    jc = desiredCombos(index, 2);  
        [peakValue, peakTime] = findNegativePeaks(squeeze(TETA_1_12(jr,jc,:)), ...
                                                  squeeze(IRF_Inf_Boot1_12(jr,jc,:)), ...
                                                  squeeze(IRF_Sup_Boot1_12(jr,jc,:)));
        fprintf('Shock %d -> Variable %d: Peak Value = %.4f at Time = %d\n', jc, jr, peakValue, peakTime);
end

% Regime 2
disp('----------------------------------------------------------------')
disp('------- Significant negative peaks for the second regime -------')
disp('----------------------------------------------------------------')

for index = 1:size(desiredCombos, 1)
    jr = desiredCombos(index, 1);  
    jc = desiredCombos(index, 2);
    [peakValue, peakTime] = findNegativePeaks(squeeze(TETA_2_12(jr,jc,:)), ...
                                              squeeze(IRF_Inf_Boot2_12(jr,jc,:)), ...
                                              squeeze(IRF_Sup_Boot2_12(jr,jc,:)));
    fprintf('Shock %d -> Variable %d: Peak Value = %.4f at Time = %d\n', jc, jr, peakValue, peakTime);
end

% Regime 3
disp('----------------------------------------------------------------')
disp('------- Significant negative peaks for the third regime --------')
disp('----------------------------------------------------------------')

for index = 1:size(desiredCombos, 1)
    jr = desiredCombos(index, 1);  
    jc = desiredCombos(index, 2);
        [peakValue, peakTime] = findNegativePeaks(squeeze(TETA_3_12(jr,jc,:)), ...
                                                  squeeze(IRF_Inf_Boot3_12(jr,jc,:)), ...
                                                  squeeze(IRF_Sup_Boot3_12(jr,jc,:)));
        fprintf('Shock %d -> Variable %d: Peak Value = %.4f at Time = %d\n', jc, jr, peakValue, peakTime);
end

% Regime 4
disp('----------------------------------------------------------------')
disp('------- Significant negative peaks for the fourth regime -------')
disp('----------------------------------------------------------------')

for index = 1:size(desiredCombos, 1)
    jr = desiredCombos(index, 1);  
    jc = desiredCombos(index, 2);
        [peakValue, peakTime] = findNegativePeaks(squeeze(TETA_4_12(jr,jc,:)), ...
                                                  squeeze(IRF_Inf_Boot4_12(jr,jc,:)), ...
                                                  squeeze(IRF_Sup_Boot4_12(jr,jc,:)));
        fprintf('Shock %d -> Variable %d: Peak Value = %.4f at Time = %d\n', jc, jr, peakValue, peakTime);
end


disp(' ')
disp('-------------------------------------------------------------------')
disp(' Endogenous UM1 and reverse causality from UM1 to UF1 specification')
disp('-------------------- First row of Table 2 -------------------------')
disp('----------------------- Coefficients ------------------------------')

disp('B=')
disp(SVAR_1Regime_endog)
disp('B+Q2=')
disp(SVAR_2Regime_endog)
disp('B+Q2+Q3=')
disp(SVAR_3Regime_endog)
disp('B+Q2+Q3+Q4=')
disp(SVAR_4Regime_endog)

disp('---------------------- Standard Errors ----------------------------')

disp('B=')
disp(SVAR_1Regime_SE_endog)
disp('B+Q2=')
disp(SVAR_2Regime_SE_endog)
disp('B+Q2+Q3=')
disp(SVAR_3Regime_SE_endog)
disp('B+Q2+Q3+Q4=')
disp(SVAR_4Regime_SE_endog)

disp(' ')
disp('-------------------------------------------------------------------')
disp('------ Necessary and suffient condition for identification --------')
disp(' ')
if Jacobian_rank_endog == StructuralParam_endog
    disp('The Jacobian matrix is full rank')
else
    disp('The Jacobian matrix is not full rank')
end

disp('-------------------------------------------------------------------')
disp('-------------------- Likelihood ratio test ------------------------')

disp('Test statistics:')
disp(2 * ((LK_1Regime(1)+LK_2Regime(1)+LK_3Regime(1)+LK_4Regime(1))+LK_Estimation_endog))
disp('P-value:')
disp(PVarl_LRTest_endog)

disp('----------------------------------------------------------------')
disp('------------------- Likelihood ratio test ----------------------')
disp('----- Between main and first (endogenous) specification --------')
disp(' ')
disp('Test statistics:')
disp(-2 * (LK_Estimation_endog-LK_Estimation))
disp('P-value:')
disp(PVarl_LRTest_ex_end)

disp(' ')
disp('---------------------------------------------------------------------------')
disp(' Endogenous UM1 and reverse causality from UM1 to UF1 specification (Covid)')
disp('----------------------- Second row of Table 2  ----------------------------')
disp('---------------------------- Coefficients ---------------------------------')

disp('B=')
disp(SVAR_1Regime_endog2)
disp('B+Q2=')
disp(SVAR_2Regime_endog2)
disp('B+Q2+Q3=')
disp(SVAR_3Regime_endog2)
disp('B+Q2+Q3+Q4=')
disp(SVAR_4Regime_endog2)

disp('---------------------- Standard Errors ----------------------------')

disp('B=')
disp(SVAR_1Regime_SE_endog2)
disp('B+Q2=')
disp(SVAR_2Regime_SE_endog2)
disp('B+Q2+Q3=')
disp(SVAR_3Regime_SE_endog2)
disp('B+Q2+Q3+Q4=')
disp(SVAR_4Regime_SE_endog2)

disp(' ')
disp('-------------------------------------------------------------------')
disp('------ Necessary and suffient condition for identification --------')
disp(' ')
if Jacobian_rank_endog2 == StructuralParam_endog2
    disp('The Jacobian matrix is full rank')
else
    disp('The Jacobian matrix is not full rank')
end

disp('-------------------------------------------------------------------')
disp('-------------------- Likelihood ratio test ------------------------')

disp('Test statistics:')
disp(2 * ((LK_1Regime(1)+LK_2Regime(1)+LK_3Regime(1)+LK_4Regime(1))+LK_Estimation_endog2))
disp('P-value:')
disp(PVarl_LRTest_endog2)

disp('----------------------------------------------------------------')
disp('------------------- Likelihood ratio test ----------------------')
disp('------ Between main and second (endogenous) specification ------')
disp(' ')
disp('Test statistics:')
disp(-2 * (LK_Estimation_endog2-LK_Estimation))
disp('P-value:')
disp(PVarl_LRTest_ex_end2)
