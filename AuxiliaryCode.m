%% Pondini R., "Uncertainty and Business Cycles: New Insights Post-COVID-19", a contribution to  Angelini G., Bacchiocchi E., Caggiano G. and Fanelli L. (2018) "Uncertainty Across Volatility Regimes" 

%-----------------------------------------------------------------------
% Press the button change folder if it's the first time running the code
%-----------------------------------------------------------------------

%% Auxiliary code for the Robustness test presented in Section 3.6 using the rate of growth of employement as a proxy for economic activity
%% and for the test (Section 3.4) where UM1 and UF1 are inverted to check for the robustness of the estimated roles they have inside the system
%%

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
load DataSet_Emp.txt

% Break dates
TB1=284; 
TB2=569;
TB3=712;

%% Data Set
DataSet = DataSet_Emp(2:end,[1 3 2]);
AllDataSet=DataSet;
M=size(DataSet,2);

UM1=DataSet(:,1); % Macro Uncertainty variable
Y=DataSet(:,2); % Employement variable
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
df_Pi_sigma = (numRegimes-1)*((M^2)*NLags + (M*(M+1))/2);
PVal_LRTest_Pi_sigma = 1-chi2cdf(LR_Pi_sigma, df_Pi_sigma);

%% Chow type tests for Sigma_u variability across regimes keeping Pi fixed
LR_sigma = 2 * ((LK_Pi_1+LK_Pi_2+LK_Pi_3+LK_Pi_4)-LK_All(1));
df_sigma = (numRegimes-1)*((M*(M+1))/2);
PVal_LRTest_sigma = 1-chi2cdf(LR_sigma, df_sigma);


%% Estimation of the structural parameters

StructuralParam=20; 
InitialValue_SVAR_Initial=0.5*ones(StructuralParam,1);

% ML function
[StructuralParam_Estiamtion_MATRIX,Likelihood_MATRIX,exitflag,output,grad,Hessian_MATRIX] = fminunc('Likelihood_SVAR_Restricted_Emp',InitialValue_SVAR_Initial',options);

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

SVAR_Q4=[StructuralParam_Estiamtion(15) 0                              StructuralParam_Estiamtion(18);
         StructuralParam_Estiamtion(16) StructuralParam_Estiamtion(17) StructuralParam_Estiamtion(19);
         0                              0                              StructuralParam_Estiamtion(20)];  

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
HSelection(29, 16) = 1;  
HSelection(32, 17) = 1;  
HSelection(34, 18) = 1;  
HSelection(35, 19) = 1;  
HSelection(36, 20) = 1;  

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

     
% Calculates the variance and the standard errors of the estimated coefficients        
VarTeta=  (Jacobian'* TetaMatrix^(-1)*Jacobian)^(-1);
SETetaJacobian= diag(VarTeta).^0.5;
 
%% IRF Q1

C_IRF = SVAR_1Regime;  % instantaneous impact at h=0
J=[eye(M) zeros(M,M*(NLags-1))]; % selection matrix

% Compute impulse responses over the horizon
HorizonIRF = 60;  % Define the response horizon
for h = 0 : HorizonIRF
    TETA_1(:,:,h+1) = J * CompanionMatrix_1Regime^h * J' * C_IRF;
end


%% IRF Q2

C_IRF = SVAR_2Regime;  % instantaneous impact at h=0

% Compute impulse responses over the horizon
HorizonIRF = 60;  % Define the response horizon
for h = 0 : HorizonIRF
    TETA_2(:,:,h+1) = J * CompanionMatrix_2Regime^h * J' * C_IRF;
end


%% IRF Q3

C_IRF = SVAR_3Regime;  % instantaneous impact at h=0

% Compute impulse responses over the horizon
HorizonIRF = 60;  % Define the response horizon
for h = 0 : HorizonIRF
    TETA_3(:,:,h+1) = J * CompanionMatrix_3Regime^h * J' * C_IRF;
end



%% IRF Q4

C_IRF = SVAR_4Regime;  % instantaneous impact at h=0

% Compute impulse responses over the horizon
HorizonIRF = 60;  % Define the response horizon
for h = 0 : HorizonIRF
    TETA_4(:,:,h+1) = J * CompanionMatrix_4Regime^h * J' * C_IRF;
end



%% IRF Combined Plot Without Bootstrap Bands

% Preparing the figures
figure(6);
hold on;
M_IRF = 3;

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

% Setting the colors for each regime
regimeColors = lines(numRegimes);  
% Array to keep track of plot handles for the legend
hPlots = zeros(numRegimes, 1);  % Preallocation

for r = 1:numRegimes
    % Select the correct IRF matrix for each regime
    eval(sprintf('currentIRF = TETA_%d;', r));

    for jr = 1:M_IRF
        for jc = 1:M_IRF
            subplot(M_IRF, M_IRF, (jr-1)*3 + jc);  
            hold on;

            % Extract and plot the time series for the current IRF
            y = squeeze(currentIRF(jr, jc, :))';
            hPlots(r) = plot(1:HorizonIRF+1, y, 'Color', regimeColors(r, :), 'LineWidth', LineWidth_IRF);  
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

%% Test where UM1 and UF1 were inverted to check for the robustness of the estimated roles they have inside the system

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
DataSet = DataSet(2:end,[2 3 1]);
AllDataSet=DataSet;
M=size(DataSet,2);

UF1=DataSet(:,1); % Financial Uncertainty variable
Y=DataSet(:,2); % Industrial Production variable
UM1=DataSet(:,3); % Macro Uncertainty variable

% Creates the data for the three regimes
DataSet_1Regime_2=DataSet(1:TB1,:); % First regime
DataSet_2Regime_2=DataSet(TB1+1-NLags:TB2,:); % Second regime
DataSet_3Regime_2=DataSet(TB2+1-NLags:TB3,:); % Third regime
DataSet_4Regime_2=DataSet(TB3+1-NLags:end,:); % Fourth regime

% Size three regimes
T1 = size(DataSet_1Regime_2,1)-NLags;
T2 = size(DataSet_2Regime_2,1)-NLags;
T3 = size(DataSet_3Regime_2,1)-NLags;
T4 = size(DataSet_4Regime_2,1)-NLags;
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

[Beta_LK_2,Log_LK,exitflag,~,grad,HESSIAN_LK] = fminunc('Likelihood_UNRESTRICTED',Beta_OLS',options2);

CommonPI=Beta_LK_2;
Errors=VAR_Variables_Y-VAR_Variables_X*Beta_LK_2';
Omega_LK_2=1/(T)*(Errors'*Errors);
LK_All_2=-Log_LK;

% ******************************************************************************
% First Regime
% ******************************************************************************
T=T1;
DataSet=DataSet_1Regime_2;
VAR_Variables_X=[ones(size(DataSet(NLags:end-1,:),1),1) DataSet(NLags:end-1,:) DataSet(NLags-1:end-2,:) DataSet(NLags-2:end-3,:) DataSet(NLags-3:end-4,:)];
VAR_Variables_Y=DataSet(NLags+1:end,:);

Beta_OLS=(VAR_Variables_X'*VAR_Variables_X)^(-1)*VAR_Variables_X'*VAR_Variables_Y;

[Beta_LK1_2,Log_LK,exitflag,~,grad,HESSIAN_LK] = fminunc('Likelihood_UNRESTRICTED',Beta_OLS',options2);

Errors1=VAR_Variables_Y-VAR_Variables_X*Beta_LK1_2';
Omega_LK1_2=1/(T)*(Errors1'*Errors1);

CompanionMatrix_1Regime_2=[Beta_LK1_2(:,2:end);
    eye(M*(NLags-1),M*(NLags-1)) zeros(M*(NLags-1),M)];     
Sigma_1Regime=Omega_LK1_2;
                            

% Likelihood of the VAR in the first regime
LK_1Regime_2=-Log_LK;


% ******************************************************************************
% Second Regime
% ******************************************************************************

T=T2;
DataSet=DataSet_2Regime_2;
VAR_Variables_X=[ones(size(DataSet(NLags:end-1,:),1),1) DataSet(NLags:end-1,:) DataSet(NLags-1:end-2,:) DataSet(NLags-2:end-3,:) DataSet(NLags-3:end-4,:)];
VAR_Variables_Y=DataSet(NLags+1:end,:);

Beta_OLS=(VAR_Variables_X'*VAR_Variables_X)^(-1)*VAR_Variables_X'*VAR_Variables_Y;

[Beta_LK2_2,Log_LK,exitflag,~,grad,HESSIAN_LK] = fminunc('Likelihood_UNRESTRICTED',Beta_OLS',options2);

Errors2=VAR_Variables_Y-VAR_Variables_X*Beta_LK2_2';
Omega_LK2_2=1/(T)*(Errors2'*Errors2);

CompanionMatrix_2Regime_2=[Beta_LK2_2(:,2:end);
    eye(M*(NLags-1),M*(NLags-1)) zeros(M*(NLags-1),M)]; 

Sigma_2Regime=Omega_LK2_2;
                             
% Likelihood of the VAR in the second regime
LK_2Regime_2=-Log_LK;

          
% ******************************************************************************
% Third Regime
% ******************************************************************************

T=T3;
DataSet=DataSet_3Regime_2;
VAR_Variables_X=[ones(size(DataSet(NLags:end-1,:),1),1) DataSet(NLags:end-1,:) DataSet(NLags-1:end-2,:) DataSet(NLags-2:end-3,:) DataSet(NLags-3:end-4,:)];
VAR_Variables_Y=DataSet(NLags+1:end,:);

Beta_OLS=(VAR_Variables_X'*VAR_Variables_X)^(-1)*VAR_Variables_X'*VAR_Variables_Y;

[Beta_LK3_2,Log_LK,exitflag,~,grad,HESSIAN_LK] = fminunc('Likelihood_UNRESTRICTED',Beta_OLS',options2);

Errors3=VAR_Variables_Y-VAR_Variables_X*Beta_LK3_2';
Omega_LK3_2=1/(T)*(Errors3'*Errors3);
                            
CompanionMatrix_3Regime_2=[Beta_LK3_2(:,2:end);
    eye(M*(NLags-1),M*(NLags-1)) zeros(M*(NLags-1),M)];                            


Sigma_3Regime=Omega_LK3_2;

% Likelihood of the VAR in the third regime  
LK_3Regime_2=-Log_LK;

% ******************************************************************************
% Fourth Regime
% ******************************************************************************

T=T4;
DataSet=DataSet_4Regime_2;
VAR_Variables_X=[ones(size(DataSet(NLags:end-1,:),1),1) DataSet(NLags:end-1,:) DataSet(NLags-1:end-2,:) DataSet(NLags-2:end-3,:) DataSet(NLags-3:end-4,:)];
VAR_Variables_Y=DataSet(NLags+1:end,:);

Beta_OLS=(VAR_Variables_X'*VAR_Variables_X)^(-1)*VAR_Variables_X'*VAR_Variables_Y;

[Beta_LK4_2,Log_LK,exitflag,~,grad,HESSIAN_LK] = fminunc('Likelihood_UNRESTRICTED',Beta_OLS',options2);

Errors4=VAR_Variables_Y-VAR_Variables_X*Beta_LK4_2';
Omega_LK4_2=1/(T)*(Errors4'*Errors4);

                            
CompanionMatrix_4Regime_2=[Beta_LK4_2(:,2:end);
    eye(M*(NLags-1),M*(NLags-1)) zeros(M*(NLags-1),M)];                            
                                      

Sigma_4Regime=Omega_LK4_2;

% Likelihood of the VAR in the fourth regime  
LK_4Regime_2=-Log_LK;


%% Estimation of the structural parameters

StructuralParam_2=21; 
InitialValue_SVAR_Initial=0.5*ones(StructuralParam_2,1);

% ML function
[StructuralParam_Estiamtion_MATRIX_2,Likelihood_MATRIX_2,exitflag,output,grad,Hessian_MATRIX] = fminunc('Likelihood_SVAR_Restricted_endog2',InitialValue_SVAR_Initial',options);

StructuralParam_Estiamtion_2=StructuralParam_Estiamtion_MATRIX_2;
LK_Estimation_2=Likelihood_MATRIX_2;
Hessian_Estimation_2=Hessian_MATRIX;
SE_Estimation_2=diag(Hessian_Estimation^(-1)).^0.5;

%% Overidentification LR test
PVarl_LRTest_2 = 1-chi2cdf(2 * ((LK_1Regime_2(1)+LK_2Regime_2(1)+LK_3Regime_2(1)+LK_4Regime_2(1))+LK_Estimation_2),24-StructuralParam_2);

%% Here below we define the matrices of the structural parameters with restrictions on the coefficients as described in the paper. SVAR_C corresponds to the B matrix in the paper, SVAR_Q2 corresponds to
% the Q2 matrix of the paper, SVAR_Q3 corresponds to the Q3 matrix of the paper and SVAR_Q4 corresponds to the Q4 matrix of the paper.

SVAR_C=[StructuralParam_Estiamtion_2(1) 0                               0;
        StructuralParam_Estiamtion_2(2) StructuralParam_Estiamtion_2(3) 0;
        0                               0                               StructuralParam_Estiamtion_2(4)];

SVAR_Q2=[StructuralParam_Estiamtion_2(5)  0                               StructuralParam_Estiamtion_2(8);
         StructuralParam_Estiamtion_2(6)  StructuralParam_Estiamtion_2(7) 0;
         0                                0                               StructuralParam_Estiamtion_2(9)];

SVAR_Q3=[0                                0                                StructuralParam_Estiamtion_2(13);
         StructuralParam_Estiamtion_2(10) StructuralParam_Estiamtion_2(12) StructuralParam_Estiamtion_2(14);
         StructuralParam_Estiamtion_2(11) 0                                StructuralParam_Estiamtion_2(15)];

SVAR_Q4=[StructuralParam_Estiamtion_2(16) StructuralParam_Estiamtion_2(17) StructuralParam_Estiamtion_2(19);
         0                                StructuralParam_Estiamtion_2(18) StructuralParam_Estiamtion_2(20);
         0                                0                                StructuralParam_Estiamtion_2(21)];  

SVAR_1Regime_2=SVAR_C; % B
SVAR_2Regime_2=SVAR_C+SVAR_Q2;   % B+Q2
SVAR_3Regime_2=SVAR_C+SVAR_Q2+SVAR_Q3;  % B+Q2+Q3
SVAR_4Regime_2=SVAR_C+SVAR_Q2+SVAR_Q3+SVAR_Q4;  % B+Q2+Q3+Q4

% Flip the sign if the parameter on the main diagonal is negative

	if SVAR_1Regime_2(1,1)<0
    SVAR_1Regime_2(:,1)=-SVAR_1Regime_2(:,1);
    end
    if SVAR_1Regime_2(2,2)<0
    SVAR_1Regime_2(:,2)=-SVAR_1Regime_2(:,2); 
    end
    if SVAR_1Regime_2(3,3)<0
    SVAR_1Regime_2(:,3)=-SVAR_1Regime_2(:,3);
    end
    
	if SVAR_2Regime_2(1,1)<0
    SVAR_2Regime_2(:,1)=-SVAR_2Regime_2(:,1);
    end
    if SVAR_2Regime_2(2,2)<0
    SVAR_2Regime_2(:,2)=-SVAR_2Regime_2(:,2); 
    end
    if SVAR_2Regime_2(3,3)<0
    SVAR_2Regime_2(:,3)=-SVAR_2Regime_2(:,3);
     end
    
    
    if SVAR_3Regime_2(1,1)<0
    SVAR_3Regime_2(:,1)=-SVAR_3Regime_2(:,1);
    end
    if SVAR_3Regime_2(2,2)<0
    SVAR_3Regime_2(:,2)=-SVAR_3Regime_2(:,2); 
    end
    if SVAR_3Regime_2(3,3)<0
    SVAR_3Regime_2(:,3)=-SVAR_3Regime_2(:,3);
    end
    

    if SVAR_4Regime_2(1,1)<0
    SVAR_4Regime_2(:,1)=-SVAR_4Regime_2(:,1);
    end
    if SVAR_4Regime_2(2,2)<0
    SVAR_4Regime_2(:,2)=-SVAR_4Regime_2(:,2); 
    end
    if SVAR_4Regime_2(3,3)<0
    SVAR_4Regime_2(:,3)=-SVAR_4Regime_2(:,3);
    end
    
     
MATRICES_2=[SVAR_1Regime_2;
          SVAR_2Regime_2;
          SVAR_3Regime_2;
          SVAR_4Regime_2];
   
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
HSelection(21,11)=1;
HSelection(23,12)=1;
HSelection(25,13)=1;
HSelection(26,14)=1;
HSelection(27, 15) = 1;
HSelection(28, 16) = 1;  
HSelection(31, 17) = 1;  
HSelection(32, 18) = 1;  
HSelection(34, 19) = 1;  
HSelection(35, 20) = 1;
HSelection(36, 21) = 1; 

Jacobian_2= RankMatrix*HSelection;

% Report the rank of the matrix for checking the identification
Jacobian_rank_2 = rank(Jacobian_2);


clc


%% Output first test


disp('----------------------------------------------------------------')
disp('-------------------- Main specification ------------------------')
disp('--------------------- Employement test--------------------------')
disp('----------------------- Coefficients ---------------------------')

disp('B=')
disp(SVAR_1Regime)
disp('B+Q2=')
disp(SVAR_2Regime)
disp('B+Q2+Q3=')
disp(SVAR_3Regime)
disp('B+Q2+Q3+Q4=')
disp(SVAR_4Regime)

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



%% Output second test

disp('----------------------------------------------------------------')
disp('-------------------- Main specification ------------------------')
disp('----------------- Inverted UM1 and Uf1 test---------------------')
disp('----------------------- Coefficients ---------------------------')

disp('B=')
disp(SVAR_1Regime_2)
disp('B+Q2=')
disp(SVAR_2Regime_2)
disp('B+Q2+Q3=')
disp(SVAR_3Regime_2)
disp('B+Q2+Q3+Q4=')
disp(SVAR_4Regime_2)

disp(' ')
disp('----------------------------------------------------------------')
disp('----- Necessary and suffient condition for identification ------')

disp(' ')
if Jacobian_rank_2 == StructuralParam_2
    disp('The Jacobian matrix is full rank')
else
    disp('The Jacobian matrix is not full rank')
end

disp('----------------------------------------------------------------')
disp('------------------- Likelihood ratio test ----------------------')
disp(' ')
disp('Test statistics:')
disp(2 * ((LK_1Regime_2(1)+LK_2Regime_2(1)+LK_3Regime_2(1)+LK_4Regime_2(1))+LK_Estimation_2))
disp('P-value:')
disp(PVarl_LRTest_2)


