Pondini R., "Uncertainty and Business Cycles: New Insights Post-COVID-19", a contribution to  Angelini G., Bacchiocchi E., Caggiano G. and Fanelli L. (2018) "Uncertainty Across Volatility Regimes" 

/*------------------------------- Files -------------------------------*/

1 - The folder Datasets contains all the Datasets used in this paper

2 - Functions: folder containing all the functions called by MainCode.m and AuxiliaryCode.m

3 - Figures contains all the plots of the paper

4 - MainCode.m is the code to produce every quantitative analysis in the paper,
    the others two missing tests are reproduced in the AuxiliaryCode.m file

5 - Uncertainty_and_Business_Cycles_New_Insights_Post_COVID_19.pdf: the paper in pdf form


/*------------------------------ MainCode.m ---------------------------*/

This file contains the code for replicating the quantitative analysis presented in the paper.
The code first loads the data from DataSet.txt and estimates the reduced-form VAR coefficients across the four volatility regimes.
It then uses maximum likelihood (ML) to estimate the structural parameters according to the three specifications discussed and presented in Table 2.

Additionally, the code incorporates DataSet12.txt to estimate the main specification using uncertainty at a one-year horizon.
Impulse response functions are computed and plotted for both cases, alongside other analyses.

Furthermore, all plots and tables presented in the paper are reproduced within this code. The results—except for the figures,
which will be displayed on the desktop—will be output in the MATLAB Command Window.


/*------------------------------ AuxiliaryCode.m ---------------------------*/


This file replicates the robustness test by using the employment growth rate as a proxy for real economic activity,
replacing the initially used industrial production growth rate (Section 3.6).
It then reproduces the test presented in Section 3.4,
where we estimate the structural model by reversing the roles of financial uncertainty and macroeconomic uncertainty.


/*---------------------------- DataSet.xlsx ---------------------------*/

This file contains, the data concerning the
following variables:

1 - UM1: Measure of 1-period-ahead macroeconomic uncertainty
2 - UF1: Measure of 1-period-ahead financial uncertainty
3 - Y: Growth rate of industrial production (%) 

The sheet contains a brief description of the variables and the
related source.

/*---------------------------- DataSet12.xlsx ---------------------------*/

This file contains, the data concerning the
following variables:

1 - UM12: Measure of 12-period-ahead macroeconomic uncertainty
2 - UF12: Measure of 12-period-ahead financial uncertainty
3 - Y: Growth rate of industrial production (%) 

The sheet contains a brief description of the variables and the
related source.

/*---------------------------- DataSet_Emp.xlsx ---------------------------*/

This file contains, the data concerning the
following variables:

1 - UM1: Measure of 1-period-ahead macroeconomic uncertainty
2 - UF1: Measure of 1-period-ahead financial uncertainty
3 - EMP: Growth rate of employment (%) 

The sheet contains a brief description of the variables and the
related source.

/*---------------------------------------------------------------------------*/

All datasets are provided in .txt format to ensure compatibility with the MATLAB code.

/*---------------------------------------------------------------------------*/

/*---------------------------- Functions ------------------------------*/

Folder containing all the functions called by MainCode.m

1 - Likelihood_UNRESTRICTED.m: obtains the reduced form parameters

2 - Likelihood_SVAR_Restricted.m: obtains the structural
    parameters for the main specification in the third row of Table 2

3 - Likelihood_SVAR_Restricted_Boot.m: obtains the structural
    parameters for the main specification in the third row of Table 2 for each bootstrap
    iteration

4 - Likelihood_SVAR_Restricted_endog.m: obtains the structural
    parameters for the specification in the first row of Table 2

5 - Likelihood_SVAR_Restricted_endog2.m: obtains the structural
    parameters for the specification in the second row of Table 2

6 - Likelihood_SVAR_Restricted_Emp.m: obtains the structural
    parameters for the specification used with DataSet_Emp.txt

7 - DoornikHansenTest.m: obtains the Doornik-Hansen Omnibus Multivariate Normality Test 
    for reduced form residuals (Table 1)

8 - BreuschGodfreyTest.m: obtains the Breusch-Godfrey autocorrelation test
    for reduced form residuals (Table 1)

9 - findNegativePeaks.m: obtains the highest and significant negative peaks for the impulse 
    response functions presented in Table 3
