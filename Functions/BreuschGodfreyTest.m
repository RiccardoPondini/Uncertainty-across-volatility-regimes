function [pValueBG, LMStatBG] = BreuschGodfreyTest(residuals)
    lags = 5;  % Number of lags to test
    T = size(residuals, 1);  % number of observations
    numVariables = size(residuals, 2);  % number of series in VAR

    % Preallocate output arrays
    pValueBG = zeros(numVariables, 1);
    LMStatBG = zeros(numVariables, 1);

    % Loop through each series of residuals
    for v = 1:numVariables
        % Initialize the design matrix for lags with an intercept
        X = ones(T-lags, 1);  % Start with an intercept

        % Add lagged variables to the design matrix
        for lag = 1:lags
            laggedData = lagmatrix(residuals(:, v), lag);
            % Exclude initial rows which contain NaN due to lagging
            X = [X, laggedData(lags+1:end, :)];
        end

        % Adjust Y to match the size of X after removing NaN rows
        Y = residuals(lags+1:end, v);

        % Run regression: Y on X
        [B, ~, R] = regress(Y, X);

        % Calculate residuals and R-squared
        SSR = R' * R;
        SST = sum((Y - mean(Y)).^2);
        R2 = 1 - SSR/SST;

        % Calculate the LM statistic for each series
        LMStatBG(v) = (T - lags) * R2;
        df = lags;  % Degrees of freedom for each series

        % Calculate the p-value from the chi-square distribution
        pValueBG(v) = 1 - chi2cdf(LMStatBG(v), df);
    end
end