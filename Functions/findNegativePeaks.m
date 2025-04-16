function [peakValue, peakTime] = findNegativePeaks(data, lowerBound, upperBound)
    
    peakValue = NaN;
    peakTime = NaN;

    % Create a logical index array that identifies where data points should be considered:
    % - Data points must be negative.
    % - Data points must not have zero between their corresponding lower and upper bounds.
    validIndices = (data < 0) & ~(lowerBound < 0 & upperBound > 0);

    % Check if there are any valid data points to consider
    if any(validIndices)
        % Restrict the data to valid indices for finding the peak
        validData = data(validIndices);
        
        % Find the minimum value within the valid data
        [peakValue, peakIndex] = min(validData);
        
        % Convert the index in 'validData' back to the original 'data' index
        originalIndices = find(validIndices);
        peakTime = originalIndices(peakIndex);
    end

    % If no valid data points were found, peakValue and peakTime remain NaN.
end


