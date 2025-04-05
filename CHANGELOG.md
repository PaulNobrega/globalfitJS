# Changelog

## Version 1.2.5
- Updated parameter error calculation to use `Math.abs` for standard error when encountering negative variance
- Added covariance regularization with `covarianceLambda` parameter to improve numerical stability
- Fixed structure of `linkMap` parameter to properly handle parameter linking
- Improved error handling and reporting during fitting process
- Enhanced documentation and code comments

## Version 1.2.0
- Initial public release
- Implemented Levenberg-Marquardt algorithm for global curve fitting
- Support for multiple datasets and parameter linking
- Included statistics calculation (Chi-squared, AIC, BIC)