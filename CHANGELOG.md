# Changelog

## Version 1.2.7
- **Refactored Logging**: Replaced `console.log` statements with a robust logging utility that respects the `logLevel` option.
- **Improved Documentation**: Enhanced inline comments and function documentation for clarity.
- **Organized Code**: Grouped related functions and constants for better readability and maintainability.
- **Bug Fixes**: Addressed minor issues in parameter mapping and error handling.
- **Performance Improvements**: Optimized the handling of large datasets and parameter structures.
- **Added Simulation Functionality**: Introduced `simulateFromParams`, a utility to generate synthetic datasets based on model functions, parameters, and noise options (Gaussian or Poisson).
- **Confidence Interval Support**: Added functionality to calculate confidence intervals for fitted model curves using the covariance matrix and Student's t-distribution.
- **Bootstrap Fallback for Confidence Intervals**: Implemented a fallback mechanism to calculate confidence intervals using bootstrapping when the covariance matrix yields negative variances.
- **SVD Fallback for Covariance Matrix**: Improved numerical stability by using SVD-based inversion for the covariance matrix, with regularization to handle ill-conditioned problems.

## Version 1.2.5
- Updated parameter error calculation to use `Math.abs` for standard error when encountering negative variance.
- Added covariance regularization with `covarianceLambda` parameter to improve numerical stability.
- Fixed structure of `linkMap` parameter to properly handle parameter linking.
- Improved error handling and reporting during fitting process.
- Enhanced documentation and code comments.

## Version 1.2.0
- Initial public release.
- Implemented Levenberg-Marquardt algorithm for global curve fitting.
- Support for multiple datasets and parameter linking.
- Included statistics calculation (Chi-squared, AIC, BIC).