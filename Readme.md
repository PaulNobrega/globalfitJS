# GlobalFit.JS: Levenberg-Marquardt Global Fitter

  

A pure JavaScript library for performing non-linear least squares curve fitting on multiple datasets simultaneously (global fitting) using the Levenberg-Marquardt algorithm. It supports parameter fixing, linking parameters across datasets, composite models (sum of multiple model functions per dataset), and calculation of common goodness-of-fit statistics.

  

Designed to run natively in modern web browsers or Node.js environments without external numerical dependencies.

  

**Version:** 1.2.0

  

## Features

  

*  **Global Fitting:** Fit multiple datasets simultaneously with shared or independent parameters.

*  **Levenberg-Marquardt Algorithm:** Robust and widely used algorithm for non-linear least squares.

*  **Parameter Fixing:** Keep specific parameters constant during the fit.

*  **Parameter Linking:** Force parameters across different datasets or models to share the same fitted value.

*  **Composite Models:** Define the model for a dataset as the sum of multiple individual model functions.

*  **Pure JavaScript:** No external numerical library dependencies (uses native SVD implementation adapted from Fortran).

*  **Goodness-of-Fit Statistics:** Calculates Degrees of Freedom, Reduced Chi-Squared, AIC, AICc, and BIC.

*  **Custom Logging:** Provides an option (`onLog` callback) for users to handle verbose fitting output.

*  **Parameter Constraints:** Supports simple box constraints (min/max) on parameters.

*  **Robust Cost Functions:** Optional use of L1 (Absolute Residual) or Lorentzian cost functions for outlier resistance.

  

## Advantages

  

*  **Improved Parameter Estimation:** Global fitting uses information from all datasets simultaneously, often leading to more precise and reliable parameter estimates, especially for shared parameters.

*  **Model Discrimination:** Allows testing hypotheses where certain parameters are expected to be the same across different experimental conditions (datasets).

*  **Flexibility:** Handles complex scenarios with multiple model components contributing to the overall signal for each dataset.

*  **Portability:** Runs directly in the browser or Node.js without requiring complex build steps or external libraries for the core numerics.

  

## Installation / Usage

  

### Browser (CDN or Local File)

  

1.  **Include the library:** Download the library files (`svd.js` and `globalfit.js`) or the combined/minified version (`lm-fitter.js` - you would create this via concatenation/minification). Include them in your HTML file. **Order matters if using separate files: `svd.js` must come before `globalfit.js`.**

  

```html

<!DOCTYPE  html>

<html>

<head>

<title>My Fitting App</title>

</head>

<body>

<!-- ... your HTML content ... -->

  

<!-- Load the library files -->

<script  src="path/to/svd.js"></script>

<script  src="path/to/globalfit.js"></script>

<!-- OR -->

<!-- <script src="path/to/lm-fitter.js"></script> -->

<!-- OR -->

<!-- <script src="https://your-cdn-path/lm-fitter.js"></script> -->

  

<script>

// Your code to use lmFitGlobal here

</script>

</body>

</html>

```

  

2.  **Call the function:** The main fitting function `lmFitGlobal` will be available on the global scope (`window.lmFitGlobal`).

  

### Node.js (Basic Example)

  

While primarily designed for browsers, you could adapt it for Node.js. The IIFE structure makes the global object `global`.

  

```javascript

// main_node_script.js

require('./svd.js'); // Loads helpers onto global.LM_SVD_HELPERS

require('./globalfit.js'); // Loads lmFitGlobal onto global.lmFitGlobal

  

// Define your data, models, etc.

const  data = { /* ... */ };

const  modelFunction = [ /* ... */ ];

const  initialParameters = [ /* ... */ ];

const  options = { /* ... */ };

  

// Run the fit

const  result = lmFitGlobal(data, modelFunction, initialParameters, options);

  

console.log(result);

```

*(Note: For more robust Node.js usage, refactoring to use CommonJS or ES Modules would be preferable.)*

  

## API Reference

  

### `lmFitGlobal(data, modelFunction, initialParameters, options)`

  

Performs the global fit.

  

**Parameters:**

  

*  `data` (`object`): Contains the experimental data. Requires properties:

*  `x` (`number[][]`): Array of arrays of independent variable values. `data.x[dsIdx]` is the x-array for dataset `dsIdx`.

*  `y` (`number[][]`): Array of arrays of dependent variable values. `data.y[dsIdx]` corresponds to `data.x[dsIdx]`.

*  `ye` (`number[][]`): Array of arrays of error/uncertainty values for `y`. `data.ye[dsIdx]` corresponds to `data.y[dsIdx]`. Assumed to be standard deviations. **Must not contain zeros.**

*  `modelFunction` (`Function[][]`): Array of arrays of model functions. `modelFunction[dsIdx]` is an array of functions for dataset `dsIdx`.

* Each individual function `func = modelFunction[dsIdx][paramGroupIdx]` represents a component of the total model for that dataset.

* It will be called as `func(paramArray, xArrayPoint)`, where `paramArray` is the corresponding parameter array from `initialParameters[dsIdx][paramGroupIdx]` (e.g., `[val1, val2]`) and `xArrayPoint` is a single-element array containing the current x-value (e.g., `[xValue]`).

* Each function **must** return a single-element array containing the calculated y-value for that component at that x-value (e.g., `[yValue]`).

* The results of all functions in `modelFunction[dsIdx]` are summed internally to get the final model value for dataset `dsIdx` at a given point.

*  `initialParameters` (`number[][][]`): Nested array of initial parameter guesses. The structure must align with `modelFunction`.

*  `initialParameters[dsIdx]` holds the parameter groups for dataset `dsIdx`.

*  `initialParameters[dsIdx][paramGroupIdx]` is an array holding the parameter value(s) needed by `modelFunction[dsIdx][paramGroupIdx]`. Example: `[[m], [c]]` or `[[amp, center, std], [offset]]`.

*  `options` (`object`, optional): Configuration object for the fit.

*  `maxIterations` (`number`, default: `100`): Maximum number of LM iterations.

*  `errorTolerance` (`number`, default: `1e-6`): Convergence threshold based on the fractional change in chi-squared.

*  `gradientTolerance` (`number`, default: `1e-6`): Convergence threshold based on the maximum absolute component of the gradient vector.

*  `linkMap` (`Array<Array<[[number, number], number]>>`, default: `[]`): Defines parameter linking. Each inner array is a group of parameters that should share the same value. Coordinates are `[[datasetIndex, modelGroupIndex], parameterIndexInGroup]`. The first non-fixed parameter in a group acts as the master.

*  `fixMap` (`boolean[][][]`, default: `null`): Defines fixed parameters. Must match the structure of `initialParameters`. `fixMap[dsIdx][paramGroupIdx][paramIndexInGroup] = true` fixes the corresponding parameter. If `null`, all parameters are initially considered free (unless linked to a fixed master).

*  `constraints` (`object[][][]`, default: `null`): Defines parameter box constraints. Matches `initialParameters` structure. Use objects like `{ min: 0, max: 10 }`. `constraints[dsIdx][paramGroupIdx][paramIndexInGroup] = { min: 0 }`.

*  `verbose` (`boolean`, default: `true`): If `true`, enables logging via the `onLog` callback during the fit.

*  `onLog` (`Function`, default: `()=>{}`): A callback function to handle log messages. Receives `(message: string, level: 'info'|'warn'|'error'|'debug')`. Allows routing logs to the console, a DOM element, etc.

*  `robustCostFunction` (`number|null`, default: `null`): Selects the cost function:

*  `null`: Standard Least Squares (minimizes sum of `(residual)^2`). Assumes Gaussian noise.

*  `1`: Absolute Residual / L1 Norm (minimizes sum of `|residual|`). More robust to outliers, assumes double-exponential noise distribution.

*  `2`: Lorentzian / Cauchy-like (minimizes sum of `log(1 + 0.5 * residual^2)`). Highly robust to outliers, assumes Lorentzian/Cauchy noise.

*  `lambdaInitial` (`number`, default: `1e-3`): Initial value for the Levenberg-Marquardt damping parameter lambda.

*  `lambdaIncreaseFactor` (`number`, default: `10`): Factor by which lambda is increased when a step is rejected.

*  `lambdaDecreaseFactor` (`number`, default: `10`): Factor by which lambda is decreased when a step is accepted.

*  `epsilon` (`number`, default: `1e-8`): Step size factor used for numerical differentiation when calculating the Jacobian matrix.

  

**Returns:**

  

*  `object`: An object containing the fitting results:

*  `p_active` (`number[]`): Array of the final optimized values for the *active* parameters only (not fixed, not slaves).

*  `p_reconstructed` (`number[][][]`): The full parameter structure (matching `initialParameters`) with final fitted, fixed, and linked values populated.

*  `chiSquared` (`number`): The final value of the cost function (Chi-Squared or robust equivalent).

*  `covarianceMatrix` (`number[][]`): The estimated covariance matrix for the *active* parameters. Dimensions are `K x K`. Can contain `NaN` if Hessian was ill-conditioned.

*  `parameterErrors` (`number[]`): Array of estimated standard errors for the *active* parameters (square root of the diagonal of `covarianceMatrix`). Can contain `NaN`.

*  `iterations` (`number`): The number of iterations performed.

*  `converged` (`boolean`): `true` if the fit converged based on tolerances, `false` otherwise.

*  `activeParamLabels` (`string[]`): Array of labels identifying each parameter in `p_active` and `parameterErrors` by its original coordinate string (e.g., `"ds0_p1_v0"`).

*  `error` (`string | null`): An error message string if the fit failed prematurely, otherwise `null`.

*  `totalPoints` (`number`): Total number of data points used (N).

*  `degreesOfFreedom` (`number`): Degrees of Freedom (DoF = N - K). `NaN` if N <= K.

*  `reducedChiSquared` (`number`): Reduced Chi-Squared (chiSquared / DoF). `NaN` if DoF <= 0.

*  `aic` (`number`): Akaike Information Criterion (`chiSquared + 2*K`). Useful for model comparison (lower is better).

*  `aicc` (`number`): Corrected AIC (`aic + (2K(K+1))/(N-K-1)`). Better for small N. `Infinity` if N <= K+1.

*  `bic` (`number`): Bayesian Information Criterion (`chiSquared + K*ln(N)`). Useful for model comparison (lower is better), penalizes parameters more heavily than AIC.

  

## Example Usage

  

```javascript

// --- 1. Define Model Functions ---

// Must accept params=[p1, p2,...] and x=[xValue], return [yValue]

function  gaussianModel(params, x) {

const [amp, center, stddev] = params;

const  val = x[0];

if (stddev === 0) return [NaN];

const  exponent = -0.5 * Math.pow((val - center) / stddev, 2);

if (Math.abs(exponent) > 700) return [0.0];

return [amp * Math.exp(exponent)];

}

  

function  linearModel(params, x) {

const [slope, intercept] = params;

return [slope * x[0] + intercept];

}

  

// --- 2. Prepare Data ---

const  data = {

x: [

[1, 2, 3, 4, 5, 6], // Dataset 0

[0, 1, 2, 3, 4, 5, 6, 7] // Dataset 1

],

y: [ // Example noisy data

[5.1, 8.2, 9.9, 10.1, 8.5, 5.3], // Noisy Gaussian

[1.9, 4.1, 5.9, 8.1, 10.0, 12.1, 13.8, 16.2] // Noisy Linear

],

ye: [ // Errors (e.g., standard deviations)

[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],

[0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]

]

};

  

// --- 3. Define Model Structure ---

// Dataset 0: Gaussian only

// Dataset 1: Linear only

const  modelFunction = [

[gaussianModel], // Model group 0 for dataset 0

[linearModel] // Model group 0 for dataset 1

];

  

// --- 4. Initial Parameter Guesses ---

// Structure matches modelFunction

const  initialParameters = [

// Dataset 0: Gaussian params [amp, center, stddev]

[ [9.0, 3.5, 1.0] ], // Params for modelFunction[0][0]

// Dataset 1: Linear params [slope, intercept]

[ [1.8, 2.5] ] // Params for modelFunction[1][0]

];

  

// --- 5. Define Options (Fixing, Linking, Logging) ---

  

// Example: Fix the Gaussian center (Dataset 0, Model Group 0, Param Index 1)

const  fixMap = [

[ [false, true, false] ], // Fix center for Gaussian in DS 0

[ [false, false] ] // Both linear params free in DS 1

];

  

// Example: Link Gaussian width (DS 0, MG 0, PI 2) to Linear intercept (DS 1, MG 0, PI 1)

// (This might not make physical sense, just demonstrates linking)

const  linkMap = [

// Group 1

[

[[0, 0], 2], // Coord for Gaussian width

[[1, 0], 1] // Coord for Linear intercept

]

];

  

// Example: Custom logger to put output in a DOM element

// const logElement = document.getElementById('my-log-div'); // Assume this exists

function  myLogger(message, level) {

// if (logElement) { ... append to element ... }

// else {

console.log(`[${level.toUpperCase()}] ${message}`); // Fallback

// }

}

  

const  fitOptions = {

maxIterations:  200,

verbose:  true,

onLog:  myLogger,

fixMap:  fixMap,

linkMap:  linkMap

// constraints: constraints, // Add constraints if needed

// robustCostFunction: 1, // Example: Use L1 norm

};

  

// --- 6. Run the Fit ---

try {

// Ensure lmFitGlobal is loaded (e.g., via <script> tag)

if (typeof  lmFitGlobal !== 'function') {

throw  new  Error("lmFitGlobal function not found.");

}

  

const  result = lmFitGlobal(data, modelFunction, initialParameters, fitOptions);

  

// --- 7. Process Results ---

if (result.error) {

console.error("Fit failed:", result.error);

myLogger(`Fit failed: ${result.error}`, 'error');

} else {

console.log("Fit Results:", result);

myLogger(`Fit ${result.converged ? 'converged' : 'did NOT converge'} in ${result.iterations} iterations.`, 'info');

myLogger(`Final Chi^2: ${result.chiSquared?.toExponential(5)}`, 'info');

myLogger(`Reduced Chi^2: ${result.reducedChiSquared?.toExponential(5)}`, 'info');

myLogger(`AIC: ${result.aic?.toExponential(5)}`, 'info');

myLogger(`BIC: ${result.bic?.toExponential(5)}`, 'info');

  

// Access final parameters

console.log("Final Active Parameters:", result.p_active);

console.log("Active Parameter Labels:", result.activeParamLabels);

console.log("Active Parameter Errors:", result.parameterErrors);

console.log("Full Reconstructed Parameters:", result.p_reconstructed);

  

// Example: Get the final linked value (width/intercept in this example)

// Find the label corresponding to the master parameter (Gaussian width was first in linkMap)

const  masterLabel = "ds0_p0_v2";

const  masterIndex = result.activeParamLabels.indexOf(masterLabel);

if (masterIndex !== -1) {

const  linkedValue = result.p_active[masterIndex];

const  linkedError = result.parameterErrors[masterIndex];

console.log(`Final linked value (width/intercept): ${linkedValue} +/- ${linkedError}`);

// You can also verify this in result.p_reconstructed[0][0][2] and result.p_reconstructed[1][0][1]

} else {

// The master parameter might have been fixed itself

console.log("Linked parameter master was not active (likely fixed). Value:", result.p_reconstructed[0][0][2]);

}

// Get the fixed Gaussian center value

console.log("Fixed Gaussian center:", result.p_reconstructed[0][0][1]);

}

} catch (e) {

console.error("Error during fitting process:", e);

myLogger(`Runtime Error: ${e.message}`, 'error');

}

  

```

  

## Notes & Considerations

  

*  **SVD Implementation:** The native SVD is adapted from Fortran code. While functional, numerical linear algebra can be complex. For highly critical applications, consider comparing results or potentially integrating with a more rigorously tested library if the environment allows. The current implementation assumes `m >= n` (more data points than parameters per dataset, generally true for fitting).

*  **Error Estimation:** Parameter errors (`parameterErrors`) are derived from the diagonal of the covariance matrix (`sqrt(C[i][i])`). `NaN` errors indicate that the uncertainty could not be reliably estimated, usually due to parameter correlations or insufficient data (ill-conditioned Hessian). This doesn't necessarily mean the parameter *values* are wrong, just that their individual uncertainty is poorly defined by the data.

*  **Model Function Signature:** Remember the specific signature required for model functions: `func(paramArray, xArrayPoint)` returning `[yValue]`.

*  **Performance:** For very large datasets or a very high number of active parameters, performance in the browser might degrade due to the single-threaded nature of JavaScript. Consider Web Workers for computationally intensive fitting tasks in complex UIs.

*  **Robust Cost Functions:** Using `robustCostFunction: 1` or `2` changes the meaning of `chiSquared`. It's no longer strictly the sum of squared normalized residuals. The parameter values obtained will be Maximum Likelihood Estimates under the assumed noise distribution (double-exponential or Lorentzian, respectively), but interpreting the absolute value of the final "chiSquared" for goodness-of-fit requires care. Reduced Chi-Squared is less meaningful in these cases. AIC/BIC based on this modified chi-squared are still useful for *comparing* models fit with the *same* robust cost function.

  

## MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
