# GlobalFit.JS: Levenberg-Marquardt Global Fitter

A pure JavaScript library for performing non-linear least squares curve fitting on multiple datasets simultaneously (global fitting) using the Levenberg-Marquardt algorithm. It supports parameter fixing, linking parameters across datasets using shared IDs, composite models (sum of multiple model functions per dataset), and calculation of common goodness-of-fit statistics.

Designed to run natively in modern web browsers or Node.js environments without external numerical dependencies.

Code has been adapted from the original Fortran project Savuka: https://osmanbilsel.com/software/savuka-a-general-purpose-global-analysis-program/

**Version:** 1.2.7

## What's New in 1.2.7
- **Improved Logging**: Added a robust logging utility that respects the `logLevel` option (`none`, `error`, `warn`, `info`, `debug`).
- **Enhanced Documentation**: Improved inline comments and organized code for better readability.
- **Performance Optimizations**: Streamlined handling of large datasets and parameter structures.
- **Simulation Functionality**: Added `simulateFromParams` to generate synthetic datasets based on model functions, parameters, and noise options (Gaussian or Poisson).
- **Confidence Interval Support**: Added functionality to calculate confidence intervals for fitted model curves using the covariance matrix and Student's t-distribution.
- **Bootstrap Fallback for Confidence Intervals**: Implemented a fallback mechanism to calculate confidence intervals using bootstrapping when the covariance matrix yields negative variances.
- **SVD Fallback for Covariance Matrix**: Improved numerical stability by using SVD-based inversion for the covariance matrix, with regularization to handle ill-conditioned problems.

## CDN Script Tags
```html
  <script src="https://cdn.jsdelivr.net/gh/PaulNobrega/globalfitJS@v1.2.7/dist/globalfit.min.js"></script>
```

## Features

-   **Global Fitting:** Fit multiple datasets simultaneously with shared or independent parameters.
-   **Levenberg-Marquardt Algorithm:** Robust and widely used algorithm for non-linear least squares.
-   **Parameter Fixing:** Keep specific parameters constant during the fit using a `fixMap`.
-   **Parameter Linking:** Force parameters across different datasets or models to share the same fitted value using a `linkMap` with shared string/number IDs (new structure similar to `fixMap`).
-   **Composite Models:** Define the model for a dataset as the sum of multiple individual model functions.
-   **Pure JavaScript:** No external numerical library dependencies (uses native SVD implementation adapted from Fortran).
-   **Goodness-of-Fit Statistics:** Calculates Degrees of Freedom, Reduced Chi-Squared, AIC, AICc, and BIC.
-   **Covariance & Errors:** Returns the covariance matrix and standard errors for active parameters. Uses `Math.abs()` for variance if numerically negative (due to ill-conditioning) and logs a warning.
-   **Covariance Regularization:** Applies a small regularization factor during covariance matrix calculation for improved numerical stability.
-   **Custom Logging:** Provides an option (`onLog` callback) for users to handle verbose fitting output.
-   **Parameter Constraints:** Supports simple box constraints (min/max) and custom constraint functions.
-   **Robust Cost Functions:** Optional use of L1 (Absolute Residual) or Lorentzian cost functions for outlier resistance.
-   **Confidence Intervals:** Calculates confidence intervals for fitted model curves using the covariance matrix and Student's t-distribution.
-   **Bootstrap Fallback for Confidence Intervals:** Automatically falls back to bootstrapping when the covariance matrix yields negative variances.
-   **Simulation Functionality:** Generate synthetic datasets using `simulateFromParams` with support for Gaussian and Poisson noise.

## Advantages

-   **Improved Parameter Estimation:** Global fitting uses information from all datasets simultaneously, often leading to more precise and reliable parameter estimates, especially for shared parameters.
-   **Model Discrimination:** Allows testing hypotheses where certain parameters are expected to be the same across different experimental conditions (datasets).
-   **Flexibility:** Handles complex scenarios with multiple model components contributing to the overall signal for each dataset.
-   **Portability:** Runs directly in the browser or Node.js without requiring complex build steps or external libraries for the core numerics.
-   **Numerical Stability:** Uses SVD-based inversion for the covariance matrix with regularization to handle ill-conditioned problems.
-   **Fallback Mechanisms:** Includes robust fallback mechanisms, such as bootstrapping for confidence intervals and SVD for covariance matrix inversion, ensuring reliable results even in challenging scenarios.

## Installation / Usage

### Browser (CDN or Local File)

1.  **Include the library:** Download the library files (`svd.js` and `globalfit.js`) or use the CDN links. Include them in your HTML file. **Order matters if using separate files: `svd.js` must come before `globalfit.js`.**

    ```html
    <!DOCTYPE html>
    <html>
    <head>
      <title>My Fitting App</title>
    </head>
    <body>
      <!-- ... your HTML content ... -->

      <!-- Load the library files -->
      <!-- Option 1: Separate Files -->
      <!-- <script src="path/to/svd.js"></script> -->
      <!-- <script src="path/to/globalfit.js"></script> -->

      <!-- Option 2: CDN (Recommended) -->
      <script src="https://cdn.jsdelivr.net/gh/PaulNobrega/globalfitJS@v1.2.7/dist/globalfit.min.js"></script>
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
const data = { /* ... */ };
const modelFunction = [ /* ... */ ];
const initialParameters = [ /* ... */ ];
const options = { /* ... */ };

// Run the fit
const result = lmFitGlobal(data, modelFunction, initialParameters, options);

console.log(result);


**Section 5: API Reference (`lmFitGlobal` Parameters)**

```

## API Reference

### `lmFitGlobal(data, modelFunction, initialParameters, options)`

Performs the global fit.

**Parameters:**

-   `data` (`object`): Contains the experimental data. Requires properties:
    -   `x` (`number[][]`): Array of arrays of independent variable values. `data.x[dsIdx]` is the x-array for dataset `dsIdx`.
    -   `y` (`number[][]`): Array of arrays of dependent variable values. `data.y[dsIdx]` corresponds to `data.x[dsIdx]`.
    -   `ye` (`number[][]`): Array of arrays of error/uncertainty values for `y`. `data.ye[dsIdx]` corresponds to `data.y[dsIdx]`. Assumed to be standard deviations. **Must not contain zeros.**
-   `modelFunction` (`Function[][]`): Array of arrays of model functions. `modelFunction[dsIdx]` is an array of functions for dataset `dsIdx`.
    -   Each individual function `func = modelFunction[dsIdx][paramGroupIdx]` represents a component of the total model for that dataset.
    -   It will be called as `func(paramArray, xArrayPoint)`, where `paramArray` is the corresponding parameter array from `initialParameters[dsIdx][paramGroupIdx]` (e.g., `[val1, val2]`) and `xArrayPoint` is a single-element array containing the current x-value (e.g., `[xValue]`).
    -   Each function **must** return a single-element array containing the calculated y-value for that component at that x-value (e.g., `[yValue]`).
    -   The results of all functions in `modelFunction[dsIdx]` are summed internally to get the final model value for dataset `dsIdx` at a given point.
-   `initialParameters` (`number[][][]`): Nested array of initial parameter guesses. The structure must align with `modelFunction`.
    -   `initialParameters[dsIdx]` holds the parameter groups for dataset `dsIdx`.
    -   `initialParameters[dsIdx][paramGroupIdx]` is an array holding the parameter value(s) needed by `modelFunction[dsIdx][paramGroupIdx]`. Example: `[[m], [c]]` or `[[amp, center, std], [offset]]`.
-   `options` (`object`, optional): Configuration object for the fit.
    -   `calculateFittedModel` (`boolean | {numPoints: number}`, default: `false`): If `true` or an object like `{numPoints: 200}`, calculates smooth fitted model curves for each dataset and includes them in the results.
    -   `constraintFunction` (`Function | null`, default: `null`): Custom function `(params) => modifiedParams` applied after box constraints. Input and output must match `initialParameters` structure.
    -   `constraints` (`object[][][]`, default: `null`): Defines parameter box constraints. Matches `initialParameters` structure. Use objects like `{ min: 0, max: 10 }`. `constraints[dsIdx][paramGroupIdx][paramIndexInGroup] = { min: 0 }`.
    -   `covarianceLambda` (`number`, default: `1e-9`): Small regularization factor added to the Hessian diagonal before inversion for calculating the covariance matrix. Helps stabilize inversion for ill-conditioned problems.
    -   `epsilon` (`number`, default: `1e-8`): Step size factor used for numerical differentiation when calculating the Jacobian matrix.
    -   `errorTolerance` (`number`, default: `1e-6`): Convergence threshold based on the fractional change in chi-squared.
    -   `fixMap` (`boolean[][][]`, default: `null`): Defines fixed parameters. Must match the structure of `initialParameters`. `fixMap[dsIdx][paramGroupIdx][paramIndexInGroup] = true` fixes the corresponding parameter. If `null`, all parameters are initially considered free (unless linked to a fixed master).
    -   `gradientTolerance` (`number`, default: `1e-6`): Convergence threshold based on the maximum absolute component of the gradient vector.
    -   `lambdaInitial` (`number`, default: `1e-3`): Initial value for the Levenberg-Marquardt damping parameter lambda.
    -   `lambdaIncreaseFactor` (`number`, default: `10`): Factor by which lambda is increased when a step is rejected.
    -   `lambdaDecreaseFactor` (`number`, default: `10`): Factor by which lambda is decreased when a step is accepted.
    -   `linkMap` (`(number|string|null)[][][]`, default: `null`): **(New Format)** Defines parameter linking. Must match the structure of `initialParameters`. Parameters with the **same non-null, non-empty string or number ID** will be linked together. The first non-fixed parameter encountered with a given ID acts as the master. `null` or `''` indicates the parameter is not linked (unless linked via another parameter).
    -   `logLevel` (`string`, default: `'info'`): Control logging verbosity ('none', 'error', 'warn', 'info', 'debug').
    -   `onLog` (`Function`, default: `()=>{}`): A callback function to handle log messages. Receives `(message: string, level: string)`. Allows routing logs to the console, a DOM element, etc.
    -   `onProgress` (`Function`, default: `()=>{}`): Callback for progress updates during fitting. Receives `(progressData: object)` where `progressData = { iteration, chiSquared, lambda, activeParameters }`.
    -   `maxIterations` (`number`, default: `100`): Maximum number of LM iterations.
    -   `robustCostFunction` (`number|null`, default: `null`): Selects the cost function:
        -   `null`: Standard Least Squares (minimizes sum of `(residual)^2`). Assumes Gaussian noise.
        -   `1`: Absolute Residual / L1 Norm (minimizes sum of `|residual|`). More robust to outliers, assumes double-exponential noise distribution.
        -   `2`: Lorentzian / Cauchy-like (minimizes sum of `log(1 + 0.5 * residual^2)`). Highly robust to outliers, assumes Lorentzian/Cauchy noise.
    -   `confidenceInterval` (`number`, default: `null`): Confidence level for calculating confidence intervals (e.g., `0.95` for 95% confidence intervals).
    -   `numBootstrapSamples` (`number`, default: `200`): Number of bootstrap samples to use for confidence interval calculation if bootstrapping is enabled.
    -   `bootstrapFallback` (`boolean`, default: `true`): Enable or disable bootstrap fallback for confidence intervals when the covariance matrix yields negative variances.

**Returns:**

-   `object`: An object containing the fitting results:
    -   `p_active` (`number[]`): Array of the final optimized values for the *active* parameters only (not fixed, not slaves).
    -   `p_reconstructed` (`number[][][]`): The full parameter structure (matching `initialParameters`) with final fitted, fixed, and linked values populated.
    -   `finalParamErrors` (`(number|null)[][][]`): Estimated standard errors for all parameters (matching `initialParameters` structure). Contains `NaN` if calculation failed (e.g., variance was non-finite), `0` for fixed parameters. Uses `sqrt(abs(variance))` if variance was numerically negative (warning logged).
    -   `chiSquared` (`number`): The final value of the cost function (Chi-Squared or robust equivalent).
    -   `covarianceMatrix` (`number[][] | null`): The estimated covariance matrix for the *active* parameters (potentially regularized). Dimensions are `K x K`. `null` if K=0 or inversion failed.
    -   `parameterErrors` (`number[]`): Array of estimated standard errors for the *active* parameters (`sqrt(abs(variance * scaleFactor))`). Can contain `NaN` if the variance itself was non-finite. A warning is logged if negative variance was encountered.
    -   `iterations` (`number`): The number of iterations performed.
    -   `converged` (`boolean`): `true` if the fit converged based on tolerances, `false` otherwise.
    -   `activeParamLabels` (`string[]`): Array of labels identifying each parameter in `p_active` and `parameterErrors` by its original coordinate string (e.g., `"ds0_p1_v0"`).
    -   `error` (`string | null`): An error message string if the fit failed prematurely, otherwise `null`.
    -   `totalPoints` (`number`): Total number of data points used (N).
    -   `degreesOfFreedom` (`number`): Degrees of Freedom (DoF = N - K). `NaN` if N <= K.
    -   `reducedChiSquared` (`number`): Chi-Squared divided by degrees of freedom. `NaN` or `Infinity` if DoF <= 0.
    -   `aic` (`number`): Akaike Information Criterion (`chiSquared + 2*K`). Useful for model comparison (lower is better).
    -   `aicc` (`number`): Corrected AIC (`aic + (2K(K+1))/(N-K-1)`). Better for small N. `Infinity` if N <= K+1.
    -   `bic` (`number`): Bayesian Information Criterion (`chiSquared + K*ln(N)`). Useful for model comparison (lower is better), penalizes parameters more heavily than AIC.
    -   `residualsPerSeries` (`number[][] | null`): Weighted residuals ((y-ymodel)/ye) for each dataset. `null` on failure.
    -   `fittedModelCurves` (`{x: number[], y: number[]}[] | null`): Calculated fitted model curves for each dataset if `options.calculateFittedModel` was set. `null` otherwise or on failure.
    -   `ci_lower` (`{x: number[], y: number[]}[] | null`): Lower bounds of the confidence intervals for each dataset if `options.confidenceInterval` was set. `null` otherwise or on failure.
    -   `ci_upper` (`{x: number[], y: number[]}[] | null`): Upper bounds of the confidence intervals for each dataset if `options.confidenceInterval` was set. `null` otherwise or on failure.

### `lmFit(data, modelFunction, initialParameters, options)`

A convenience wrapper for fitting a **single** dataset.

-   Accepts `data` as `{x: number[], y: number[], ye: number[]}`.
-   Accepts `modelFunction` as `Function | Function[]`.
-   Accepts `initialParameters` as `number[] | number[][]`.
-   Accepts `options` like `lmFitGlobal`, but `fixMap`, `linkMap`, and `constraints` should be provided in the single-dataset format (e.g., `fixMap = [[false, true], [false]]`, `linkMap = [[null, 'id1'], ['id1']]`).
-   Returns the same result object structure as `lmFitGlobal`.

### `lmFitIndependent(data, modelFunction, initialParameters, options)`

Fits multiple datasets **independently** by calling `lmFitGlobal` sequentially for each dataset.

-   Accepts `data`, `modelFunction`, `initialParameters` in the same multi-dataset format as `lmFitGlobal`.
-   Accepts most `options` like `lmFitGlobal`.
-   `linkMap` is **ignored** as linking only makes sense between datasets in a global fit.
-   `fixMap` and `constraints` apply per-dataset if provided in the full nested structure.
-   `onLog` and `onProgress` callbacks receive an additional `datasetIndex` argument.
-   Returns an **array** of result objects, one for each dataset fit.

## Example Usage

```javascript
// --- 1. Define Model Functions ---
// Must accept params=[p1, p2,...] and x=[xValue], return [yValue]

function gaussianModel(params, x) {
  const [amp, center, stddev] = params;
  const val = x[0];
  if (stddev === 0) return [NaN];
  const exponent = -0.5 * Math.pow((val - center) / stddev, 2);
  if (Math.abs(exponent) > 700) return [0.0];
  return [amp * Math.exp(exponent)];
}

function linearModel(params, x) {
  const [slope, intercept] = params;
  return [slope * x[0] + intercept];
}

// --- 2. Prepare Data ---
// inputs are 2-d lists where each row corresponds to a data series
const data = {
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
const modelFunction = [
  [gaussianModel], // Model group 0 for dataset 0
  [linearModel] // Model group 0 for dataset 1
];

// --- 4. Initial Parameter Guesses ---
// Structure matches modelFunction
const initialParameters = [
  // Dataset 0: Gaussian params [amp, center, stddev]
  [ [9.0, 3.5, 1.0] ], // Params for modelFunction[0][0]
  // Dataset 1: Linear params [slope, intercept]
  [ [1.8, 2.5] ] // Params for modelFunction[1][0]
];

// --- 5. Define Options (Fixing, Linking, Logging) ---

// Example: Fix the Gaussian center (Dataset 0, Model Group 0, Param Index 1)
const fixMap = [
  [ [false, true, false] ], // Fix center for Gaussian in DS 0
  [ [false, false] ] // Both linear params free in DS 1
];

// Example: Link Gaussian width (DS 0, MG 0, PI 2) to Linear intercept (DS 1, MG 0, PI 1)
// Use the NEW nested array structure with shared IDs
const linkMap_newFormat = [
    // Dataset 0
    [
        // Model 0 (Gaussian) params: [Amp, Center, StdDev]
        [null, null, "shared_param"] // Link StdDev
    ],
    // Dataset 1
    [
        // Model 0 (Linear) params: [Slope, Intercept]
        [null, "shared_param"] // Link Intercept
    ]
];


// Example: Custom logger
function myLogger(message, level) {
  console.log(`[${level.toUpperCase()}] ${message}`);
}

const fitOptions = {
  maxIterations:  200,
  logLevel: 'info', // Use 'debug' for more details
  onLog:  myLogger,
  fixMap:  fixMap,
  linkMap:  linkMap_newFormat, // Pass the new linkMap
  covarianceLambda: 1e-9, // Optional regularization
  confidenceInterval: 0.95, // 95% confidence intervals
  numBootstrapSamples: 200, // Number of bootstrap samples for fallback
  bootstrapFallback: true   // Enable bootstrap fallback
  // constraints: constraints, // Add constraints if needed
  // robustCostFunction: 1, // Example: Use L1 norm
};

// --- 6. Run the Fit ---
try {
  if (typeof  lmFitGlobal !== 'function') { throw new Error("lmFitGlobal function not found."); }

  const result = lmFitGlobal(data, modelFunction, initialParameters, fitOptions);

  // --- 7. Process Results ---
  if (result.error) {
    console.error("Fit failed:", result.error);
  } else {
    console.log("Fit Results:", result);
    myLogger(`Fit ${result.converged ? 'converged' : 'did NOT converge'} in ${result.iterations} iterations.`, 'info');
    myLogger(`Final Chi^2: ${result.chiSquared?.toExponential(5)}`, 'info');

    // Access final parameters
    console.log("Final Active Parameters:", result.p_active);
    console.log("Active Parameter Labels:", result.activeParamLabels);
    console.log("Active Parameter Errors:", result.parameterErrors);
    console.log("Full Reconstructed Parameters:", result.p_reconstructed);
    console.log("Final Parameter Errors (Full):", result.finalParamErrors);
    console.log("Covariance Matrix (Active):", result.covarianceMatrix);

    // Example: Get the final linked value (width/intercept in this example)
    // Check the reconstructed parameters directly
    const linkedVal1 = result.p_reconstructed[0][0][2]; // Gaussian width
    const linkedVal2 = result.p_reconstructed[1][0][1]; // Linear intercept
    console.log(`Final linked value check: DS0 Width=${linkedVal1}, DS1 Intercept=${linkedVal2}`);

    // Get the fixed Gaussian center value
    console.log("Fixed Gaussian center:", result.p_reconstructed[0][0][1]);

    // Access confidence intervals
    console.log("Confidence Intervals (Lower):", result.ci_lower);
    console.log("Confidence Intervals (Upper):", result.ci_upper);
  }
} catch (e) {
  console.error("Error during fitting process:", e);
}
```

## Notes & Considerations

-   **SVD Implementation:** The native SVD is adapted from Fortran code. While functional, numerical linear algebra can be complex. For highly critical applications, consider comparing results or potentially integrating with a more rigorously tested library if the environment allows. The current implementation assumes `m >= n` (more data points than parameters per dataset, generally true for fitting).
-   **Error Estimation:** Parameter errors (`parameterErrors`, `finalParamErrors`) are derived from the diagonal of the covariance matrix. The calculation uses `Math.sqrt(Math.abs(variance * scaleFactor))` to avoid `NaN` output when numerical instability leads to small negative variances for ill-conditioned problems, but a warning is logged in such cases, indicating the error estimate might be less reliable. `NaN` can still occur if the variance itself is non-finite (e.g., due to complete failure of matrix inversion). Analyzing the `covarianceMatrix` and parameter correlations is recommended if `NaN` or warnings appear frequently.
-   **Covariance Matrix:** The returned `covarianceMatrix` is calculated after applying a small regularization (`options.covarianceLambda`) to the Hessian diagonal before inversion. This improves stability but slightly modifies the true covariance.
-   **Model Function Signature:** Remember the specific signature required for model functions: `func(paramArray, xArrayPoint)` returning `[yValue]`.
-   **Performance:** For very large datasets or a very high number of active parameters, performance in the browser might degrade due to the single-threaded nature of JavaScript. Consider Web Workers for computationally intensive fitting tasks in complex UIs.
-   **Robust Cost Functions:** Using `robustCostFunction: 1` or `2` changes the meaning of `chiSquared`. It's no longer strictly the sum of squared normalized residuals. The parameter values obtained will be Maximum Likelihood Estimates under the assumed noise distribution (double-exponential or Lorentzian, respectively), but interpreting the absolute value of the final "chiSquared" for goodness-of-fit requires care. Reduced Chi-Squared is less meaningful in these cases. AIC/BIC based on this modified chi-squared are still useful for *comparing* models fit with the *same* robust cost function.

## MIT License

Copyright (c) 2025 R. Paul Nobrega 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.