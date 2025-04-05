// svd.js

/**
 * @fileoverview Provides linear algebra helper functions, including SVD decomposition,
 * matrix inversion using SVD, matrix multiplication, and transposition.
 * Designed for use with the lmFitGlobal fitting library.
 * Uses native JavaScript, adapted from Fortran numerical recipes logic.
 * Version: 1.0.0
 */

(function(global) {
    'use strict';

    /**
     * Solves the linear system Ax = b using Gaussian elimination with partial pivoting.
     * @param {number[][]} matrix - The matrix A (n x n).
     * @param {number[]} vector - The vector b (n).
     * @returns {number[]} The solution vector x (n).
     * @throws {Error} If the matrix is invalid, singular, or numerical issues occur.
     * @private
     */
    function solveLinearSystem(matrix, vector) {
        const n = vector.length;
        if (!matrix || matrix.length !== n || matrix.some(row => !row || row.length !== n)) {
            throw new Error(`Invalid matrix or vector size for solveLinearSystem. Matrix: ${matrix?.length}x${matrix?.[0]?.length}, Vector: ${n}`);
        }
        const augmentedMatrix = matrix.map((row, i) => [...row, vector[i]]);
        for (let i = 0; i < n; i++) {
            let maxRow = i;
            for (let k = i + 1; k < n; k++) {
                if (Math.abs(augmentedMatrix[k][i]) > Math.abs(augmentedMatrix[maxRow][i])) maxRow = k;
            }
            [augmentedMatrix[i], augmentedMatrix[maxRow]] = [augmentedMatrix[maxRow], augmentedMatrix[i]];
            if (Math.abs(augmentedMatrix[i][i]) < 1e-12) {
                let foundPivot = false;
                for(let k = i + 1; k < n; k++) {
                    if (Math.abs(augmentedMatrix[k][i]) >= 1e-12) {
                        [augmentedMatrix[i], augmentedMatrix[k]] = [augmentedMatrix[k], augmentedMatrix[i]];
                        foundPivot = true; break;
                    }
                }
                if (!foundPivot) throw new Error(`Matrix is singular or near-singular during Gaussian elimination at step ${i}.`);
            }
            for (let k = i + 1; k < n; k++) {
                const factor = augmentedMatrix[k][i] / augmentedMatrix[i][i];
                if (!isFinite(factor)) throw new Error(`Non-finite factor encountered during elimination at [${k},${i}]. Pivot: ${augmentedMatrix[i][i]}`);
                augmentedMatrix[k][i] = 0;
                for (let j = i + 1; j <= n; j++) augmentedMatrix[k][j] -= factor * augmentedMatrix[i][j];
            }
        }
        const solution = new Array(n).fill(0);
        for (let i = n - 1; i >= 0; i--) {
            if (Math.abs(augmentedMatrix[i][i]) < 1e-12) throw new Error(`Zero pivot encountered during back substitution at step ${i}.`);
            let sum = 0;
            for (let j = i + 1; j < n; j++) sum += augmentedMatrix[i][j] * solution[j];
            solution[i] = (augmentedMatrix[i][n] - sum) / augmentedMatrix[i][i];
            if (!isFinite(solution[i])) throw new Error(`Non-finite solution component encountered at index ${i}. Pivot: ${augmentedMatrix[i][i]}`);
        }
        return solution;
    }

    /**
     * Performs Singular Value Decomposition (SVD) on a matrix using a native JavaScript implementation.
     * Adapted from Fortran SVDCMP logic. Decomposes matrix A such that A = U * S * V^T. Assumes m >= n.
     * @param {number[][]} matrix - The input matrix A (m x n).
     * @returns {{u: number[][], s: number[], v: number[][]}} Object containing U (m x n), S (vector n), V (n x n).
     * @throws {Error} If convergence fails.
     * @private
     */
    function svdDecompose(matrix) {
        const a = matrix.map(row => [...row]);
        const m = a.length; if (m === 0) return { u: [], s: [], v: [] };
        const n = a[0].length; if (n === 0) return { u: a.map(() => []), s: [], v: [] };
        // Warning only, attempt to proceed if m < n
        if (m < n) console.warn("svdDecompose warning: Implementation assumes m >= n. Results may be incorrect.");

        const w = new Array(n).fill(0.0);
        const v = Array.from({ length: n }, () => new Array(n).fill(0.0));
        const rv1 = new Array(n).fill(0.0);
        let g = 0.0, scale = 0.0, anorm = 0.0;

        // Householder reduction
        for (let i = 0; i < n; i++) {
            scale = 0.0; for (let k = i; k < m; k++) scale += Math.abs(a[k][i]);
            if (scale !== 0.0) {
                let s = 0.0; for (let k = i; k < m; k++) { a[k][i] /= scale; s += a[k][i] * a[k][i]; }
                let f = a[i][i]; g = -Math.sign(f) * Math.sqrt(s); let h = f * g - s; a[i][i] = f - g;
                for (let j = i + 1; j < n; j++) { s = 0.0; for (let k = i; k < m; k++) s += a[k][i] * a[k][j]; f = s / h; for (let k = i; k < m; k++) a[k][j] += f * a[k][i]; }
                for (let k = i; k < m; k++) a[k][i] *= scale;
            }
            w[i] = scale * g; g = 0.0; scale = 0.0;
            if (i < n - 1) {
                for (let k = i + 1; k < n; k++) scale += Math.abs(a[i][k]);
                if (scale !== 0.0) {
                    let s = 0.0; for (let k = i + 1; k < n; k++) { a[i][k] /= scale; s += a[i][k] * a[i][k]; }
                    let f = a[i][i + 1]; g = -Math.sign(f) * Math.sqrt(s); let h = f * g - s; a[i][i + 1] = f - g;
                    for (let k = i + 1; k < n; k++) rv1[k] = a[i][k] / h;
                    for (let j = i + 1; j < m; j++) { s = 0.0; for (let k = i + 1; k < n; k++) s += a[j][k] * a[i][k]; for (let k = i + 1; k < n; k++) a[j][k] += s * rv1[k]; }
                    for (let k = i + 1; k < n; k++) a[i][k] *= scale;
                }
            }
            anorm = Math.max(anorm, (Math.abs(w[i]) + Math.abs(g))); rv1[i] = g;
        }
        // Accumulation of V
        for (let i = n - 1; i >= 0; i--) {
            if (i < n - 1) { if (rv1[i] !== 0.0) { for (let j = i + 1; j < n; j++) { let s = 0.0; for (let k = i + 1; k < n; k++) s += a[i][k] * v[k][j]; s = (s / a[i][i + 1]) / rv1[i]; for (let k = i + 1; k < n; k++) v[k][j] += s * a[i][k]; } } }
            for (let j = i + 1; j < n; j++) { v[i][j] = 0.0; v[j][i] = 0.0; } v[i][i] = 1.0;
        }
        // Accumulation of U (explicitly)
        const u = Array.from({ length: m }, () => new Array(n).fill(0.0));
        for (let i = n - 1; i >= 0; i--) {
            let l = i + 1; g = w[i]; for (let j = l; j < n; j++) a[i][j] = 0.0;
            if (g !== 0.0) { g = 1.0 / g; for (let j = l; j < n; j++) { let s = 0.0; for (let k = l; k < m; k++) s += a[k][i] * a[k][j]; let f = (s / a[i][i]) * g; for (let k = i; k < m; k++) a[k][j] += f * a[k][i]; } for (let j = i; j < m; j++) a[j][i] *= g; }
            else { for (let j = i; j < m; j++) a[j][i] = 0.0; } a[i][i] += 1.0;
        }
        for (let i = 0; i < m; i++) { for (let j = 0; j < n; j++) u[i][j] = a[i][j]; }
        // Diagonalization (QR)
        const maxIterations = 30 * n;
        for (let k = n - 1; k >= 0; k--) {
            for (let its = 0; its < maxIterations; its++) {
                let flag = true; let l = k;
                for (; l >= 0; l--) { if (l === 0 || Math.abs(rv1[l-1]) + anorm === anorm) { flag = false; break; } if (Math.abs(w[l - 1]) + anorm === anorm) break; }
                let c = 0.0; let s = 1.0;
                if (flag) { if (l > 0) { for (let i = l; i <= k; i++) { let f = s * rv1[i]; rv1[i] = c * rv1[i]; if (Math.abs(f) + anorm === anorm) break; g = w[i]; let h = Math.hypot(f, g); w[i] = h; h = 1.0 / h; c = g * h; s = -f * h; for (let j = 0; j < m; j++) { let y = u[j][l - 1]; let z = u[j][i]; u[j][l - 1] = y * c + z * s; u[j][i] = z * c - y * s; } } } }
                let z = w[k];
                if (l === k) { if (z < 0.0) { w[k] = -z; for (let j = 0; j < n; j++) v[j][k] = -v[j][k]; } break; }
                if (its === maxIterations - 1) { throw new Error("svdDecompose error: No convergence after maximum iterations."); }
                let x = w[l]; let y = w[k - 1]; g = rv1[k - 1]; let h = rv1[k]; let f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
                g = Math.hypot(f, 1.0); let f_sign = (f >= 0.0) ? 1.0 : -1.0; f = ((x - z) * (x + z) + h * ((y / (f + f_sign * g)) - h)) / x;
                c = 1.0; s = 1.0;
                for (let j = l; j < k; j++) {
                    let i = j + 1; g = rv1[i]; y = w[i]; h = s * g; g = c * g; z = Math.hypot(f, h); rv1[j] = z; c = f / z; s = h / z;
                    f = x * c + g * s; g = g * c - x * s; h = y * s; y = y * c;
                    for (let jj = 0; jj < n; jj++) { x = v[jj][j]; z = v[jj][i]; v[jj][j] = x * c + z * s; v[jj][i] = z * c - x * s; }
                    z = Math.hypot(f, h); w[j] = z; if (z !== 0.0) { z = 1.0 / z; c = f * z; s = h * z; }
                    f = c * g + s * y; x = c * y - s * g;
                    for (let jj = 0; jj < m; jj++) { y = u[jj][j]; z = u[jj][i]; u[jj][j] = y * c + z * s; u[jj][i] = z * c - y * s; }
                } rv1[l] = 0.0; rv1[k] = f; w[k] = x;
            }
        }
        return { u: u, s: w, v: v };
    }

    /**
     * Multiplies two matrices A and B.
     * @param {number[][]} a - Matrix A.
     * @param {number[][]} b - Matrix B.
     * @returns {number[][]} The resulting matrix A * B.
     * @throws {Error} If dimensions mismatch or elements are invalid.
     * @private
     */
    function multiplyMatrices(a, b) {
        const aRows = a.length; if (aRows === 0) return []; const aCols = a[0].length;
        const bRows = b.length; if (bRows === 0) return a.map(row => []); const bCols = b[0].length;
        if (aCols !== bRows) throw new Error(`Matrix dimension mismatch: A(${aRows}x${aCols}), B(${bRows}x${bCols})`);
        const result = new Array(aRows);
        for (let i = 0; i < aRows; i++) {
            result[i] = new Array(bCols).fill(0);
            for (let j = 0; j < bCols; j++) {
                for (let k = 0; k < aCols; k++) {
                    if (a[i]?.[k] === undefined || b[k]?.[j] === undefined) throw new Error(`Undefined element accessed: A[${i}][${k}] or B[${k}][${j}]`);
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        } return result;
    }

    /**
     * Transposes a matrix.
     * @param {number[][]} matrix - The matrix to transpose.
     * @returns {number[][]} The transposed matrix.
     * @throws {Error} If elements are invalid.
     * @private
     */
    function transposeMatrix(matrix) {
        const rows = matrix.length; if (rows === 0) return []; const cols = matrix[0].length; if (cols === 0) return matrix.map(() => []);
        const result = new Array(cols);
        for (let j = 0; j < cols; j++) {
            result[j] = new Array(rows);
            for (let i = 0; i < rows; i++) {
                if (matrix[i]?.[j] === undefined) throw new Error(`Undefined element accessed during transpose at [${i}][${j}]`);
                result[j][i] = matrix[i][j];
            }
        } return result;
    }

    /**
     * Calculates the pseudo-inverse of a square matrix using SVD.
     * @param {number[][]} matrix - The square matrix to invert.
     * @returns {number[][]} The pseudo-inverse matrix.
     * @throws {Error} If matrix is invalid or SVD fails.
     * @private
     */
    function invertMatrixUsingSVD(matrix) {
        const n = matrix.length; if (n === 0 || matrix.some(row => !row || row.length !== n)) throw new Error("Matrix must be square for SVD inversion.");
        const { u, s, v } = svdDecompose(matrix);
        if (!u || u.length === 0 || !s || s.length === 0 || !v || v.length === 0) throw new Error("SVD decomposition failed or returned empty matrices.");
        const uCols = u[0]?.length, sLen = s.length, vRows = v.length, vCols = v[0]?.length;
        // SVD of NxN matrix should yield U(MxN), S(N), V(NxN) where M>=N. Here M=N.
        if (uCols !== n || sLen !== n || vRows !== n || vCols !== n) console.warn(`SVD returned unexpected dimensions: U(${u.length}x${uCols}), S(${sLen}), V(${vRows}x${vCols}) for input NxN (${n}x${n})`);
        const tolerance = 1e-10;
        const sInv = new Array(n).fill(0).map(() => new Array(n).fill(0));
        const numSingularValues = Math.min(n, s.length);
        for (let i = 0; i < numSingularValues; i++) { if (Math.abs(s[i]) > tolerance) sInv[i][i] = 1 / s[i]; else sInv[i][i] = 0; }
        const vMatrix = v; const uTranspose = transposeMatrix(u);
        if (uTranspose.length !== n) throw new Error(`Dimension mismatch for U transpose: Expected ${n} rows, got ${uTranspose.length}`);
        const vsInv = multiplyMatrices(vMatrix, sInv); const inverse = multiplyMatrices(vsInv, uTranspose);
        if (inverse.length !== n || inverse[0]?.length !== n) console.warn(`SVD inverse resulted in non-square matrix (${inverse.length}x${inverse[0]?.length}) for square input (${n}x${n}).`);
        return inverse;
    }

    // Expose necessary functions to the global scope under a namespace
    global.LM_SVD_HELPERS = {
        solveLinearSystem,
        svdDecompose,
        multiplyMatrices,
        transposeMatrix,
        invertMatrixUsingSVD
    };

// Establish the root object, `window` in the browser, or `global` on the server.
})(typeof window !== 'undefined' ? window : global);
// globalfit.js

/**
 * @fileoverview Provides Levenberg-Marquardt fitting functions:
 *   - lmFitGlobal: For simultaneous fitting of multiple datasets with linking/fixing.
 *   - lmFit: A wrapper for fitting a single dataset using lmFitGlobal.
 *   - lmFitIndependent: Sequentially fits multiple datasets independently using lmFitGlobal.
 * Includes helpers for parameter mapping, chi-squared calculation, Jacobian calculation,
 * constraint application, and statistics calculation.
 * Depends on svd.js for linear algebra operations.
 * Version: 1.2.0
 */

(function(global) {
    'use strict';

    // Check if SVD helpers are loaded from svd.js
    if (!global.LM_SVD_HELPERS) {
        throw new Error("Error: svd.js must be loaded before fit.js");
    }
    // Import helpers from the global object created by svd.js
    const {
        solveLinearSystem,
        invertMatrixUsingSVD
    } = global.LM_SVD_HELPERS;


    // ============================================================================
    // Fitting Algorithm Helpers (Internal)
    // ============================================================================

    /** Log levels for controlling verbosity */
    const LOG_LEVELS = { none: 0, error: 1, warn: 2, info: 3, debug: 4 };

    /**
     * Calculates the gradient vector (J^T * r).
     * @private
     */
    function calculateGradient(jacobian, residuals) {
        const numPoints = jacobian.length; if (numPoints === 0) return []; const numParams = jacobian[0]?.length ?? 0;
        if (numParams === 0 && residuals.length === 0) return []; if (numParams === 0 && residuals.length > 0) throw new Error("Jacobian has zero columns but residuals exist.");
        if (numPoints !== residuals.length) throw new Error(`Jacobian rows (${numPoints}) must match residuals length (${residuals.length}).`);
        const gradient = new Array(numParams).fill(0);
        for (let j = 0; j < numParams; j++) {
            for (let i = 0; i < numPoints; i++) {
                if (jacobian[i]?.[j] === undefined || !isFinite(jacobian[i][j]) || residuals[i] === undefined || !isFinite(residuals[i])) throw new Error(`Invalid value in gradient calculation at point ${i}, param ${j}.`);
                gradient[j] += jacobian[i][j] * residuals[i];
            }
        } return gradient;
    }

    /**
     * Calculates the approximate Hessian matrix (J^T * J).
     * @private
     */
    function calculateHessian(jacobian) {
        const numPoints = jacobian.length; if (numPoints === 0) return []; const numParams = jacobian[0]?.length ?? 0; if (numParams === 0) return [];
        const hessian = new Array(numParams).fill(0).map(() => new Array(numParams).fill(0));
        for (let j = 0; j < numParams; j++) {
            for (let k = j; k < numParams; k++) {
                let sum = 0;
                for (let i = 0; i < numPoints; i++) {
                    if (jacobian[i]?.[j] === undefined || !isFinite(jacobian[i][j]) || jacobian[i]?.[k] === undefined || !isFinite(jacobian[i][k])) throw new Error(`Invalid value in Hessian calculation at point ${i}, params ${j}, ${k}.`);
                    sum += jacobian[i][j] * jacobian[i][k];
                }
                hessian[j][k] = sum; if (j !== k) hessian[k][j] = sum;
            }
        } return hessian;
    }

    /**
     * Creates mapping between full parameter structure and the flat array of active parameters.
     * Internal helper function for lmFitGlobal.
     * @private
     */
    function setupParameterMapping(initialParameters, linkMap, fixMapInput, logFn) {
        logFn("--- Running setupParameterMapping ---", 'debug');
        logFn(`Received linkMap: ${JSON.stringify(linkMap)}`, 'debug');
        logFn(`Received fixMapInput: ${JSON.stringify(fixMapInput)}`, 'debug');

        const paramStructure = []; const flatInitialParams = []; const paramCoordinates = []; let currentFlatIndex = 0;
        const fixMap = fixMapInput ? JSON.parse(JSON.stringify(fixMapInput)) : [];
        initialParameters.forEach((dsParams, dsIdx) => {
            paramStructure.push([]); if (!fixMap[dsIdx]) fixMap[dsIdx] = [];
            dsParams.forEach((pArray, pIdx) => {
                paramStructure[dsIdx].push(pArray.length); if (!fixMap[dsIdx][pIdx]) fixMap[dsIdx][pIdx] = new Array(pArray.length).fill(false);
                pArray.forEach((pValue, vIdx) => {
                    flatInitialParams.push(pValue); paramCoordinates.push([[dsIdx, pIdx], vIdx]);
                    if (fixMap[dsIdx][pIdx].length <= vIdx) fixMap[dsIdx][pIdx][vIdx] = false;
                    currentFlatIndex++;
                });
            });
        });
        const nTotalParams = flatInitialParams.length; const masterMap = new Array(nTotalParams).fill(-1); const isFixed = new Array(nTotalParams).fill(false);
        const activeParamInfo = []; const activeInitialParams = []; const activeParamLabels = [];

        // 2. Apply fixMap
        paramCoordinates.forEach((coord, flatIdx) => { const [[dsIdx, paramIdx], valIdx] = coord; if (fixMap[dsIdx]?.[paramIdx]?.[valIdx] === true) isFixed[flatIdx] = true; });
        logFn(`Initial isFixed array: ${JSON.stringify(isFixed)}`, 'debug');

        // 3. Process linkMap
        const linkGroups = linkMap || [];
        linkGroups.forEach((group, groupIdx) => {
            logFn(`Processing Link Group ${groupIdx}: ${JSON.stringify(group)}`, 'debug');
            let masterFlatIndex = -1; let masterCoord = null;
            for (const coord of group) {
                const [[dsIdx, paramIdx], valIdx] = coord;
                const flatIdx = paramCoordinates.findIndex(c => c[0][0] === dsIdx && c[0][1] === paramIdx && c[1] === valIdx);
                if (flatIdx !== -1) { if (!isFixed[flatIdx]) { masterFlatIndex = flatIdx; masterCoord = coord; break; } }
                else logFn(`Coordinate ${JSON.stringify(coord)} in linkMap not found.`, 'warn');
            }
            if (masterFlatIndex === -1 && group.length > 0) {
                const firstCoord = group[0]; const firstFlatIdx = paramCoordinates.findIndex(c => c[0][0] === firstCoord[0][0] && c[0][1] === firstCoord[0][1] && c[1] === firstCoord[1]);
                if (firstFlatIdx !== -1) {
                    masterFlatIndex = firstFlatIdx;
                    logFn(`Link Group ${groupIdx} seems fixed, master set to flat index ${masterFlatIndex}`, 'debug');
                    group.forEach(coord => {
                        const flatIdx = paramCoordinates.findIndex(c => c[0][0] === coord[0][0] && c[0][1] === coord[0][1] && c[1] === coord[1]);
                        if (flatIdx !== -1 && flatIdx !== masterFlatIndex) {
                            if (!isFixed[flatIdx]) { logFn(`Marking ${JSON.stringify(coord)} as fixed due to fixed link group.`, 'debug'); isFixed[flatIdx] = true; }
                            masterMap[flatIdx] = masterFlatIndex;
                        }
                    });
                } else logFn(`Could not find fixed parameter for link group ${JSON.stringify(firstCoord)}`, 'warn');
            } else if (masterFlatIndex !== -1) {
                 logFn(`Link Group ${groupIdx} master: ${JSON.stringify(masterCoord)} (flat index ${masterFlatIndex})`, 'debug');
                group.forEach(coord => {
                    const flatIdx = paramCoordinates.findIndex(c => c[0][0] === coord[0][0] && c[0][1] === coord[0][1] && c[1] === coord[1]);
                    if (flatIdx !== -1 && flatIdx !== masterFlatIndex) {
                        if (isFixed[flatIdx]) { logFn(`Linking overrides fixed status for ${JSON.stringify(coord)}.`, 'warn'); isFixed[flatIdx] = false; }
                        masterMap[flatIdx] = masterFlatIndex;
                        flatInitialParams[flatIdx] = flatInitialParams[masterFlatIndex];
                        logFn(`  Linked slave ${JSON.stringify(coord)} (flat ${flatIdx}) to master ${masterFlatIndex}`, 'debug');
                    }
                });
            }
        });
        logFn(`Final isFixed array: ${JSON.stringify(isFixed)}`, 'debug');
        logFn(`Final masterMap array: ${JSON.stringify(masterMap)}`, 'debug');

        // 4. Identify active parameters
        let activeIndex = 0;
        for (let i = 0; i < nTotalParams; i++) {
            if (!isFixed[i] && masterMap[i] === -1) {
                activeParamInfo.push({ originalCoord: paramCoordinates[i], flatIndex: i, activeIndex: activeIndex });
                activeInitialParams.push(flatInitialParams[i]); const [[ds, p], v] = paramCoordinates[i]; activeParamLabels.push(`ds${ds}_p${p}_v${v}`); activeIndex++;
            }
        }
        const totalActiveParams = activeInitialParams.length;
        logFn(`Identified ${totalActiveParams} active parameters: ${JSON.stringify(activeParamLabels)}`, 'debug');
        logFn(`Active Initial Params: ${JSON.stringify(activeInitialParams)}`, 'debug');

        // 5. Create the reconstruction function
        const reconstructParams = (activeParamsCurrent) => {
            if (activeParamsCurrent.length !== totalActiveParams) throw new Error(`reconstructParams expects ${totalActiveParams} params, received ${activeParamsCurrent.length}`);
            const reconstructedFlat = [...flatInitialParams];
            activeParamInfo.forEach((info, actIdx) => { if (actIdx >= activeParamsCurrent.length) throw new Error(`Mismatch activeParamInfo/activeParams.`); reconstructedFlat[info.flatIndex] = activeParamsCurrent[actIdx]; });
            for (let i = 0; i < nTotalParams; i++) { if (masterMap[i] !== -1) { if (masterMap[i] >= reconstructedFlat.length) throw new Error(`Master index ${masterMap[i]} out of bounds.`); reconstructedFlat[i] = reconstructedFlat[masterMap[i]]; } }
            const nestedParams = []; let currentFlatIdx_recon = 0;
            paramStructure.forEach((dsStruct, dsIdx) => {
                nestedParams[dsIdx] = [];
                dsStruct.forEach((pLen, pIdx) => {
                    nestedParams[dsIdx][pIdx] = [];
                    for (let vIdx = 0; vIdx < pLen; vIdx++) { if (currentFlatIdx_recon >= reconstructedFlat.length) throw new Error(`Flat index ${currentFlatIdx_recon} out of bounds.`); nestedParams[dsIdx][pIdx][vIdx] = reconstructedFlat[currentFlatIdx_recon]; currentFlatIdx_recon++; }
                });
            });
            if (currentFlatIdx_recon !== nTotalParams) logFn(`Param reconstruction mismatch: ${currentFlatIdx_recon} vs ${nTotalParams}`, 'warn');
            return nestedParams;
        };

        // Return details needed for error reconstruction as well
        return {
            activeInitialParams, reconstructParams, activeParamInfo, totalActiveParams,
            paramStructure, activeParamLabels,
            // Internal details needed later:
            isFixed, masterMap, paramCoordinates, nTotalParams
        };
    } // <-- End of setupParameterMapping function

    /**
     * Calculates the global chi-squared value.
     * Internal helper function for lmFitGlobal.
     * @private
     */
    function calculateChiSquaredGlobal(data, modelFunction, reconstructParamsFunc, activeParams, robustCostFunction, paramStructure, logFn) {
        const reconstructedParams = reconstructParamsFunc(activeParams); let chiSquared = 0; let totalPoints = 0;
        data.x.forEach((xDataset, dsIdx) => {
            const yDataset = data.y[dsIdx]; const yeDataset = data.ye[dsIdx]; const models = modelFunction[dsIdx]; const paramsForDs = reconstructedParams[dsIdx];
            if (!xDataset || !yDataset || !yeDataset || !models || !paramsForDs || xDataset.length !== yDataset.length || xDataset.length !== yeDataset.length) { logFn(`Data length/param mismatch dataset ${dsIdx}`, 'error'); return NaN; }
            for (let ptIdx = 0; ptIdx < xDataset.length; ptIdx++) {
                totalPoints++; const xPoint = xDataset[ptIdx]; const yPoint = yDataset[ptIdx]; const yePoint = yeDataset[ptIdx];
                if (yePoint === 0) { logFn(`Zero error point ${ptIdx} ds ${dsIdx}. Skipping.`, 'warn'); continue; }
                if (!isFinite(yPoint) || !isFinite(yePoint)) { logFn(`Non-finite y/ye point ${ptIdx} ds ${dsIdx}. Skipping.`, 'warn'); continue; }
                let combinedModelValue = 0;
                try {
                    models.forEach((modelFunc, paramIdx) => {
                        if (!paramsForDs[paramIdx]) throw new Error(`Missing params ds ${dsIdx}, pIdx ${paramIdx}.`);
                        const componentParams = paramsForDs[paramIdx]; const componentModelResult = modelFunc(componentParams, [xPoint]);
                        if (!componentModelResult || componentModelResult.length !== 1 || !isFinite(componentModelResult[0])) throw new Error(`Model ${dsIdx}-${paramIdx} invalid result x=${xPoint}: ${JSON.stringify(componentModelResult)}`);
                        combinedModelValue += componentModelResult[0];
                    });
                } catch (error) { logFn(`Error evaluating model ds ${dsIdx}, pt ${ptIdx}: ${error.message}`, 'error'); return NaN; }
                if (!isFinite(combinedModelValue)) { logFn(`Non-finite model value pt ${ptIdx} ds ${dsIdx}. Skipping.`, 'warn'); continue; }
                const residual = (yPoint - combinedModelValue) / yePoint;
                if (robustCostFunction === 1) chiSquared += Math.abs(residual);
                else if (robustCostFunction === 2) chiSquared += Math.log(1 + 0.5 * (residual * residual));
                else chiSquared += residual * residual;
            }
        });
        if (!isFinite(chiSquared)) { logFn("Non-finite chi-squared calculated.", 'error'); return Infinity; }
        return chiSquared;
    }

    /**
     * Calculates the global Jacobian matrix and residuals vector using numerical differentiation.
     * Internal helper function for lmFitGlobal.
     * @private
     */
    function calculateJacobianAndResidualsGlobal(data, modelFunction, reconstructParamsFunc, activeParams, activeParamInfo, epsilon, paramStructure, logFn) {
        const reconstructedParams = reconstructParamsFunc(activeParams); const residuals = []; const jacobianRows = []; const totalActiveParams = activeParamInfo.length; let totalPoints = 0;
        data.x.forEach((xDataset, dsIdx) => {
            const yDataset = data.y[dsIdx]; const yeDataset = data.ye[dsIdx]; const models = modelFunction[dsIdx]; const paramsForDs = reconstructedParams[dsIdx];
            if (!xDataset || !yDataset || !yeDataset || !models || !paramsForDs || xDataset.length !== yDataset.length || xDataset.length !== yeDataset.length) throw new Error(`Data length mismatch ds ${dsIdx} Jacobian.`);
            for (let ptIdx = 0; ptIdx < xDataset.length; ptIdx++) {
                totalPoints++; const xPoint = xDataset[ptIdx]; const yPoint = yDataset[ptIdx]; const yePoint = yeDataset[ptIdx];
                if (yePoint === 0) throw new Error(`Zero error pt ${ptIdx} ds ${dsIdx} Jacobian.`);
                if (!isFinite(yPoint) || !isFinite(yePoint)) throw new Error(`Non-finite y/ye pt ${ptIdx} ds ${dsIdx} Jacobian.`);
                let originalCombinedModelValue = 0;
                try {
                    models.forEach((modelFunc, paramIdx) => {
                        if (!paramsForDs[paramIdx]) throw new Error(`Missing params ds ${dsIdx}, pIdx ${paramIdx} (Jacobian orig).`);
                        const componentParams = paramsForDs[paramIdx]; const componentModelResult = modelFunc(componentParams, [xPoint]);
                        if (!componentModelResult || componentModelResult.length !== 1 || !isFinite(componentModelResult[0])) throw new Error(`Model ${dsIdx}-${paramIdx} invalid result x=${xPoint}: ${JSON.stringify(componentModelResult)}`);
                        originalCombinedModelValue += componentModelResult[0];
                    });
                } catch (error) { logFn(`Error evaluating model (orig) ds ${dsIdx}, pt ${ptIdx}: ${error.message}`, 'error'); throw error; }
                if (!isFinite(originalCombinedModelValue)) throw new Error(`Non-finite orig model value pt ${ptIdx} ds ${dsIdx}.`);
                const residual = (yPoint - originalCombinedModelValue) / yePoint; residuals.push(residual);
                const jacobianRow = new Array(totalActiveParams).fill(0);
                for (let actIdx = 0; actIdx < totalActiveParams; actIdx++) {
                    const perturbedActiveParams = [...activeParams]; const originalValue = perturbedActiveParams[actIdx];
                    let h = epsilon * Math.abs(originalValue) + epsilon; if (h === 0) h = epsilon;
                    perturbedActiveParams[actIdx] += h; const perturbedReconstructed = reconstructParamsFunc(perturbedActiveParams);
                    let perturbedCombinedModelValue = 0; let perturbationFailed = false;
                    try {
                        const perturbedParamsForDs = perturbedReconstructed[dsIdx];
                        models.forEach((modelFunc, paramIdx) => {
                            if (perturbationFailed) return; if (!perturbedParamsForDs || !perturbedParamsForDs[paramIdx]) throw new Error(`Missing params ds ${dsIdx}, pIdx ${paramIdx} (Jacobian pert).`);
                            const componentParams = perturbedParamsForDs[paramIdx]; const componentModelResult = modelFunc(componentParams, [xPoint]);
                            if (!componentModelResult || componentModelResult.length !== 1 || !isFinite(componentModelResult[0])) { logFn(`Model ${dsIdx}-${paramIdx} invalid result during perturbation x=${xPoint}`, 'warn'); perturbationFailed = true; return; }
                            perturbedCombinedModelValue += componentModelResult[0];
                        });
                    } catch (error) { logFn(`Error evaluating model (pert) ds ${dsIdx}, pt ${ptIdx}, actP ${actIdx}: ${error.message}`, 'error'); perturbationFailed = true; }
                    if (perturbationFailed || !isFinite(perturbedCombinedModelValue)) { logFn(`Non-finite/failed pert model value pt ${ptIdx}, ds ${dsIdx}, actP ${actIdx}. Deriv=0.`, 'warn'); jacobianRow[actIdx] = 0; }
                    else { const derivative = (perturbedCombinedModelValue - originalCombinedModelValue) / h; jacobianRow[actIdx] = -derivative / yePoint; }
                    if (!isFinite(jacobianRow[actIdx])) { logFn(`Non-finite Jacobian element pt ${ptIdx}, ds ${dsIdx}, actP ${actIdx}. Setting=0.`, 'warn'); jacobianRow[actIdx] = 0; }
                } jacobianRows.push(jacobianRow);
            }
        });
        if (residuals.length !== jacobianRows.length) throw new Error(`Internal error: Residuals (${residuals.length}) != Jacobian rows (${jacobianRows.length})`);
        if (jacobianRows.length > 0 && jacobianRows[0].length !== totalActiveParams) throw new Error(`Internal error: Jacobian cols (${jacobianRows[0].length}) != active params (${totalActiveParams})`);
        return { jacobian: jacobianRows, residuals };
    }

    /**
     * Applies constraints (box and custom function) to the reconstructed parameter structure.
     * Internal helper function for lmFitGlobal.
     * @private
     */
    function applyConstraintsGlobal(reconstructedParams, constraints, activeParamInfo, paramStructure, constraintFunction, logFn) {
        let constrainedParams = reconstructedParams; // Start with input
        let changedActive = false; // Tracks changes from box constraints
        let boxConstraintApplied = false;

        // 1. Apply Box Constraints first
        if (constraints) {
            // Avoid deep copy unless necessary
            let paramsCopy = null;
            paramStructure.forEach((dsStruct, dsIdx) => {
                if (!constraints[dsIdx]) return;
                dsStruct.forEach((pLen, pIdx) => {
                    if (!constraints[dsIdx][pIdx]) return;
                    for (let vIdx = 0; vIdx < pLen; vIdx++) {
                        const constraint = constraints[dsIdx]?.[pIdx]?.[vIdx];
                        if (constraint) {
                            // Create copy only when a constraint is actually applied
                            if (!paramsCopy) paramsCopy = JSON.parse(JSON.stringify(reconstructedParams));
                            let value = paramsCopy[dsIdx][pIdx][vIdx];
                            const originalValue = value;

                            if (constraint.min !== undefined && value < constraint.min) value = constraint.min;
                            if (constraint.max !== undefined && value > constraint.max) value = constraint.max;

                            if (value !== originalValue) {
                                paramsCopy[dsIdx][pIdx][vIdx] = value;
                                boxConstraintApplied = true;
                                const isActive = activeParamInfo.some(info => info.originalCoord[0][0] === dsIdx && info.originalCoord[0][1] === pIdx && info.originalCoord[1] === vIdx);
                                if (isActive) changedActive = true;
                            }
                        }
                    }
                });
            });
            if (paramsCopy) constrainedParams = paramsCopy; // Use the copy if changes were made
        }

        // 2. Apply Custom Constraint Function (if provided)
        if (typeof constraintFunction === 'function') {
            try {
                // If box constraints were applied, pass the already modified params
                // Otherwise, pass the original (or a copy if paranoia is high)
                const paramsForCustomFn = boxConstraintApplied ? constrainedParams : reconstructedParams;
                const paramsBeforeCustom = boxConstraintApplied ? null : JSON.stringify(paramsForCustomFn); // Only stringify if needed

                constrainedParams = constraintFunction(paramsForCustomFn); // Apply user function

                // Check if the custom function actually returned a valid structure (basic check)
                if (!Array.isArray(constrainedParams) || constrainedParams.length !== paramStructure.length) {
                    throw new Error("Constraint function did not return a valid parameter structure.");
                }

                // Check if the custom function modified the parameters
                const paramsAfterCustom = JSON.stringify(constrainedParams);
                if (!boxConstraintApplied && paramsAfterCustom !== paramsBeforeCustom) {
                    changedActive = true; // Assume active might have changed if any change occurred
                    logFn("Custom constraint function modified parameters.", 'debug');
                } else if (boxConstraintApplied && paramsAfterCustom !== JSON.stringify(paramsForCustomFn)) {
                     changedActive = true; // Assume active might have changed if any change occurred
                     logFn("Custom constraint function modified parameters after box constraints.", 'debug');
                }

            } catch (e) {
                logFn(`Error executing custom constraint function: ${e.message}`, 'error');
                // Revert to state *before* custom constraint function was called
                constrainedParams = boxConstraintApplied ? JSON.parse(JSON.stringify(reconstructedParams)) : reconstructedParams; // Revert if box was applied
                // Or maybe just revert to the input reconstructedParams always? Let's revert to input.
                // constrainedParams = reconstructedParams; // Revert to original input on error
                // changedActive might need reset depending on desired behavior on error
            }
        }

        return { constrainedParams, changedActive };
    }


    /**
     * Calculates the final weighted residuals for each dataset after fitting.
     * @private
     */
    function calculateFinalResiduals(data, modelFunction, finalReconstructedParams, onLog) {
        if (!data || !data.x || !data.y || !data.ye || !modelFunction || !finalReconstructedParams) { onLog("Missing data/models/params for final residual calculation.", 'error'); return null; }
        const residualsPerSeries = []; let errorOccurred = false;
        data.x.forEach((xDataset, dsIdx) => {
            const yDataset = data.y[dsIdx]; const yeDataset = data.ye[dsIdx]; const models = modelFunction[dsIdx]; const paramsForDs = finalReconstructedParams[dsIdx]; const currentResiduals = [];
            if (!yDataset || !yeDataset || !models || !paramsForDs || xDataset.length !== yDataset.length || xDataset.length !== yeDataset.length) { onLog(`Inconsistent data/model/params for dataset ${dsIdx} in final residual calculation. Skipping.`, 'warn'); residualsPerSeries.push([]); return; }
            for (let ptIdx = 0; ptIdx < xDataset.length; ptIdx++) {
                const xPoint = xDataset[ptIdx]; const yPoint = yDataset[ptIdx]; const yePoint = yeDataset[ptIdx];
                if (yePoint === 0 || !isFinite(yPoint) || !isFinite(yePoint)) { onLog(`Invalid data/error at point ${ptIdx} in dataset ${dsIdx} for final residuals. Storing NaN.`, 'warn'); currentResiduals.push(NaN); continue; }
                let combinedModelValue = 0; let modelEvalError = false;
                try { models.forEach((modelFunc, paramIdx) => { if (!paramsForDs[paramIdx]) throw new Error(`Missing params for model ${paramIdx}.`); const componentParams = paramsForDs[paramIdx]; const componentModelResult = modelFunc(componentParams, [xPoint]); if (!componentModelResult || componentModelResult.length !== 1 || !isFinite(componentModelResult[0])) throw new Error(`Model ${dsIdx}-${paramIdx} returned invalid result.`); combinedModelValue += componentModelResult[0]; }); }
                catch (error) { onLog(`Error evaluating model for final residual at ds ${dsIdx}, pt ${ptIdx}: ${error.message}`, 'error'); modelEvalError = true; errorOccurred = true; }
                if (modelEvalError || !isFinite(combinedModelValue)) { currentResiduals.push(NaN); } else { currentResiduals.push((yPoint - combinedModelValue) / yePoint); }
            } residualsPerSeries.push(currentResiduals);
        });
        return errorOccurred ? null : residualsPerSeries;
    }


    /**
     * Calculates the fitted model curve over the range of the input data.
     * @private
     */
    function calculateFittedModelCurves(data, modelFunction, finalReconstructedParams, numPoints, onLog) {
         if (!data || !data.x || !modelFunction || !finalReconstructedParams || numPoints <= 1) { onLog("Missing data/models/params or invalid numPoints for fitted curve calculation.", 'error'); return null; }
        const fittedCurves = []; let errorOccurred = false;
         data.x.forEach((xDataset, dsIdx) => {
            const models = modelFunction[dsIdx]; const paramsForDs = finalReconstructedParams[dsIdx]; const currentCurve = { x: [], y: [] };
            if (!models || !paramsForDs || xDataset.length === 0) { onLog(`Inconsistent data/model/params for dataset ${dsIdx} in fitted curve calculation. Skipping.`, 'warn'); fittedCurves.push(currentCurve); return; }
            let xMin = xDataset[0]; let xMax = xDataset[0]; for(let i=1; i<xDataset.length; ++i) { if (xDataset[i] < xMin) xMin = xDataset[i]; if (xDataset[i] > xMax) xMax = xDataset[i]; }
            if (!isFinite(xMin) || !isFinite(xMax) || xMax < xMin) { onLog(`Invalid X range [${xMin}, ${xMax}] for dataset ${dsIdx} in fitted curve calculation. Skipping.`, 'warn'); fittedCurves.push(currentCurve); return; }
            const dx = (xMax === xMin) ? 0 : (xMax - xMin) / (numPoints - 1);
            for (let i = 0; i < numPoints; i++) {
                const xCalc = (numPoints === 1 || dx === 0) ? xMin : xMin + i * dx; let yCalc = 0; let modelEvalError = false;
                try { models.forEach((modelFunc, paramIdx) => { if (!paramsForDs[paramIdx]) throw new Error(`Missing params for model ${paramIdx}.`); const componentParams = paramsForDs[paramIdx]; const componentModelResult = modelFunc(componentParams, [xCalc]); if (!componentModelResult || componentModelResult.length !== 1 || !isFinite(componentModelResult[0])) throw new Error(`Model ${dsIdx}-${paramIdx} returned invalid result.`); yCalc += componentModelResult[0]; }); }
                catch(error) { onLog(`Error evaluating model for fitted curve at ds ${dsIdx}, x=${xCalc}: ${error.message}`, 'error'); modelEvalError = true; errorOccurred = true; }
                currentCurve.x.push(xCalc); currentCurve.y.push(modelEvalError || !isFinite(yCalc) ? NaN : yCalc);
            } fittedCurves.push(currentCurve);
         });
         return errorOccurred ? null : fittedCurves;
    }

    /**
     * Helper function to get details needed for error reconstruction.
     * This duplicates some logic from setupParameterMapping but avoids returning
     * internal details from the main setup function.
     * @private
     */
    function setupParameterMappingDetails(initialParameters, linkMap, fixMapInput) {
        const paramStructure = []; const paramCoordinates = []; let currentFlatIndex = 0;
        const fixMap = fixMapInput ? JSON.parse(JSON.stringify(fixMapInput)) : [];
        initialParameters.forEach((dsParams, dsIdx) => {
            paramStructure.push([]); if (!fixMap[dsIdx]) fixMap[dsIdx] = [];
            dsParams.forEach((pArray, pIdx) => {
                paramStructure[dsIdx].push(pArray.length); if (!fixMap[dsIdx][pIdx]) fixMap[dsIdx][pIdx] = new Array(pArray.length).fill(false);
                pArray.forEach((pValue, vIdx) => {
                    paramCoordinates.push([[dsIdx, pIdx], vIdx]); // Store coordinate
                    if (fixMap[dsIdx][pIdx].length <= vIdx) fixMap[dsIdx][pIdx][vIdx] = false;
                    currentFlatIndex++;
                });
            });
        });
        const nTotalParams = currentFlatIndex; // Use calculated length
        const masterMap = new Array(nTotalParams).fill(-1);
        const isFixed = new Array(nTotalParams).fill(false);
        paramCoordinates.forEach((coord, flatIdx) => { const [[dsIdx, paramIdx], valIdx] = coord; if (fixMap[dsIdx]?.[paramIdx]?.[valIdx] === true) isFixed[flatIdx] = true; });
        const linkGroups = linkMap || [];
        linkGroups.forEach(group => {
            let masterFlatIndex = -1;
            for (const coord of group) { const [[dsIdx, paramIdx], valIdx] = coord; const flatIdx = paramCoordinates.findIndex(c => c[0][0] === dsIdx && c[0][1] === paramIdx && c[1] === valIdx); if (flatIdx !== -1 && !isFixed[flatIdx]) { masterFlatIndex = flatIdx; break; } }
            if (masterFlatIndex === -1 && group.length > 0) { const firstCoord = group[0]; const firstFlatIdx = paramCoordinates.findIndex(c => c[0][0] === firstCoord[0][0] && c[0][1] === firstCoord[0][1] && c[1] === firstCoord[1]); if (firstFlatIdx !== -1) { masterFlatIndex = firstFlatIdx; group.forEach(coord => { const flatIdx = paramCoordinates.findIndex(c => c[0][0] === coord[0][0] && c[0][1] === coord[0][1] && c[1] === coord[1]); if (flatIdx !== -1 && flatIdx !== masterFlatIndex) { isFixed[flatIdx] = true; masterMap[flatIdx] = masterFlatIndex; } }); } }
            else if (masterFlatIndex !== -1) { group.forEach(coord => { const flatIdx = paramCoordinates.findIndex(c => c[0][0] === coord[0][0] && c[0][1] === coord[0][1] && c[1] === coord[1]); if (flatIdx !== -1 && flatIdx !== masterFlatIndex) { if (isFixed[flatIdx]) isFixed[flatIdx] = false; masterMap[flatIdx] = masterFlatIndex; } }); }
        });
        return { isFixed, masterMap, paramCoordinates, nTotalParams, paramStructure }; // Return details needed
    }


    // ============================================================================
    // Main Global Fit Function (Exported)
    // ============================================================================

    /**
     * Performs global curve fitting for multiple datasets using the Levenberg-Marquardt algorithm.
     * Allows linking and fixing parameters across datasets and uses composite models.
     * Includes calculation of goodness-of-fit statistics.
     *
     * @param {object} data - Contains the experimental data.
     *   @param {number[][]} data.x - Array of arrays of independent variable values.
     *   @param {number[][]} data.y - Array of arrays of dependent variable values.
     *   @param {number[][]} data.ye - Array of arrays of error values for y.
     * @param {Function[][]} modelFunction - Array of arrays of model functions.
     * @param {number[][][]} initialParameters - Nested array of initial parameter guesses.
     * @param {object} [options={}] - Optional configuration for the fitting process.
     *   @param {number} [options.maxIterations=100]
     *   @param {number} [options.errorTolerance=1e-6]
     *   @param {number} [options.gradientTolerance=1e-6]
     *   @param {Array<Array<[[number, number], number]>>} [options.linkMap=[]]
     *   @param {boolean[][][]} [options.fixMap=null]
     *   @param {object[][][]} [options.constraints=null] - Box constraints {min, max}.
     *   @param {Function | null} [options.constraintFunction=null] - Custom function: `(params) => modifiedParams`. Applied after box constraints.
     *   @param {string} [options.logLevel='info'] - Control logging verbosity ('none', 'error', 'warn', 'info', 'debug').
     *   @param {Function} [options.onLog=()=>{}] - Callback for logs: `(message, level) => {}`.
     *   @param {Function} [options.onProgress=()=>{}] - Callback for progress: `(progressData) => {}`, where `progressData = { iteration, chiSquared, lambda, activeParameters }`.
     *   @param {number|null} [options.robustCostFunction=null] - null, 1 (L1), or 2 (Lorentzian).
     *   @param {number} [options.lambdaInitial=1e-3]
     *   @param {number} [options.lambdaIncreaseFactor=10]
     *   @param {number} [options.lambdaDecreaseFactor=10]
     *   @param {number} [options.epsilon=1e-8]
     *   @param {boolean | {numPoints: number}} [options.calculateFittedModel=false] - If true or object, calculate fitted curve.
     * @returns {object} - Fitting results including statistics.
     *   (Includes p_active, p_reconstructed, finalParamErrors, chiSquared, covarianceMatrix, parameterErrors,
     *    iterations, converged, activeParamLabels, error, totalPoints, degreesOfFreedom,
     *    reducedChiSquared, aic, aicc, bic, residualsPerSeries, fittedModelCurves)
     */
    function lmFitGlobal(data, modelFunction, initialParameters, options = {}) {
        // --- Options Processing ---
        const maxIterations = options.maxIterations || 100;
        const errorTolerance = options.errorTolerance || 1e-6;
        const gradientTolerance = options.gradientTolerance || 1e-6;
        const linkMap = options.linkMap || [];
        const fixMapInput = options.fixMap || null; // Rename to avoid conflict
        const constraints = options.constraints || null;
        const constraintFunction = options.constraintFunction || null; // Get custom constraint fn
        const logLevelStr = options.logLevel || 'info';
        const logLevel = LOG_LEVELS[logLevelStr.toLowerCase()] ?? LOG_LEVELS.info;
        const onLog = options.onLog && typeof options.onLog === 'function' ? options.onLog : () => {};
        const onProgress = options.onProgress && typeof options.onProgress === 'function' ? options.onProgress : () => {}; // Progress callback
        const robustCostFunction = options.robustCostFunction ?? null;
        let lambda = options.lambdaInitial || 1e-3;
        const lambdaIncreaseFactor = options.lambdaIncreaseFactor || 10;
        const lambdaDecreaseFactor = options.lambdaDecreaseFactor || 10;
        const epsilon = options.epsilon || 1e-8;
        const calculateFittedOpt = options.calculateFittedModel ?? false;
        const numPointsForCurve = (typeof calculateFittedOpt === 'object' && calculateFittedOpt.numPoints > 1)
                                  ? calculateFittedOpt.numPoints
                                  : 300;
        const shouldCalculateFitted = calculateFittedOpt === true || (typeof calculateFittedOpt === 'object');

        // Internal logging helpers respecting logLevel
        const logFn = (message, level) => {
            const messageLevel = LOG_LEVELS[level] ?? LOG_LEVELS.info;
            if (logLevel >= messageLevel) {
                onLog(message, level);
            }
        };
        const logInfo = (message) => logFn(message, 'info');
        const logWarn = (message) => logFn(message, 'warn');
        const logError = (message) => logFn(message, 'error'); // Always log errors? Or respect level? Let's respect level >= error
        const logDebug = (message) => logFn(message, 'debug');

        logInfo("Starting lmFitGlobal...");

        // --- Calculate Total Data Points (N) ---
        let totalPoints = 0;
        if (data && data.x) { data.x.forEach(xDataset => { if (Array.isArray(xDataset)) totalPoints += xDataset.length; else logWarn("Non-array in data.x."); }); }
        logInfo(`Total data points (N): ${totalPoints}`);
        const baseErrorReturn = { p_active: [], p_reconstructed: initialParameters, finalParamErrors: null, chiSquared: NaN, covarianceMatrix: null, parameterErrors: null, iterations: 0, converged: false, activeParamLabels: [], totalPoints: totalPoints, degreesOfFreedom: NaN, reducedChiSquared: NaN, aic: NaN, aicc: NaN, bic: NaN, residualsPerSeries: null, fittedModelCurves: null };
        if (totalPoints === 0) { const errMsg = "No data points."; logError(errMsg); return { ...baseErrorReturn, error: errMsg }; }

        // --- Parameter Mapping Setup (K = totalActiveParams) ---
        let setupResults;
         try { setupResults = setupParameterMapping(initialParameters, linkMap, fixMapInput, logDebug); }
         catch (error) { const errMsg = `Parameter setup failed: ${error.message}`; logError(errMsg); return { ...baseErrorReturn, error: errMsg }; }

        const { activeInitialParams, reconstructParams, activeParamInfo, totalActiveParams, paramStructure, activeParamLabels } = setupResults;
        const K = totalActiveParams;
        // Get mapping details needed for error reconstruction
        const { isFixed, masterMap, paramCoordinates, nTotalParams } = setupParameterMappingDetails(initialParameters, linkMap, fixMapInput);

        // --- Handle Case: No Active Parameters ---
        if (K === 0) {
            logWarn("No active parameters to fit. Calculating initial stats.");
            let initialChiSq = NaN; let dof = totalPoints; let redChiSq = NaN, aic = NaN, aicc = NaN, bic = NaN;
            let finalResiduals = null; let fittedCurves = null; let finalErrors = null;
            const initialReconstructed = reconstructParams([]);
            try {
                initialChiSq = calculateChiSquaredGlobal(data, modelFunction, reconstructParams, [], robustCostFunction, paramStructure, logFn);
                finalResiduals = calculateFinalResiduals(data, modelFunction, initialReconstructed, logFn);
                if (shouldCalculateFitted) { fittedCurves = calculateFittedModelCurves(data, modelFunction, initialReconstructed, numPointsForCurve, logFn); }
                finalErrors = initialReconstructed.map(ds => ds.map(pg => pg.map(() => 0))); // All fixed -> 0 error
                if (isFinite(initialChiSq) && dof > 0) { redChiSq = initialChiSq / dof; aic = initialChiSq; bic = initialChiSq; if (totalPoints > 1) aicc = aic; else aicc = Infinity; }
            } catch(e) { logError(`Error calculating initial ChiSq/Residuals: ${e.message}`); }
            return { p_active: [], p_reconstructed: initialReconstructed, finalParamErrors: finalErrors, chiSquared: initialChiSq, covarianceMatrix: [], parameterErrors: [], iterations: 0, converged: true, activeParamLabels: [], totalPoints: totalPoints, degreesOfFreedom: dof, reducedChiSquared: redChiSq, aic: aic, aicc: aicc, bic: bic, error: null, residualsPerSeries: finalResiduals, fittedModelCurves: fittedCurves };
        }

        // --- Initial Calculations ---
        let activeParameters = [...activeInitialParams];
        let chiSquared = NaN;
        try { chiSquared = calculateChiSquaredGlobal(data, modelFunction, reconstructParams, activeParameters, robustCostFunction, paramStructure, logFn); }
        catch (error) { const errMsg = `Initial Chi-Squared calculation failed: ${error.message}`; logError(errMsg); return { ...baseErrorReturn, p_reconstructed: reconstructParams(activeParameters), error: errMsg }; }

        let converged = false; let covarianceMatrix = null; let parameterErrors = []; let iterationsPerformed = 0; let finalHessian = null;

        logInfo(`Total active parameters (K): ${K}`); logInfo(`Active Parameter Labels: ${activeParamLabels.join(', ')}`); logInfo(`Initial Active Parameters: ${activeParameters.map(p=>p.toExponential(3)).join(', ')}`); logInfo(`Initial Chi-Squared: ${chiSquared}`);

        if (!isFinite(chiSquared)) { const errMsg = "Initial Chi-Squared is not finite."; logError(errMsg); return { ...baseErrorReturn, p_reconstructed: reconstructParams(activeParameters), chiSquared: chiSquared, error: errMsg }; }

        // --- LM Iteration Loop ---
        let iteration;
        for (iteration = 0; iteration < maxIterations; iteration++) {
            iterationsPerformed = iteration + 1; logInfo(`--- Iteration ${iterationsPerformed} (Lambda: ${lambda.toExponential(3)}) ---`); logDebug(`Iter ${iterationsPerformed} - Current Active Params: ${activeParameters.map(p=>p.toExponential(5)).join(', ')}`);
            let jacobian, residuals; try { ({ jacobian, residuals } = calculateJacobianAndResidualsGlobal(data, modelFunction, reconstructParams, activeParameters, activeParamInfo, epsilon, paramStructure, logFn)); } catch (error) { const errMsg = `Jacobian failed: ${error.message}`; logError(`Error Jacobian/Resid iter ${iterationsPerformed}: ${errMsg}`); return { ...baseErrorReturn, p_reconstructed: reconstructParams(activeParameters), chiSquared: chiSquared, iterations: iterationsPerformed, error: errMsg }; }
            let gradient, currentHessian; try { gradient = calculateGradient(jacobian, residuals); currentHessian = calculateHessian(jacobian); /* Store hessian used for this step */ } catch (error) { const errMsg = `Grad/Hess failed: ${error.message}`; logError(`Error Grad/Hess iter ${iterationsPerformed}: ${errMsg}`); return { ...baseErrorReturn, p_reconstructed: reconstructParams(activeParameters), chiSquared: chiSquared, iterations: iterationsPerformed, error: errMsg }; }
            if (gradient.some(g => !isFinite(g)) || currentHessian.some(row => row.some(h => !isFinite(h)))) { const errMsg = "Non-finite grad/hess."; logError(`Non-finite grad/hess iter ${iterationsPerformed}.`); return { ...baseErrorReturn, p_reconstructed: reconstructParams(activeParameters), chiSquared: chiSquared, iterations: iterationsPerformed, error: errMsg }; }
            const maxGradient = Math.max(...gradient.map(Math.abs)); if (maxGradient < gradientTolerance) { logInfo(`Converged (grad tol ${gradientTolerance}). Max Grad: ${maxGradient.toExponential(3)}`); converged = true; break; } // Convergence Check 1
            logInfo(`Gradient: ${gradient.map(g => g.toExponential(3)).join(', ')}`);
            let parameterUpdates; let solveSuccess = false; let attempt = 0; const maxSolveAttempts = 5; let currentLambda = lambda;
            while (!solveSuccess && attempt < maxSolveAttempts) {
                const dampedHessian = currentHessian.map((row, i) => row.map((value, j) => (i === j ? value + currentLambda : value)));
                try { const negativeGradient = gradient.map(g => -g); parameterUpdates = solveLinearSystem(dampedHessian, negativeGradient); if (parameterUpdates.some(up => !isFinite(up))) throw new Error("NaN/Inf in updates."); solveSuccess = true; }
                catch (error) { attempt++; logWarn(`Solve failed (Att ${attempt}/${maxSolveAttempts}, Iter ${iterationsPerformed}): ${error.message}. Inc lambda.`); currentLambda = Math.min(currentLambda * lambdaIncreaseFactor * (attempt > 1 ? lambdaIncreaseFactor : 1) , 1e10); logInfo(`Attempting solve with Lambda: ${currentLambda.toExponential(3)}`); if (attempt >= maxSolveAttempts) { const errMsg = "Failed solve."; logError(`Failed solve after ${maxSolveAttempts} attempts.`); return { ...baseErrorReturn, p_reconstructed: reconstructParams(activeParameters), chiSquared: chiSquared, iterations: iterationsPerformed, error: errMsg }; } }
            }
            if (!solveSuccess) continue; lambda = currentLambda; logInfo(`Parameter Updates: ${parameterUpdates.map(pu => pu.toExponential(3)).join(', ')}`);
            const proposedActiveParams = activeParameters.map((p, i) => p + parameterUpdates[i]);
            let proposedReconstructed = reconstructParams(proposedActiveParams); // Initial proposal

            // Apply constraints (box and custom function)
            const { constrainedParams, changedActive } = applyConstraintsGlobal(
                proposedReconstructed, constraints, activeParamInfo, paramStructure, constraintFunction, logFn // Pass constraintFunction & logFn
            );
            proposedReconstructed = constrainedParams; // Use the potentially constrained parameters

            let finalProposedActiveParams = [...proposedActiveParams];
            if (changedActive) { const tempActive = []; activeParamInfo.forEach(info => { const [[ds, p], v] = info.originalCoord; tempActive.push(proposedReconstructed[ds][p][v]); }); finalProposedActiveParams = tempActive; logInfo("Constraints modified parameters."); }

            let newChiSquared = NaN; try { newChiSquared = calculateChiSquaredGlobal(data, modelFunction, reconstructParams, finalProposedActiveParams, robustCostFunction, paramStructure, logFn); } catch (error) { logError(`Error ChiSq proposed step iter ${iterationsPerformed}: ${error.message}`); newChiSquared = Infinity; }
            logInfo(`New Chi-Squared: ${newChiSquared}`);

            if (isFinite(newChiSquared) && newChiSquared < chiSquared) {
                const chiSquaredChange = chiSquared - newChiSquared; activeParameters = finalProposedActiveParams; chiSquared = newChiSquared; lambda = Math.max(lambda / lambdaDecreaseFactor, 1e-12);
                logInfo(`Accepted. ChiSq decreased by ${chiSquaredChange.toExponential(3)}. Lambda decreased to: ${lambda.toExponential(3)}`);
                // --- Post Progress Update ---
                try {
                    onProgress({ iteration: iterationsPerformed, chiSquared: chiSquared, lambda: lambda, activeParameters: [...activeParameters] });
                } catch (e) { logWarn(`Error in onProgress callback: ${e.message}`); }
                // --------------------------
                if (chiSquaredChange < errorTolerance) { logInfo(`Converged (chiSq tol ${errorTolerance}).`); converged = true; break; } // Convergence Check 2
            } else { lambda = Math.min(lambda * lambdaIncreaseFactor, 1e10); logInfo(`Rejected (ChiSq ${isNaN(newChiSquared) ? 'NaN' : 'increased/stagnant'}). Lambda increased to: ${lambda.toExponential(3)}`); if (lambda >= 1e10) logWarn("Lambda reached maximum limit."); }
        } // End of LM iteration loop

        // --- Post-Loop Processing & Statistics ---
        if (!converged && iteration === maxIterations) { logWarn(`lmFitGlobal did not converge within ${maxIterations} iterations.`); }

        logInfo("Recalculating final Jacobian/Hessian for covariance...");
        try { const { jacobian: finalJacobian } = calculateJacobianAndResidualsGlobal(data, modelFunction, reconstructParams, activeParameters, activeParamInfo, epsilon, paramStructure, logFn); finalHessian = calculateHessian(finalJacobian); }
        catch (error) { logError(`Failed to recalculate final Hessian: ${error.message}`); finalHessian = null; }

        let degreesOfFreedom = NaN, reducedChiSquared = NaN, aic = NaN, aicc = NaN, bic = NaN;
        if (isFinite(chiSquared) && totalPoints > 0) { degreesOfFreedom = totalPoints - K; if (degreesOfFreedom > 0) { reducedChiSquared = chiSquared / degreesOfFreedom; aic = chiSquared + 2 * K; const aiccCorrectionDenom = totalPoints - K - 1; if (aiccCorrectionDenom > 0) aicc = aic + (2 * K * (K + 1)) / aiccCorrectionDenom; else { aicc = Infinity; logWarn("AICc denominator (N-K-1) is zero or negative."); } bic = chiSquared + K * Math.log(totalPoints); } else { logWarn("Degrees of freedom is zero or negative."); aic = chiSquared + 2 * K; bic = chiSquared + K * Math.log(totalPoints); aicc = Infinity; } }
        else { logWarn("Final Chi-Squared is not finite. Cannot calculate statistics."); }

        parameterErrors = new Array(K).fill(NaN);
        if (finalHessian && K > 0) { try { covarianceMatrix = invertMatrixUsingSVD(finalHessian); parameterErrors = covarianceMatrix.map((row, i) => { const variance = row[i]; const scaledVariance = variance * (reducedChiSquared || 1.0); return isFinite(scaledVariance) ? Math.sqrt(Math.abs(scaledVariance)) : NaN; }); if (parameterErrors.some(isNaN)) { logWarn("NaN encountered in parameter errors."); } } catch (error) { logError(`Failed to calculate covariance matrix/parameter errors: ${error.message}`); covarianceMatrix = new Array(K).fill(0).map(() => new Array(K).fill(NaN)); parameterErrors = new Array(K).fill(NaN); } }
        else { if (K > 0) logWarn("Could not calculate covariance matrix (no valid final Hessian?)."); covarianceMatrix = new Array(K).fill(0).map(() => new Array(K).fill(NaN)); }

        const finalReconstructedParams = reconstructParams(activeParameters);

        let finalParamErrors = null; try { finalParamErrors = []; let currentFlatIdx = 0; paramStructure.forEach((dsStruct, dsIdx) => { finalParamErrors[dsIdx] = []; dsStruct.forEach((pLen, pIdx) => { finalParamErrors[dsIdx][pIdx] = []; for (let vIdx = 0; vIdx < pLen; vIdx++) { let errorValue = NaN; if (isFixed[currentFlatIdx]) { errorValue = 0; } else if (masterMap[currentFlatIdx] !== -1) { const masterFlatIdx = masterMap[currentFlatIdx]; const masterInfo = activeParamInfo.find(info => info.flatIndex === masterFlatIdx); if (masterInfo) { errorValue = parameterErrors[masterInfo.activeIndex]; } else { errorValue = 0; } } else { const activeInfo = activeParamInfo.find(info => info.flatIndex === currentFlatIdx); if (activeInfo) { errorValue = parameterErrors[activeInfo.activeIndex]; } else { logWarn(`Could not find active info for supposedly active parameter at flat index ${currentFlatIdx}`); } } finalParamErrors[dsIdx][pIdx][vIdx] = errorValue; currentFlatIdx++; } }); }); } catch (e) { logError(`Error constructing finalParamErrors structure: ${e.message}`); finalParamErrors = null; }

        let finalResiduals = null; try { finalResiduals = calculateFinalResiduals(data, modelFunction, finalReconstructedParams, logFn); } catch (error) { logError(`Failed to calculate final residuals: ${error.message}`); }
        let fittedCurves = null; if (shouldCalculateFitted) { logInfo(`Calculating fitted model curves with ${numPointsForCurve} points...`); try { fittedCurves = calculateFittedModelCurves(data, modelFunction, finalReconstructedParams, numPointsForCurve, logFn); } catch (error) { logError(`Failed to calculate fitted model curves: ${error.message}`); } }

        logInfo("--------------------"); logInfo("lmFitGlobal Finished."); logInfo(`Iterations Performed: ${iterationsPerformed}`); logInfo(`Total Points (N): ${totalPoints}`); logInfo(`Active Parameters (K): ${K}`); logInfo(`Degrees of Freedom (N-K): ${degreesOfFreedom}`); logInfo(`Final Active Parameters: ${activeParameters.map(p=>p.toExponential(5)).join(', ')}`); logInfo(`Final Chi-Squared: ${chiSquared}`); logInfo(`Reduced Chi-Squared: ${reducedChiSquared}`); logInfo(`AIC: ${aic}`); logInfo(`AICc: ${aicc}`); logInfo(`BIC: ${bic}`); logInfo(`Parameter Errors (Active): ${parameterErrors.map(p=>isNaN(p)?'NaN':p.toExponential(3)).join(', ')}`); logInfo(`Converged: ${converged}`); logInfo("--------------------");

        return {
            p_active: activeParameters, p_reconstructed: finalReconstructedParams, finalParamErrors,
            chiSquared, covarianceMatrix, parameterErrors, iterations: iterationsPerformed, converged,
            activeParamLabels, error: null, totalPoints, degreesOfFreedom, reducedChiSquared,
            aic, aicc, bic, residualsPerSeries: finalResiduals, fittedModelCurves: fittedCurves
        };
    } // <-- End of lmFitGlobal function definition


    // ============================================================================
    // Helper Functions for Wrappers
    // ============================================================================

     /**
     * Checks if the input likely represents a single dataset.
     * Simple check based on data.x being a 1D array.
     * @param {object} data - The data object.
     * @returns {boolean} True if likely single dataset format.
     * @private
     */
     function isSingleDataset(data) {
         return data && Array.isArray(data.x) && (data.x.length === 0 || !Array.isArray(data.x[0]));
     }

    /**
     * Wraps a single dataset's inputs into the nested format required by lmFitGlobal.
     * @param {object} data - Single dataset {x: number[], y: number[], ye: number[]}.
     * @param {Function[]} modelFunction - Array of model functions for the dataset.
     * @param {number[][]} initialParameters - Array of parameter arrays, e.g., [[p1], [p2a, p2b]].
     * @param {object} options - Original options object.
     * @returns {object} Wrapped inputs { data, modelFunction, initialParameters, options }
     * @private
     */
    function wrapSingleDatasetInput(data, modelFunction, initialParameters, options) {
        const wrappedData = {
            x: [data.x],
            y: [data.y],
            ye: [data.ye]
        };
        // Ensure modelFunction is an array of functions, even if only one model
        const wrappedModelFunction = [ Array.isArray(modelFunction) ? modelFunction : [modelFunction] ];
        // Ensure initialParameters matches the structure [[ [...], [...], ... ]]
        const wrappedInitialParameters = [ initialParameters ];

        // Wrap fixMap and constraints if they exist and are not already wrapped
        const wrappedOptions = { ...options }; // Shallow copy options
        if (options.fixMap && !Array.isArray(options.fixMap[0])) {
            wrappedOptions.fixMap = [options.fixMap];
        }
        if (options.constraints && !Array.isArray(options.constraints[0])) {
            wrappedOptions.constraints = [options.constraints];
        }
        // linkMap doesn't make sense for a single dataset, remove it
        delete wrappedOptions.linkMap;

        return {
            data: wrappedData,
            modelFunction: wrappedModelFunction,
            initialParameters: wrappedInitialParameters,
            options: wrappedOptions
        };
    }

    // ============================================================================
    // User-Facing Wrapper Functions
    // ============================================================================

    /**
     * Fits a single dataset using the Levenberg-Marquardt algorithm.
     * This is a convenience wrapper around lmFitGlobal for simpler use cases.
     *
     * @param {object} data - Contains the experimental data for ONE dataset {x: number[], y: number[], ye: number[]}.
     * @param {Function | Function[]} modelFunction - A single model function or an array of model functions for this dataset.
     * @param {number[] | number[][]} initialParameters - Initial parameter guesses. If multiple models, use nested array: [[p1a, p1b], [p2a]]. If single model, can be flat array: [p1a, p1b].
     * @param {object} [options={}] - Optional configuration (see lmFitGlobal docs, but linkMap is ignored).
     *                                 fixMap and constraints should be provided in the single-dataset format (e.g., `fixMap = [[false, true], [false]]`).
     * @returns {object} The fitting result object (see lmFitGlobal docs). Structure is the same, but arrays like residualsPerSeries will only have one element.
     */
    function lmFit(data, modelFunction, initialParameters, options = {}) {
         // Input validation
         if (!isSingleDataset(data)) {
             throw new Error("lmFit requires single dataset input format (e.g., data.x as a 1D array). Use lmFitGlobal or lmFitIndependent for multiple datasets.");
         }
         // Normalize modelFunction to array
         const modelFuncArray = Array.isArray(modelFunction) ? modelFunction : [modelFunction];
         // Normalize initialParameters to nested array structure
         const initialParamsNested = modelFuncArray.length === 1 && !Array.isArray(initialParameters[0])
             ? [initialParameters] // Single model, flat params -> [[p1, p2, ...]]
             : initialParameters; // Assumed already nested [[p1a, p1b], [p2a]]

         // Wrap inputs for lmFitGlobal
         const {
             data: wrappedData,
             modelFunction: wrappedModelFunc,
             initialParameters: wrappedInitialParams,
             options: wrappedOptions
         } = wrapSingleDatasetInput(data, modelFuncArray, initialParamsNested, options);

         // Call the global fitter
         const result = lmFitGlobal(wrappedData, wrappedModelFunc, wrappedInitialParams, wrappedOptions);

         // Return the result (structure is already suitable)
         return result;
    }


    /**
     * Fits multiple datasets independently (sequentially) using lmFitGlobal.
     * Useful when parameters are not shared or linked across datasets.
     *
     * @param {object} data - Contains the experimental data for MULTIPLE datasets {x: number[][], y: number[][], ye: number[][]}.
     * @param {Function[][]} modelFunction - Array of arrays of model functions, matching data structure.
     * @param {number[][][]} initialParameters - Nested array of initial parameter guesses, matching data structure.
     * @param {object} [options={}] - Optional configuration. Includes most lmFitGlobal options.
     *                                 `linkMap` is ignored. `fixMap` and `constraints` apply per-dataset.
     *                                 `onLog` and `onProgress` receive an additional `datasetIndex` argument.
     * @returns {object[]} An array of fitting result objects (one per dataset, in the original order).
     *                     Individual fit errors are in the 'error' property of each result object.
     */
    function lmFitIndependent(data, modelFunction, initialParameters, options = {}) {
        // Handle potential single dataset input gracefully
        if (isSingleDataset(data)) {
            console.warn("lmFitIndependent received single dataset input. Calling lmFit instead.");
            try {
                // Need to adjust modelFunction and initialParameters format for lmFit call
                 const modelFuncArray = Array.isArray(modelFunction) ? modelFunction : [modelFunction];
                 const initialParamsInput = (modelFuncArray.length === 1 && !Array.isArray(initialParameters[0]))
                     ? initialParameters // Assume flat array for single model
                     : initialParameters[0]; // Use the first element if nested structure was passed
                 const modelFuncInput = modelFuncArray.length === 1 ? modelFuncArray[0] : modelFuncArray[0]; // Use first element

                const result = lmFit(data, modelFuncInput, initialParamsInput, options);
                return [result]; // Return result wrapped in an array
            } catch (e) {
                 // Return error state consistent with multi-dataset failure
                 return [{ error: `Error processing single dataset input for lmFitIndependent: ${e.message}`, converged: false }];
            }
        }

        const numDatasets = data.x.length;
        if (numDatasets === 0) return []; // No work to do

        const allResults = [];
        const rootOnLog = options.onLog || (() => {});
        const rootOnProgress = options.onProgress || (() => {});

        // --- Sequential Loop ---
        for (let i = 0; i < numDatasets; i++) {
            const datasetIndex = i; // Capture index for callbacks
            console.log(`--- Starting Independent Fit for Dataset ${datasetIndex} ---`); // Console log for clarity

            // Prepare single dataset inputs
            const singleData = { x: data.x[i], y: data.y[i], ye: data.ye[i] };
            const singleModelFunc = modelFunction[i];
            const singleInitialParams = initialParameters[i];

            // Prepare options for this specific task
            const singleOptions = { ...options };
            if (options.fixMap) singleOptions.fixMap = options.fixMap[i];
            if (options.constraints) singleOptions.constraints = options.constraints[i];
            delete singleOptions.linkMap; // Ignored
            delete singleOptions.onLog; // Use wrapper
            delete singleOptions.onProgress; // Use wrapper

            // Wrap callbacks to include dataset index
            singleOptions.onLog = (message, level) => {
                rootOnLog(message, level, datasetIndex);
            };
            singleOptions.onProgress = (progressData) => {
                rootOnProgress(progressData, datasetIndex);
            };

            // Wrap inputs for lmFitGlobal call
            const {
                data: wrappedData,
                modelFunction: wrappedModelFunc,
                initialParameters: wrappedInitialParams,
                options: wrappedOptions // Use the options prepared for this single dataset
            } = wrapSingleDatasetInput(singleData, singleModelFunc, singleInitialParams, singleOptions);

            try {
                // Call lmFitGlobal for the single wrapped dataset
                const result = lmFitGlobal(wrappedData, wrappedModelFunc, wrappedInitialParams, wrappedOptions);
                allResults.push(result);
            } catch (error) {
                console.error(`Error fitting dataset ${datasetIndex}:`, error);
                rootOnLog(`Fit failed for dataset ${datasetIndex}: ${error.message}`, 'error', datasetIndex);
                // Push an error object into the results array
                allResults.push({
                    error: `Fit failed: ${error.message}`,
                    converged: false,
                    // Include other default fields to maintain structure
                    p_active: [], p_reconstructed: singleInitialParams, finalParamErrors: null, chiSquared: NaN,
                    covarianceMatrix: null, parameterErrors: [], iterations: 0, activeParamLabels: [],
                    totalPoints: singleData.x?.length ?? 0, degreesOfFreedom: NaN, reducedChiSquared: NaN,
                    aic: NaN, aicc: NaN, bic: NaN, residualsPerSeries: null, fittedModelCurves: null
                });
            }
            console.log(`--- Finished Independent Fit for Dataset ${datasetIndex} ---`);
        } // End sequential loop

        return allResults;
    }


    // Expose public functions
    global.lmFitGlobal = lmFitGlobal;
    global.lmFit = lmFit;
    global.lmFitIndependent = lmFitIndependent;

// Establish the root object, `window` in the browser, or `global` on the server.
})(typeof window !== 'undefined' ? window : global); // <-- End of the IIFE wrapper