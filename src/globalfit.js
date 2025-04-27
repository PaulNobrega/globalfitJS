/**
 * @fileoverview Provides Levenberg-Marquardt fitting functions:
 *   - lmFitGlobal: For simultaneous fitting of multiple datasets with linking/fixing.
 *   - lmFit: A wrapper for fitting a single dataset using lmFitGlobal.
 *   - lmFitIndependent: Sequentially fits multiple datasets independently using lmFitGlobal.
 * Includes helpers for parameter mapping, chi-squared calculation, Jacobian calculation,
 * constraint application, statistics calculation, and optional confidence interval bands for fitted curves.
 * Depends on svd.js for linear algebra operations.
 * Version: 1.2.8 (Adds model_x_range option)
 */

(function(global) {
    'use strict';

    // Check if SVD helpers are loaded from svd.js
    if (!global.LM_SVD_HELPERS) {
        throw new Error("Error: svd.js must be loaded before globalfit.js");
    }

    // Import helpers from the global object created by svd.js
    const { solveLinearSystem, invertMatrixUsingSVD } = global.LM_SVD_HELPERS;

    // ============================================================================
    // Constants and Logging Levels
    // ============================================================================

    const LOG_LEVELS = { none: 0, error: 1, warn: 2, info: 3, debug: 4 };

    // ============================================================================
    // Logging Utility
    // ============================================================================

    function createLogger(logFn, logLevel) {
        return {
            log: (message, level) => {
                const messageLevel = LOG_LEVELS[level] ?? LOG_LEVELS.info;
                if (logLevel >= messageLevel) {
                    logFn(message, level);
                }
            },
            info: (message) => logFn(message, 'info'),
            warn: (message) => logFn(message, 'warn'),
            error: (message) => logFn(message, 'error'),
            debug: (message) => logFn(message, 'debug'),
        };
    }

    // ============================================================================
    // Statistical Helper Functions (Adapted from JStat)
    // ============================================================================

    const IBETA_EPS = 1e-14; // Epsilon for beta functions
    const IBETA_MAX_ITER = 150; // Max iterations for beta functions

    /** Log-gamma function (Lanczos approximation). @private */
    function _jstat_gammaln(x) {
        const lanczosCoeffs = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]; const g = 7;
        if (x <= 0) return Infinity; if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - _jstat_gammaln(1 - x); const x_adj = x - 1.0; let series = lanczosCoeffs[0]; const tmp = x_adj + g + 0.5; for (let i = 1; i < lanczosCoeffs.length; i++) series += lanczosCoeffs[i] / (x_adj + i); return (x_adj + 0.5) * Math.log(tmp) - tmp + Math.log(Math.sqrt(2 * Math.PI) * series);
    }

    /** Continued fraction evaluation for the incomplete beta function. @private */
    function _jstat_ibeta_cont_frac(x, a, b) {
        let f = 1.0, C = f, D = 0.0, delta = 0.0, iter = 0;
        do { iter++; const m = iter - 1, m2 = 2 * m; let d1_num = -(a + m) * (a + b + m) * x, d1_den = (a + m2) * (a + m2 + 1); let d1 = (d1_den === 0) ? 0 : d1_num / d1_den; let d2 = 0; if (m > 0) { let d2_num = m * (b - m) * x, d2_den = (a + m2 - 1) * (a + m2); d2 = (d2_den === 0) ? 0 : d2_num / d2_den; } D = 1.0 + d1 * D; if (Math.abs(D) < IBETA_EPS) D = IBETA_EPS; C = 1.0 + d1 / C; if (Math.abs(C) < IBETA_EPS) C = IBETA_EPS; D = 1.0 / D; delta = C * D; f *= delta; if (m > 0) { D = 1.0 + d2 * D; if (Math.abs(D) < IBETA_EPS) D = IBETA_EPS; C = 1.0 + d2 / C; if (Math.abs(C) < IBETA_EPS) C = IBETA_EPS; D = 1.0 / D; delta = C * D; f *= delta; } if (Math.abs(delta - 1.0) < IBETA_EPS) break; } while (iter < IBETA_MAX_ITER); if (iter >= IBETA_MAX_ITER) console.warn('_jstat_ibeta_cont_frac: Failed to converge.'); return f;
    }

    /** Regularized Incomplete Beta function I_x(a,b). @private */
    function _jstat_ibeta(x, a, b) {
        if (x < 0 || x > 1 || a <= 0 || b <= 0) return NaN; if (x === 0) return 0; if (x === 1) return 1; const logBeta = _jstat_gammaln(a) + _jstat_gammaln(b) - _jstat_gammaln(a + b); const logPowerFactor = a * Math.log(x) + b * Math.log(1 - x); const frontFactor = Math.exp(logPowerFactor - logBeta); if (x < (a + 1) / (a + b + 2)) return frontFactor * _jstat_ibeta_cont_frac(x, a, b) / a; else return 1.0 - frontFactor * _jstat_ibeta_cont_frac(1 - x, b, a) / b;
    }

    /** Inverse of the regularized incomplete beta function I_x(a, b) = p. @private */
    function _jstat_ibeta_inv(p, a, b) {
        if (p < 0 || p > 1 || a <= 0 || b <= 0) return NaN; if (p === 0) return 0; if (p === 1) return 1; const INV_EPS = 1e-12, MAX_ITER_INV = 100; let x_low = 0.0, x_high = 1.0, x = 0.5, dx = 0.5; x = (p < 0.5) ? Math.pow(p * a * Math.exp(_jstat_gammaln(a) + _jstat_gammaln(b) - _jstat_gammaln(a + b)), 1 / a) : 1 - Math.pow((1 - p) * b * Math.exp(_jstat_gammaln(a) + _jstat_gammaln(b) - _jstat_gammaln(a + b)), 1 / b); if (x <= 0) x = INV_EPS; if (x >= 1) x = 1.0 - INV_EPS; const logBeta = _jstat_gammaln(a) + _jstat_gammaln(b) - _jstat_gammaln(a + b);
        for (let iter = 0; iter < MAX_ITER_INV; iter++) { const f = _jstat_ibeta(x, a, b) - p; let df_log = (a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - logBeta; let df = Math.exp(df_log); dx = (df !== 0 && isFinite(df)) ? f / df : (f > 0 ? -INV_EPS * 10 : INV_EPS * 10); let x_new = x - dx; if (f > 0) x_high = x; else x_low = x; if (x_new <= x_low || x_new >= x_high) { x_new = (x_low + x_high) / 2.0; dx = x - x_new; } x = x_new; if (Math.abs(dx) < INV_EPS || Math.abs(f) < INV_EPS) return x; } console.warn(`_jstat_ibeta_inv: Failed to converge. p=${p}, a=${a}, b=${b}. Est: ${x}`); return x;
    }

    /** Inverse of the Student's t-distribution CDF (quantile function). @private */
    function _jstat_studentt_inv(p, df) {
        if (p < 0 || p > 1 || df <= 0 || !isFinite(df)) return NaN; if (p === 0) return -Infinity; if (p === 1) return Infinity; if (p === 0.5) return 0; let p_beta = (p < 0.5) ? 2.0 * p : 2.0 * (1.0 - p); let x = _jstat_ibeta_inv(p_beta, 0.5 * df, 0.5); if (isNaN(x) || x === 0) return (p < 0.5) ? -Infinity : Infinity; if (x === 1) return 0; let t_squared = df * (1.0 - x) / x; if (t_squared < 0) { console.warn(`_jstat_studentt_inv: Negative t_squared (${t_squared}).`); return NaN; } let t = Math.sqrt(t_squared); return (p < 0.5) ? -t : t;
    }

    /** Calculates the critical value from the Student's t-distribution. @private */
    function calculateTValue(probability, dof, logFn) {
        if (dof <= 0) { logFn(`Cannot calculate t-value: Degrees of freedom (${dof}) must be positive.`, 'warn'); return NaN; }
        if (probability <= 0 || probability >= 1) { logFn(`Cannot calculate t-value: Cumulative probability (${probability}) must be between 0 and 1.`, 'warn'); return NaN; }
        try { const tVal = _jstat_studentt_inv(probability, dof); if (!isFinite(tVal)) { logFn(`Calculated t-value is not finite (${tVal}) for p=${probability}, dof=${dof}.`, 'warn'); return NaN; } return tVal; }
        catch (e) { logFn(`Error calculating t-value: ${e.message}`, 'error'); return NaN; }
    }

    // ============================================================================
    // Fitting Algorithm Helpers (Internal) - Includes existing helpers
    // ============================================================================

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

    function setupParameterMapping(initialParameters, linkMapInput, fixMapInput, logFn) {
        logFn("--- Running setupParameterMapping (New linkMap structure v1.4.3 logic) ---", 'debug');
        logFn(`Received linkMapInput (structure like fixMap): ${linkMapInput ? 'Provided' : 'Null'}`, 'debug');
        logFn(`Received fixMapInput: ${JSON.stringify(fixMapInput)}`, 'debug');

        // --- 1. Flatten Parameters and Create Coordinate Mapping ---
        const paramStructure = [];
        const flatInitialParams = [];
        const paramCoordinates = []; // Stores [[dsIdx, pIdx], vIdx] for each flat index
        let currentFlatIndex = 0;
        const fixMap = fixMapInput ? JSON.parse(JSON.stringify(fixMapInput)) : []; // Deep copy or empty
        const linkMap = linkMapInput ? JSON.parse(JSON.stringify(linkMapInput)) : null; // Deep copy or null

        initialParameters.forEach((dsParams, dsIdx) => {
            paramStructure.push([]);
            if (!fixMap[dsIdx]) fixMap[dsIdx] = [];
            dsParams.forEach((pArray, pIdx) => {
                paramStructure[dsIdx].push(pArray.length);
                if (!fixMap[dsIdx][pIdx]) fixMap[dsIdx][pIdx] = new Array(pArray.length).fill(false);
                pArray.forEach((pValue, vIdx) => {
                    flatInitialParams.push(pValue);
                    // Store coordinates based on the ORIGINAL initialParameters structure
                    paramCoordinates.push([[dsIdx, pIdx], vIdx]);
                    if (fixMap[dsIdx][pIdx].length <= vIdx) fixMap[dsIdx][pIdx][vIdx] = false;
                    currentFlatIndex++;
                });
            });
        });

        const nTotalParams = flatInitialParams.length;
        const masterMap = new Array(nTotalParams).fill(-1); // Ensure initialized
        const isFixed = new Array(nTotalParams).fill(false); // Ensure initialized
        const activeParamInfo = []; const activeInitialParams = []; const activeParamLabels = [];

        // --- 2. Apply fixMap ---
        // Use paramCoordinates which maps flatIdx -> original coordinate
        paramCoordinates.forEach((coord, flatIdx) => {
            const [[dsIdx, paramIdx], valIdx] = coord;
            if (fixMap[dsIdx]?.[paramIdx]?.[valIdx] === true) {
                isFixed[flatIdx] = true;
            }
        });
        logFn(`Initial isFixed array (after fixMap): ${JSON.stringify(isFixed)}`, 'debug');

        // --- 3. Process linkMap (NEW STRUCTURE LOGIC - v1.4.3 style) ---
        const linkGroupsById = {}; // { groupId: [flatIdx1, flatIdx2, ...], ... }

        if (linkMap) {
            // Iterate through the structure of linkMapInput, which should match the structure of initialParameters
            linkMap.forEach((dsLinkMap, dsLinkIdx) => {
                if (!dsLinkMap || dsLinkIdx >= initialParameters.length) return; // Skip if dataset doesn't exist in params
                dsLinkMap.forEach((modelLinkMap, modelLinkIdx) => {
                    if (!modelLinkMap || modelLinkIdx >= initialParameters[dsLinkIdx].length) return; // Skip if model group doesn't exist
                    modelLinkMap.forEach((groupId, paramLinkIdx) => {
                         if (paramLinkIdx >= initialParameters[dsLinkIdx][modelLinkIdx].length) return; // Skip if param index is out of bounds

                        if (groupId !== null && groupId !== undefined && groupId !== '') {
                            // Find the corresponding flatIndex using paramCoordinates by matching coordinates
                            const targetCoord = [dsLinkIdx, modelLinkIdx, paramLinkIdx];
                            const flatIdx = paramCoordinates.findIndex(pc =>
                                pc[0][0] === targetCoord[0] &&
                                pc[0][1] === targetCoord[1] &&
                                pc[1] === targetCoord[2]
                            );

                            if (flatIdx !== -1) {
                                if (!linkGroupsById[groupId]) {
                                    linkGroupsById[groupId] = [];
                                }
                                linkGroupsById[groupId].push(flatIdx);
                            } else {
                                // This case should ideally not happen if initialParameters and linkMapInput are consistent
                                logFn(`Could not find flat index for link coordinate [${targetCoord.join(', ')}]`, 'warn');
                            }
                        }
                    });
                });
            });
            logFn(`Collected Link Groups by ID: ${JSON.stringify(linkGroupsById)}`, 'debug'); // Log collected groups

            // Process collected groups
            Object.entries(linkGroupsById).forEach(([groupId, flatIndices]) => {
                if (flatIndices.length < 2) {
                    logFn(`Link Group ${groupId} has only one member (${flatIndices[0]}). Ignoring link.`, 'debug');
                    return;
                }
                logFn(`Processing Link Group ${groupId}: Flat indices ${JSON.stringify(flatIndices)}`, 'debug');

                let masterFlatIndex = -1;
                let masterCoord = null;
                let allInitiallyFixed = true;

                for (const flatIdx of flatIndices) {
                    if (flatIdx < 0 || flatIdx >= nTotalParams) { logFn(`  Invalid flat index ${flatIdx} found in link group ${groupId}. Skipping.`, 'warn'); continue; }
                    if (!isFixed[flatIdx]) {
                        allInitiallyFixed = false;
                        if (masterFlatIndex === -1) { masterFlatIndex = flatIdx; masterCoord = paramCoordinates[flatIdx]; }
                    }
                }

                if (allInitiallyFixed) {
                    masterFlatIndex = flatIndices[0];
                    if (masterFlatIndex < 0 || masterFlatIndex >= nTotalParams) { logFn(`  Invalid nominal master index ${masterFlatIndex} for fixed group ${groupId}. Skipping group.`, 'warn'); return; }
                    logFn(`  Link Group ${groupId} consists entirely of initially fixed parameters. Nominal master: flat index ${masterFlatIndex}`, 'debug');
                    if (!isFixed[masterFlatIndex]) {
                         logFn(`  Marking nominal master ${JSON.stringify(paramCoordinates[masterFlatIndex])} (flat ${masterFlatIndex}) as fixed because entire group is fixed.`, 'debug');
                         isFixed[masterFlatIndex] = true;
                    }
                    flatIndices.forEach(flatIdx => {
                         if (flatIdx < 0 || flatIdx >= nTotalParams) return;
                        if (flatIdx !== masterFlatIndex) {
                            masterMap[flatIdx] = masterFlatIndex;
                            flatInitialParams[flatIdx] = flatInitialParams[masterFlatIndex];
                            logFn(`    Linked fixed slave ${JSON.stringify(paramCoordinates[flatIdx])} (flat ${flatIdx}) to fixed master ${masterFlatIndex}. State: isFixed=${isFixed[flatIdx]}, masterMap=${masterMap[flatIdx]}`, 'debug');
                        } else {
                             logFn(`    Nominal Master ${JSON.stringify(paramCoordinates[flatIdx])} (flat ${flatIdx}). State: isFixed=${isFixed[flatIdx]}, masterMap=${masterMap[flatIdx]}`, 'debug');
                        }
                    });
                }
                else if (masterFlatIndex !== -1) {
                    logFn(`  Link Group ${groupId} master: ${JSON.stringify(masterCoord)} (flat index ${masterFlatIndex})`, 'debug');
                    logFn(`    Master ${JSON.stringify(paramCoordinates[masterFlatIndex])} (flat ${masterFlatIndex}). State: isFixed=${isFixed[masterFlatIndex]}, masterMap=${masterMap[masterFlatIndex]}`, 'debug');
                    flatIndices.forEach(flatIdx => {
                         if (flatIdx < 0 || flatIdx >= nTotalParams) return;
                        if (flatIdx !== masterFlatIndex) {
                            const initiallyFixed = isFixed[flatIdx];
                            if (isFixed[flatIdx]) {
                                logFn(`    Linking overrides fixed status for ${JSON.stringify(paramCoordinates[flatIdx])} (flat ${flatIdx}).`, 'warn');
                                isFixed[flatIdx] = false;
                            }
                            masterMap[flatIdx] = masterFlatIndex;
                            flatInitialParams[flatIdx] = flatInitialParams[masterFlatIndex];
                            logFn(`    Linked slave ${JSON.stringify(paramCoordinates[flatIdx])} (flat ${flatIdx}) to master ${masterFlatIndex}. InitialFixed=${initiallyFixed}. Final State: isFixed=${isFixed[flatIdx]}, masterMap=${masterMap[flatIdx]}`, 'debug');
                        }
                    });
                } else {
                     logFn(`  Internal Logic Error: Could not determine master type for group ${groupId}.`, 'error');
                }
            });
        } else {
            logFn("No linkMap provided or linkMap is null.", 'debug');
        }

        logFn(`Final isFixed array (after linking): ${JSON.stringify(isFixed)}`, 'debug');
        logFn(`Final masterMap array: ${JSON.stringify(masterMap)}`, 'debug');

        // --- 4. Identify active parameters ---
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

        // --- 5. Create the reconstruction function ---
        const reconstructParams = (activeParamsCurrent) => {
            if (activeParamsCurrent.length !== totalActiveParams) throw new Error(`reconstructParams expects ${totalActiveParams} params, received ${activeParamsCurrent.length}`);
            const reconstructedFlat = [...flatInitialParams];
            activeParamInfo.forEach((info, actIdx) => { if (actIdx >= activeParamsCurrent.length) throw new Error(`Mismatch activeParamInfo/activeParams.`); reconstructedFlat[info.flatIndex] = activeParamsCurrent[actIdx]; });
            for (let i = 0; i < nTotalParams; i++) { if (masterMap[i] !== -1) { if (masterMap[i] < 0 || masterMap[i] >= reconstructedFlat.length) throw new Error(`Master index ${masterMap[i]} out of bounds.`); reconstructedFlat[i] = reconstructedFlat[masterMap[i]]; } }
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

        return {
            activeInitialParams, reconstructParams, activeParamInfo, totalActiveParams,
            paramStructure, activeParamLabels,
            isFixed, masterMap, paramCoordinates, nTotalParams
        };
    }

    /**
     * Helper function to get details needed for error reconstruction.
     * Uses NEW linkMap structure logic. (v1.4.3 style)
     * @private
     */
    function setupParameterMappingDetails(initialParameters, linkMapInput, fixMapInput) {
        const paramStructure = [];
        const paramCoordinates = [];
        let currentFlatIndex = 0;
        const fixMap = fixMapInput ? JSON.parse(JSON.stringify(fixMapInput)) : [];
        const linkMap = linkMapInput ? JSON.parse(JSON.stringify(linkMapInput)) : null;

        initialParameters.forEach((dsParams, dsIdx) => {
            paramStructure.push([]);
            if (!fixMap[dsIdx]) fixMap[dsIdx] = [];
            dsParams.forEach((pArray, pIdx) => {
                paramStructure[dsIdx].push(pArray.length);
                if (!fixMap[dsIdx][pIdx]) fixMap[dsIdx][pIdx] = new Array(pArray.length).fill(false);
                pArray.forEach((pValue, vIdx) => {
                    paramCoordinates.push([[dsIdx, pIdx], vIdx]);
                    if (fixMap[dsIdx][pIdx].length <= vIdx) fixMap[dsIdx][pIdx][vIdx] = false;
                    currentFlatIndex++;
                });
            });
        });

        const nTotalParams = currentFlatIndex;
        const masterMap = new Array(nTotalParams).fill(-1);
        const isFixed = new Array(nTotalParams).fill(false);

        paramCoordinates.forEach((coord, flatIdx) => {
            const [[dsIdx, paramIdx], valIdx] = coord;
            if (fixMap[dsIdx]?.[paramIdx]?.[valIdx] === true) isFixed[flatIdx] = true;
        });

        const linkGroupsById = {};
        if (linkMap) {
            linkMap.forEach((dsLinkMap, dsLinkIdx) => {
                if (!dsLinkMap || dsLinkIdx >= initialParameters.length) return;
                dsLinkMap.forEach((modelLinkMap, modelLinkIdx) => {
                    if (!modelLinkMap || modelLinkIdx >= initialParameters[dsLinkIdx].length) return;
                    modelLinkMap.forEach((groupId, paramLinkIdx) => {
                        if (paramLinkIdx >= initialParameters[dsLinkIdx][modelLinkIdx].length) return;
                        if (groupId !== null && groupId !== undefined && groupId !== '') {
                            const targetCoord = [dsLinkIdx, modelLinkIdx, paramLinkIdx];
                            const flatIdx = paramCoordinates.findIndex(pc =>
                                pc[0][0] === targetCoord[0] &&
                                pc[0][1] === targetCoord[1] &&
                                pc[1] === targetCoord[2]
                            );
                            if (flatIdx !== -1) {
                                if (!linkGroupsById[groupId]) linkGroupsById[groupId] = [];
                                linkGroupsById[groupId].push(flatIdx);
                            }
                        }
                    });
                });
            });

            Object.values(linkGroupsById).forEach(flatIndices => {
                if (flatIndices.length < 2) return;
                let masterFlatIndex = -1;
                let allInitiallyFixed = true;
                for (const flatIdx of flatIndices) {
                    if (flatIdx < 0 || flatIdx >= nTotalParams) continue;
                    if (!isFixed[flatIdx]) {
                        allInitiallyFixed = false;
                        if (masterFlatIndex === -1) masterFlatIndex = flatIdx;
                    }
                }
                if (allInitiallyFixed) {
                    masterFlatIndex = flatIndices[0];
                    if (masterFlatIndex < 0 || masterFlatIndex >= nTotalParams) return;
                    if (!isFixed[masterFlatIndex]) isFixed[masterFlatIndex] = true;
                    flatIndices.forEach(flatIdx => {
                        if (flatIdx < 0 || flatIdx >= nTotalParams) return;
                        if (flatIdx !== masterFlatIndex) masterMap[flatIdx] = masterFlatIndex;
                    });
                } else if (masterFlatIndex !== -1) {
                    flatIndices.forEach(flatIdx => {
                        if (flatIdx < 0 || flatIdx >= nTotalParams) return;
                        if (flatIdx !== masterFlatIndex) {
                            if (isFixed[flatIdx]) isFixed[flatIdx] = false;
                            masterMap[flatIdx] = masterFlatIndex;
                        }
                    });
                }
            });
        }

        return { isFixed, masterMap, paramCoordinates, nTotalParams, paramStructure };
    }

    function calculateChiSquaredGlobal(data, modelFunction, reconstructParamsFunc, activeParams, robustCostFunction, paramStructure, logFn) { /* ... existing code ... */
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
                else if (robustCostFunction === 2) chiSquared += Math.log(1 + 0.5 (residual * residual));
                else chiSquared += residual * residual;
            }
        });
        if (!isFinite(chiSquared)) { logFn("Non-finite chi-squared calculated.", 'error'); return Infinity; }
        return chiSquared;
    }

    function calculateJacobianAndResidualsGlobal(data, modelFunction, reconstructParamsFunc, activeParams, activeParamInfo, epsilon, paramStructure, logFn) { /* ... existing code ... */
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

    function applyConstraintsGlobal(reconstructedParams, constraints, activeParamInfo, paramStructure, constraintFunction, logFn) { /* ... existing code ... */
        let constrainedParams = reconstructedParams; let changedActive = false; let boxConstraintApplied = false;
        if (constraints) {
            let paramsCopy = null;
            paramStructure.forEach((dsStruct, dsIdx) => {
                if (!constraints[dsIdx]) return;
                dsStruct.forEach((pLen, pIdx) => {
                    if (!constraints[dsIdx][pIdx]) return;
                    for (let vIdx = 0; vIdx < pLen; vIdx++) {
                        const constraint = constraints[dsIdx]?.[pIdx]?.[vIdx];
                        if (constraint) {
                            if (!paramsCopy) paramsCopy = JSON.parse(JSON.stringify(reconstructedParams));
                            let value = paramsCopy[dsIdx][pIdx][vIdx]; const originalValue = value;
                            if (constraint.min !== undefined && value < constraint.min) value = constraint.min;
                            if (constraint.max !== undefined && value > constraint.max) value = constraint.max;
                            if (value !== originalValue) {
                                paramsCopy[dsIdx][pIdx][vIdx] = value; boxConstraintApplied = true;
                                const isActive = activeParamInfo.some(info => info.originalCoord[0][0] === dsIdx && info.originalCoord[0][1] === pIdx && info.originalCoord[1] === vIdx);
                                if (isActive) changedActive = true;
                            }
                        }
                    }
                });
            });
            if (paramsCopy) constrainedParams = paramsCopy;
        }
        if (typeof constraintFunction === 'function') {
            try {
                const paramsForCustomFn = boxConstraintApplied ? constrainedParams : reconstructedParams;
                const paramsBeforeCustom = boxConstraintApplied ? null : JSON.stringify(paramsForCustomFn);
                constrainedParams = constraintFunction(paramsForCustomFn);
                if (!Array.isArray(constrainedParams) || constrainedParams.length !== paramStructure.length) { throw new Error("Constraint function did not return a valid parameter structure."); }
                const paramsAfterCustom = JSON.stringify(constrainedParams);
                if (!boxConstraintApplied && paramsAfterCustom !== paramsBeforeCustom) { changedActive = true; logFn("Custom constraint function modified parameters.", 'debug'); }
                else if (boxConstraintApplied && paramsAfterCustom !== JSON.stringify(paramsForCustomFn)) { changedActive = true; logFn("Custom constraint function modified parameters after box constraints.", 'debug'); }
            } catch (e) { logFn(`Error executing custom constraint function: ${e.message}`, 'error'); constrainedParams = boxConstraintApplied ? JSON.parse(JSON.stringify(reconstructedParams)) : reconstructedParams; }
        }
        return { constrainedParams, changedActive };
    }

    function calculateFinalResiduals(data, modelFunction, finalReconstructedParams, logFn) { /* ... existing code ... */
        if (!data || !data.x || !data.y || !data.ye || !modelFunction || !finalReconstructedParams) { logFn("Missing data/models/params for final residual calculation.", 'error'); return null; }
        const residualsPerSeries = []; let errorOccurred = false;
        data.x.forEach((xDataset, dsIdx) => {
            const yDataset = data.y[dsIdx]; const yeDataset = data.ye[dsIdx]; const models = modelFunction[dsIdx]; const paramsForDs = finalReconstructedParams[dsIdx]; const currentResiduals = [];
            if (!yDataset || !yeDataset || !models || !paramsForDs || xDataset.length !== yDataset.length || xDataset.length !== yeDataset.length) { logFn(`Inconsistent data/model/params for dataset ${dsIdx} in final residual calculation. Skipping.`, 'warn'); residualsPerSeries.push([]); return; }
            for (let ptIdx = 0; ptIdx < xDataset.length; ptIdx++) {
                const xPoint = xDataset[ptIdx]; const yPoint = yDataset[ptIdx]; const yePoint = yeDataset[ptIdx];
                if (yePoint === 0 || !isFinite(yPoint) || !isFinite(yePoint)) { logFn(`Invalid data/error at point ${ptIdx} in dataset ${dsIdx} for final residuals. Storing NaN.`, 'warn'); currentResiduals.push(NaN); continue; }
                let combinedModelValue = 0; let modelEvalError = false;
                try { models.forEach((modelFunc, paramIdx) => { if (!paramsForDs[paramIdx]) throw new Error(`Missing params for model ${paramIdx}.`); const componentParams = paramsForDs[paramIdx]; const componentModelResult = modelFunc(componentParams, [xPoint]); if (!componentModelResult || componentModelResult.length !== 1 || !isFinite(componentModelResult[0])) throw new Error(`Model ${dsIdx}-${paramIdx} returned invalid result.`); combinedModelValue += componentModelResult[0]; }); }
                catch (error) { logFn(`Error evaluating model for final residual at ds ${dsIdx}, pt ${ptIdx}: ${error.message}`, 'error'); modelEvalError = true; errorOccurred = true; }
                if (modelEvalError || !isFinite(combinedModelValue)) { currentResiduals.push(NaN); } else { currentResiduals.push((yPoint - combinedModelValue) / yePoint); }
            } residualsPerSeries.push(currentResiduals);
        });
        return errorOccurred ? null : residualsPerSeries;
    }

    // --- NEW: Helper to determine X range ---
    /**
     * Determines the appropriate x-range for curve calculation.
     * @param {number[]} seriesXData - The original x-data for the dataset.
     * @param {Array<number> | null | undefined} modelXRangeSeries - The user-provided range [min, max] for this series.
     * @param {number} seriesIndex - The index of the dataset.
     * @param {Function} logFn - Logging function.
     * @returns {{min: number, max: number} | null} - The determined range or null if invalid.
     * @private
     */
    function _determineXRange(seriesXData, modelXRangeSeries, seriesIndex, logFn) {
        let xMin = null, xMax = null;

        // 1. Try user-provided range
        if (modelXRangeSeries && Array.isArray(modelXRangeSeries) && modelXRangeSeries.length === 2) {
            const minR = Number(modelXRangeSeries[0]);
            const maxR = Number(modelXRangeSeries[1]);
            if (isFinite(minR) && isFinite(maxR) && minR <= maxR) {
                xMin = minR;
                xMax = maxR;
                logFn(`Using provided x-range [${xMin}, ${xMax}] for series ${seriesIndex} curve.`, 'debug');
            } else {
                logFn(`Provided model_x_range for series ${seriesIndex} is invalid [${modelXRangeSeries[0]}, ${modelXRangeSeries[1]}]. Falling back to data range.`, 'warn');
            }
        } else if (modelXRangeSeries !== null && modelXRangeSeries !== undefined) {
             logFn(`Provided model_x_range for series ${seriesIndex} is not an array of two numbers. Falling back to data range.`, 'warn');
        }

        // 2. Fallback to data range
        if (xMin === null || xMax === null) {
            logFn(`Determining x-range from data for series ${seriesIndex} curve.`, 'debug');
            if (seriesXData && seriesXData.length > 0) {
                try {
                    // Ensure data is numeric before min/max
                    const numericX = seriesXData.filter(x => isFinite(x));
                    if (numericX.length > 0) {
                        xMin = Math.min(...numericX);
                        xMax = Math.max(...numericX);
                    } else {
                         logFn(`No finite x-data found for series ${seriesIndex}. Cannot determine data range.`, 'error');
                         return null;
                    }
                } catch (e) {
                     logFn(`Error determining data range for series ${seriesIndex}: ${e.message}`, 'error');
                     return null;
                }
            } else {
                xMin = 0.0; xMax = 0.0; // Default for empty data
            }
        }

        // Final check
        if (xMin === null || xMax === null || !isFinite(xMin) || !isFinite(xMax)) {
             logFn(`Could not determine valid x-range for series ${seriesIndex}.`, 'error');
             return null;
        }

        return { min: xMin, max: xMax };
    }
    
    /**
     * Calculates the fitted model curve AND optional confidence interval bands using Hessian/Covariance.
     * Includes enhanced checks for inputs and calculation results.
     * @returns {{curves: Array<{x: number[], y: number[], ci_lower?: number[], ci_upper?: number[]}> | null, negativeVarianceEncountered: boolean}}
     * @private
     */
    function calculateFittedModelCurves(
        data, modelFunction, finalReconstructedParams, numPoints, logFn,
        confidenceLevel, finalActiveParams, covarianceMatrix, activeParamInfo, reconstructParams, epsilon, degreesOfFreedom, K,
        model_x_range_list // <<< ADDED: List of optional ranges
    ) {

        // --- Input Validation ---
        if (!data || !data.x || !modelFunction || !finalReconstructedParams || !Array.isArray(finalReconstructedParams)) {
            logFn("Missing or invalid primary arguments for fitted curve calculation.", 'error');
            return { curves: null, negativeVarianceEncountered: false };
        }
        if (numPoints <= 1) {
            logFn(`Invalid numPoints (${numPoints}) for fitted curve calculation.`, 'error');
            return { curves: null, negativeVarianceEncountered: false };
        }
        if (confidenceLevel !== null && (typeof confidenceLevel !== 'number' || confidenceLevel <= 0 || confidenceLevel >= 1)) {
            logFn(`Invalid confidenceLevel (${confidenceLevel}) passed internally.`, 'warn');
            confidenceLevel = null;
        }

        let errorOccurred = false;
        let negativeVarianceEncountered = false;
        const fittedCurves = [];
        const K_active = finalActiveParams?.length ?? 0;

        const canCalculateCI = confidenceLevel !== null &&
                               finalActiveParams && K_active > 0 &&
                               covarianceMatrix && Array.isArray(covarianceMatrix) &&
                               covarianceMatrix.length === K_active &&
                               covarianceMatrix[0]?.length === K_active &&
                               degreesOfFreedom > 0 && isFinite(degreesOfFreedom) &&
                               activeParamInfo && Array.isArray(activeParamInfo) && activeParamInfo.length === K_active &&
                               typeof reconstructParams === 'function' &&
                               isFinite(epsilon) && epsilon > 0;

        if (confidenceLevel !== null && !canCalculateCI) {
            logFn("Standard CI calculation prerequisites not met. Skipping standard CI calculation.", 'warn');
        }

        try {
            data.x.forEach((xDataset, dsIdx) => {
                if (!xDataset || !Array.isArray(xDataset) || !modelFunction[dsIdx] || !finalReconstructedParams[dsIdx]) {
                    logFn(`Skipping dataset ${dsIdx}: Missing data, model, or params.`, 'warn');
                    fittedCurves.push(null);
                    return;
                }

                const models = modelFunction[dsIdx];
                const paramsForDs = finalReconstructedParams[dsIdx];
                const currentCurve = { x: [], y: [], ci_lower: [], ci_upper: [] };
                let calculateCIForThisDs = canCalculateCI;
                let tCrit = NaN;

                if (calculateCIForThisDs) {
                    const alpha = 1.0 - confidenceLevel;
                    const probT = 1.0 - alpha / 2.0;
                    tCrit = calculateTValue(probT, degreesOfFreedom, logFn);
                    if (isNaN(tCrit)) {
                        logFn(`Failed to calculate critical t-value for dataset ${dsIdx}. Skipping CI calculation.`);
                        calculateCIForThisDs = false;
                    }
                }

                // --- Determine X Range using helper ---
                const currentSeriesRangeOpt = model_x_range_list?.[dsIdx] ?? null;
                const xRange = _determineXRange(xDataset, currentSeriesRangeOpt, dsIdx, logFn);
                if (xRange === null) {
                    logFn(`Invalid X range for dataset ${dsIdx}. Skipping curve calculation.`, 'warn');
                    fittedCurves.push(null); 
                    return;
                }
                const { min: xMin, max: xMax } = xRange;
                // --- End Determine X Range ---

                const dx = (xMax === xMin || numPoints <= 1) ? 0 : (xMax - xMin) / (numPoints - 1);

                for (let i = 0; i < numPoints; i++) {
                    const xCalc = (dx === 0) ? xMin : xMin + i * dx;
                    let yCalc = 0;
                    let modelEvalError = false;
                    let ciLower = NaN;
                    let ciUpper = NaN;

                    try {
                        models.forEach((modelFunc, paramIdx) => {
                            if (!paramsForDs[paramIdx]) throw new Error(`Missing params for model ${paramIdx}.`);
                            const componentParams = paramsForDs[paramIdx];
                            const componentModelResult = modelFunc(componentParams, [xCalc]);
                            if (!componentModelResult || componentModelResult.length !== 1 || !isFinite(componentModelResult[0])) throw new Error(`Model ${dsIdx}-${paramIdx} returned invalid result.`);
                            yCalc += componentModelResult[0];
                        });
                        if (!isFinite(yCalc)) throw new Error("Calculated y value is not finite.");
                    } catch (error) {
                        logFn(`Error evaluating model for fitted curve at ds ${dsIdx}, x=${xCalc.toFixed(3)}: ${error.message}`, 'error');
                        modelEvalError = true; errorOccurred = true; yCalc = NaN;
                    }

                    if (calculateCIForThisDs && isFinite(yCalc)) {
                        try {
                            const jModelRow = new Array(K_active).fill(0);
                            for (let actIdx = 0; actIdx < K_active; actIdx++) {
                                const perturbedActiveParams = [...finalActiveParams];
                                const originalValue = perturbedActiveParams[actIdx];
                                let h = epsilon * Math.abs(originalValue) + epsilon; if (h === 0) h = epsilon;
                                perturbedActiveParams[actIdx] += h;
                                const perturbedReconstructed = reconstructParams(perturbedActiveParams);
                                const perturbedParamsForDs = perturbedReconstructed[dsIdx];
                                let perturbedCombinedModelValue = 0;
                                let perturbationFailed = false;

                                try {
                                    models.forEach((modelFunc, paramIdx) => {
                                        if (perturbationFailed) return;
                                        if (!perturbedParamsForDs || !perturbedParamsForDs[paramIdx]) throw new Error(`Missing perturbed params ds ${dsIdx}, pIdx ${paramIdx}.`);
                                        const componentParams = perturbedParamsForDs[paramIdx];
                                        const componentModelResult = modelFunc(componentParams, [xCalc]);
                                        if (!componentModelResult || componentModelResult.length !== 1 || !isFinite(componentModelResult[0])) {
                                            perturbationFailed = true; return;
                                        }
                                        perturbedCombinedModelValue += componentModelResult[0];
                                    });
                                    if (perturbationFailed || !isFinite(perturbedCombinedModelValue)) throw new Error("Perturbed model evaluation failed or yielded non-finite result.");
                                } catch (pertError) {
                                    logFn(`Error evaluating perturbed model (ds ${dsIdx}, x=${xCalc.toFixed(3)}, actP ${actIdx}): ${pertError.message}. Derivative set to 0.`, 'warn');
                                    perturbationFailed = true;
                                }

                                jModelRow[actIdx] = perturbationFailed ? 0 : (perturbedCombinedModelValue - yCalc) / h;
                                if (!isFinite(jModelRow[actIdx])) {
                                    logFn(`Non-finite model Jacobian element (ds ${dsIdx}, x=${xCalc.toFixed(3)}, actP ${actIdx}). Setting to 0.`, 'warn');
                                    jModelRow[actIdx] = 0;
                                }
                            }

                            let varianceY = 0;
                            for (let r = 0; r < K_active; r++) {
                                for (let c = 0; c < K_active; c++) {
                                    if (!covarianceMatrix[r] || !isFinite(covarianceMatrix[r][c])) {
                                        throw new Error(`Invalid covariance matrix element Cov[${r}][${c}]: ${covarianceMatrix[r]?.[c]}`);
                                    }
                                    if (!isFinite(jModelRow[r]) || !isFinite(jModelRow[c])) {
                                        throw new Error(`Non-finite Jacobian element encountered during variance calculation.`);
                                    }
                                    varianceY += jModelRow[r] * covarianceMatrix[r][c] * jModelRow[c];
                                }
                            }

                            if (!isFinite(varianceY)) throw new Error(`Calculated variance is not finite (${varianceY}).`);
                            if (varianceY < 0) {
                                logFn(`Warning: Calculated variance for CI at x=${xCalc.toFixed(3)} is negative (${varianceY.toExponential(3)}). Using absolute value for SE.`);
                                negativeVarianceEncountered = true;
                                varianceY = Math.abs(varianceY);
                            }

                            const seY = Math.sqrt(varianceY);
                            if (!isFinite(seY)) throw new Error(`Calculated standard error is not finite.`);
                            const ciHalfWidth = tCrit * seY;
                            ciLower = yCalc - ciHalfWidth;
                            ciUpper = yCalc + ciHalfWidth;
                            if (!isFinite(ciLower) || !isFinite(ciUpper)) {
                                logFn(`Non-finite CI bound calculated at x=${xCalc.toFixed(3)}. Setting to NaN.`);
                                ciLower = NaN; ciUpper = NaN;
                            }

                        } catch (ciError) {
                            logFn(`Error calculating standard confidence interval at ds ${dsIdx}, x=${xCalc.toFixed(3)}: ${ciError.message}`, 'error');
                            ciLower = NaN; ciUpper = NaN;
                        }
                    }

                    currentCurve.x.push(xCalc);
                    currentCurve.y.push(yCalc);
                    if (calculateCIForThisDs) {
                        currentCurve.ci_lower.push(ciLower);
                        currentCurve.ci_upper.push(ciUpper);
                    }
                }

                if (calculateCIForThisDs && currentCurve.ci_lower.length > 0) {
                    fittedCurves.push(currentCurve);
                } else {
                    fittedCurves.push({ x: currentCurve.x, y: currentCurve.y });
                }
            });

        } catch (outerError) {
            logFn(`Error processing datasets for curve calculation: ${outerError.message}`, 'error');
            return { curves: null, negativeVarianceEncountered: negativeVarianceEncountered };
        }

        const finalCurves = fittedCurves.filter(c => c !== null);
        return { curves: finalCurves.length > 0 ? finalCurves : null, negativeVarianceEncountered: negativeVarianceEncountered };
    }

/**
     * Calculates confidence interval bands using bootstrapping.
     * Includes enhanced error handling.
     * @private
     */
async function calculateBootstrapCIBands(
    originalData, modelFunction, finalReconstructedParams, activeParamInfo, reconstructParams, paramStructure,
    numBootstrapSamples, confidenceLevel, numPointsForCurve, logFn, baseOptions,
    model_x_range_list // <<< ADDED
) {
    logFn(`--- Starting Bootstrap CI Calculation (${numBootstrapSamples} samples) ---`, 'info');

    // --- Input Validation ---
    if (!originalData || !originalData.x || !modelFunction || !finalReconstructedParams) { logFn("Bootstrap Error: Missing primary input data/models/params.", 'error'); return null; }
    if (!numBootstrapSamples || numBootstrapSamples < 2) { logFn(`Bootstrap Warning: Invalid numBootstrapSamples (${numBootstrapSamples}). Must be >= 2. Skipping bootstrap.`, 'warn'); return null; }
    if (!confidenceLevel || confidenceLevel <= 0 || confidenceLevel >= 1) { logFn(`Bootstrap Warning: Invalid confidenceLevel (${confidenceLevel}). Skipping bootstrap.`, 'warn'); return null; }
    if (numPointsForCurve <= 1) { logFn(`Bootstrap Warning: Invalid numPointsForCurve (${numPointsForCurve}). Skipping bootstrap.`, 'warn'); return null; }

    const numDatasets = originalData.x.length;
    if (numDatasets === 0) { logFn("Bootstrap Info: No datasets to process.", 'info'); return []; }

    // --- Get Original Curve Shape & Initialize Storage ---
    const collectedYValues = []; // Structure: collectedYValues[dsIdx][xPointIdx] = [y_boot1, y_boot2, ...]
    const originalFittedCurves = []; // Stores {x: [...], y: [...]} from original fit
    try {
         // Use a dummy logger for this internal call to avoid excessive logging
         const dummyLogFn = () => {};
         const originalCurveResult = calculateFittedModelCurves(
             originalData, modelFunction, finalReconstructedParams, numPointsForCurve, dummyLogFn,
             null, [], null, [], reconstructParams, 0, 0, 0, // Dummy values
             model_x_range_list // Pass the range list
         );
         if (!originalCurveResult || !originalCurveResult.curves) { throw new Error(`Failed to calculate original curve shape (result: ${originalCurveResult})`); }
         if (originalCurveResult.curves.length !== numDatasets) { throw new Error(`Original curve calculation returned ${originalCurveResult.curves.length} curves, but expected ${numDatasets}.`); }

         originalCurveResult.curves.forEach((curve, dsIdx) => {
             if (curve && curve.x && curve.y && curve.x.length === numPointsForCurve) {
                 originalFittedCurves[dsIdx] = { x: curve.x, y: curve.y };
                 collectedYValues[dsIdx] = curve.x.map(() => []); // Initialize arrays for each x point
             } else {
                  logFn(`Bootstrap Info: Original curve for dataset ${dsIdx} is invalid or has wrong length. Skipping this dataset.`, 'warn');
                  originalFittedCurves[dsIdx] = null;
                  collectedYValues[dsIdx] = null;
             }
         });
         if (originalFittedCurves.every(c => c === null)) { throw new Error("No valid original curves found to base bootstrap on."); }
    } catch(err) {
         logFn(`Bootstrap Error getting original curve shape: ${err.message}`, 'error');
         console.error("Original Curve Shape Error:", err);
         return null; // Cannot proceed without original shape
    }

    // --- Bootstrap Loop ---
    let successfulSamples = 0;
    let i_loop = 0; // Use a different variable name for the loop counter
    // Define bootstrapOptions within the scope where 'i_loop' is accessible
    const bootstrapOptions = {
        ...baseOptions, // Inherit tolerances, link/fix maps etc. from the main fit call
        confidenceInterval: null, // Explicitly disable CI calc within bootstrap
        calculateFittedModel: { numPoints: numPointsForCurve }, // Ensure curves are calculated
        logLevel: 'error', // Reduce verbosity during bootstrap fits
        onLog: (msg, level) => {
            // Only log errors/warnings from inner fits, referencing the sample number
            if (level === 'error' || 'warn') {
                // Use i_loop here
                logFn(`Bootstrap Fit [Sample ${i_loop + 1}] ${level.toUpperCase()}: ${msg}`, 'warn');
            }
            // Add more levels here if needed for debugging bootstrapOptions.logLevel
        },
        onProgress: () => {}, // Disable progress reporting for inner fits
    };


    for (i_loop = 0; i_loop < numBootstrapSamples; i_loop++) {
        if (i_loop > 0 && i_loop % 50 === 0) { logFn(`Bootstrap progress: ${i_loop} / ${numBootstrapSamples} samples completed...`, 'info'); }

        // 1. Create Bootstrap Dataset
        const bootstrapData = { x: [], y: [], ye: [] };
        let totalPointsInSample = 0;
        try {
            for (let dsIdx = 0; dsIdx < numDatasets; dsIdx++) {
                // Skip if original dataset was invalid/empty
                if (!originalData.x[dsIdx] || originalData.x[dsIdx].length === 0) {
                    bootstrapData.x[dsIdx] = []; bootstrapData.y[dsIdx] = []; bootstrapData.ye[dsIdx] = [];
                    continue;
                }
                const N_points = originalData.x[dsIdx].length;
                const indices = Array.from({ length: N_points }, () => Math.floor(Math.random() * N_points)); // Sample indices

                // Check original data consistency before mapping
                if (!originalData.y[dsIdx] || originalData.y[dsIdx].length !== N_points || !originalData.ye[dsIdx] || originalData.ye[dsIdx].length !== N_points) {
                     throw new Error(`Inconsistent original data lengths for dataset ${dsIdx}.`);
                }

                // Create resampled dataset
                bootstrapData.x[dsIdx] = indices.map(idx => originalData.x[dsIdx][idx]);
                bootstrapData.y[dsIdx] = indices.map(idx => originalData.y[dsIdx][idx]);
                bootstrapData.ye[dsIdx] = indices.map(idx => Math.max(1e-9, originalData.ye[dsIdx][idx])); // Ensure positive errors
                totalPointsInSample += N_points;
            }
             if (totalPointsInSample === 0 && numDatasets > 0) {
                 throw new Error("Bootstrap sample generation resulted in zero total points (all original datasets might be empty).");
             }
        } catch (resampleErr) {
             logFn(`Bootstrap Warning: Error creating bootstrap sample ${i_loop + 1}: ${resampleErr.message}. Skipping sample.`, 'warn');
             continue; // Skip to next sample
        }

        // 2. Refit using bootstrap data
        let bootResult = null;
        try {
            // Check modelFunction structure matches bootstrapData structure
             if (modelFunction.length !== bootstrapData.x.length) {
                 throw new Error(`Internal error: Model function array length (${modelFunction.length}) does not match bootstrap data length (${bootstrapData.x.length}).`);
             }

            // Use finalReconstructedParams from original fit as initial guess for speed
            bootResult = await lmFitGlobal(bootstrapData, modelFunction, finalReconstructedParams, {
                ...baseOptions,
                calculateFittedModel: { numPoints: numPointsForCurve },
                model_x_range: model_x_range_list, // <<< Ensure model_x_range is passed here
                confidenceInterval: null,
                logLevel: 'error',
            });

            // 3. Collect results if successful
            if (bootResult && bootResult.converged && !bootResult.error && bootResult.fittedModelCurves) {
                successfulSamples++;
                bootResult.fittedModelCurves.forEach((curve, dsIdx) => {
                    // Check if this dataset is valid and structure matches
                    if (collectedYValues[dsIdx] && curve && curve.y && curve.y.length === collectedYValues[dsIdx].length) {
                        curve.y.forEach((yVal, xIdx) => {
                            if (isFinite(yVal)) {
                                // Ensure the target array exists (should have been initialized)
                                if (collectedYValues[dsIdx][xIdx]) {
                                     collectedYValues[dsIdx][xIdx].push(yVal);
                                } else {
                                     // This case should ideally not happen if initialization was correct
                                     logFn(`Bootstrap Warning: Target array for collected values missing at ds ${dsIdx}, xIdx ${xIdx}.`, 'warn');
                                }
                            }
                            // Optionally log NaN yVal from bootstrap fit curve?
                            // else { logFn(`Bootstrap Debug: NaN yVal encountered at ds ${dsIdx}, xIdx ${xIdx} in sample ${i_loop + 1}.`, 'debug'); }
                        });
                    } else if (collectedYValues[dsIdx]) {
                         // Log if curve structure from bootstrap fit doesn't match expected
                         logFn(`Bootstrap Debug: Curve data mismatch or invalid for dataset ${dsIdx} in sample ${i_loop + 1}. Expected length ${collectedYValues[dsIdx].length}, Got ${curve?.y?.length}.`, 'debug');
                    }
                });
            } else {
                 // Log failure reason more clearly
                 let failureReason = "Unknown";
                 if (!bootResult) failureReason = "lmFitGlobal returned null/undefined";
                 else if (bootResult.error) failureReason = `Error property set: ${bootResult.error}`;
                 else if (!bootResult.converged) failureReason = "Did not converge";
                 else if (!bootResult.fittedModelCurves) failureReason = "Converged but missing fittedModelCurves";
                 logFn(`Bootstrap sample ${i_loop + 1} failed. Reason: ${failureReason}.`, 'debug'); // Keep as debug unless it happens a lot
            }
        } catch (fitErr) {
             // Catch errors from lmFitGlobal itself during bootstrap fit
             logFn(`Bootstrap Warning: Error during fit for sample ${i_loop + 1}: ${fitErr.message}. Skipping sample.`, 'warn');
             console.error(`Bootstrap Fit ${i_loop + 1} Exception:`, fitErr); // Log stack trace
        }
    } // End bootstrap loop

    // --- Post-Loop Checks and Logging ---
    logFn(`--- Bootstrap CI Calculation Finished ---`, 'info');
    logFn(`Successful bootstrap samples: ${successfulSamples} / ${numBootstrapSamples}`, 'info');
    if (successfulSamples < Math.min(10, numBootstrapSamples * 0.1)) { // Check if very few succeeded
         logFn(`Bootstrap Error: Very few successful samples (${successfulSamples}). Cannot reliably calculate CI bands.`, 'error');
         // Return original curves without bands as fallback
         return originalFittedCurves.map(curve => curve ? { x: curve.x, y: curve.y } : null).filter(c => c !== null);
    } else if (successfulSamples < numBootstrapSamples * 0.5) {
         logFn(`Bootstrap Warning: Less than 50% of bootstrap samples were successful. Results may be less reliable.`, 'warn');
    }

    // --- Calculate Percentiles ---
    const bootstrapCurvesResult = []; // Array to hold the final curve objects with CIs
    const alpha = 1.0 - confidenceLevel;
    const lowerP = alpha / 2.0;
    const upperP = 1.0 - lowerP;
    let percentileCalculationOk = true; // Flag for success

    try {
        logFn(`Bootstrap Debug: Starting percentile calculation. NumDatasets: ${numDatasets}`, 'debug');
        for (let dsIdx = 0; dsIdx < numDatasets; dsIdx++) {
             // Skip dataset if original curve was invalid or no bootstrap data collected
             if (!originalFittedCurves[dsIdx] || !collectedYValues[dsIdx]) {
                 logFn(`Bootstrap Debug: Skipping percentile calc for ds ${dsIdx} (no original curve or collected values).`, 'debug');
                 // Don't push null here, just skip
                 continue;
             }
             const curve = originalFittedCurves[dsIdx]; // Use original x and central y
             const ci_lower = [];
             const ci_upper = [];
             logFn(`Bootstrap Debug: Calculating percentiles for ds ${dsIdx}, ${curve.x.length} points.`, 'debug');

             for (let xIdx = 0; xIdx < curve.x.length; xIdx++) {
                 // Check if collectedYValues[dsIdx] exists and has the index xIdx
                 if (!collectedYValues[dsIdx] || !collectedYValues[dsIdx][xIdx]) {
                      logFn(`Bootstrap Warning: Missing collected y-values array at ds ${dsIdx}, xIdx ${xIdx}. Setting bands to NaN.`, 'warn');
                      ci_lower.push(NaN);
                      ci_upper.push(NaN);
                      continue; // Skip to next x point
                 }

                 const yValues = collectedYValues[dsIdx][xIdx];
                 // Check if we have enough *successful* samples for this point
                 if (yValues && yValues.length >= 2) { // Use >= 2 for basic percentile
                     // Filter out potential non-finite values just in case
                     const finiteYValues = yValues.filter(y => isFinite(y));
                     if (finiteYValues.length < 2) {
                          logFn(`Bootstrap Warning: Not enough finite data points (${finiteYValues.length}) after filtering at ds ${dsIdx}, xIdx ${xIdx}. Setting bands to NaN.`, 'warn');
                          ci_lower.push(NaN);
                          ci_upper.push(NaN);
                          continue; // Skip to next x point
                     }

                     finiteYValues.sort((a, b) => a - b);
                     // Calculate indices carefully
                     let lowIdx = Math.floor(lowerP * finiteYValues.length);
                     let highIdx = Math.ceil(upperP * finiteYValues.length) - 1; // -1 for 0-based index

                     // Boundary checks (more robust)
                     lowIdx = Math.max(0, lowIdx);
                     highIdx = Math.min(finiteYValues.length - 1, highIdx);
                     if (highIdx < lowIdx) highIdx = lowIdx; // Ensure high >= low

                     // Final check before accessing array elements
                     if (lowIdx < finiteYValues.length && highIdx < finiteYValues.length) {
                        ci_lower.push(finiteYValues[lowIdx]);
                        ci_upper.push(finiteYValues[highIdx]);
                     } else {
                         // This case should be rare given the checks above, but handles edge cases
                         logFn(`Bootstrap Warning: Invalid index calculated for percentile at ds ${dsIdx}, xIdx ${xIdx} (low:${lowIdx}, high:${highIdx}, len:${finiteYValues.length}). Setting bands to NaN.`, 'warn');
                         ci_lower.push(NaN);
                         ci_upper.push(NaN);
                     }
                 } else {
                     logFn(`Bootstrap Warning: Not enough successful samples (${yValues?.length}) to calculate percentile at ds ${dsIdx}, xIdx ${xIdx}. Setting bands to NaN.`, 'warn');
                     ci_lower.push(NaN);
                     ci_upper.push(NaN);
                 }
             } // End xIdx loop

             // Push the result for this dataset (original x/y + bootstrap CIs)
             bootstrapCurvesResult.push({ x: curve.x, y: curve.y, ci_lower: ci_lower, ci_upper: ci_upper });
             logFn(`Bootstrap Debug: Finished percentiles for ds ${dsIdx}.`, 'debug');
        } // End dsIdx loop
    } catch (percentileError) {
         logFn(`Bootstrap Error during percentile calculation: ${percentileError.message}`, 'error');
         console.error("Percentile Calculation Error:", percentileError);
         percentileCalculationOk = false; // Mark as failed
    }

    // --- Return results ---
    if (percentileCalculationOk) {
        // Return the array of curve objects we built
        logFn(`Bootstrap Debug: Returning ${bootstrapCurvesResult.length} calculated curves.`, 'debug');
        return bootstrapCurvesResult;
    } else {
        // Return null if percentile calculation itself failed
        logFn(`Bootstrap Debug: Returning null due to percentile calculation error.`, 'debug');
        return null;
    }
} // End of calculateBootstrapCIBands

    /**
     * Performs global curve fitting for multiple datasets using the Levenberg-Marquardt algorithm.
     * Allows linking and fixing parameters across datasets and uses composite models.
     * Includes calculation of goodness-of-fit statistics and optional confidence intervals for fitted curves.
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
     *   @param {(number|string|null)[][][]} [options.linkMap=null] - NEW: Nested array matching initialParameters. Non-null values indicate link group ID.
     *   @param {boolean[][][]} [options.fixMap=null] - Nested array matching initialParameters. True indicates fixed.
     *   @param {object[][][]} [options.constraints=null] - Box constraints {min, max}, nested array matching initialParameters.
     *   @param {Function | null} [options.constraintFunction=null] - Custom function: `(params) => modifiedParams`. Applied after box constraints.
     *   @param {string} [options.logLevel='info'] - Control logging verbosity ('none', 'error', 'warn', 'info', 'debug').
     *   @param {Function} [options.onLog=()=>{}] - Callback for logs: `(message, level) => {}`.
     *   @param {Function} [options.onProgress=()=>{}] - Callback for progress: `(progressData) => {}`, where `progressData = { iteration, chiSquared, lambda, activeParameters }`.
     *   @param {number|null} [options.robustCostFunction=null] - null, 1 (L1), or 2 (Lorentzian).
     *   @param {number} [options.lambdaInitial=1e-3]
     *   @param {number} [options.lambdaIncreaseFactor=10]
     *   @param {number} [options.lambdaDecreaseFactor=10]
     *   @param {number} [options.epsilon=1e-8] - Step size for numerical differentiation.
     *   @param {boolean | {numPoints: number}} [options.calculateFittedModel=false] - If true or object, calculate fitted curve.
     *   @param {number} [options.covarianceLambda=1e-9] - Regularization factor added to Hessian diagonal for covariance calculation.
     *   @param {number | null} [options.confidenceInterval=null] - Confidence level for CI bands on fitted curves (e.g., 0.95 for 95%). If null or invalid, CIs are not calculated.
     *   @param {boolean} [options.bootstrapFallback=true] - If true, attempt bootstrapping if standard CI calculation encounters negative variance.
     *   @param {number} [options.numBootstrapSamples=200] - Number of samples for bootstrap fallback.
     * @returns {object} - Fitting results including statistics.
     *   @property {number[]} p_active - Final values of active parameters.
     *   @property {number[][][]} p_reconstructed - Final values of all parameters (including fixed/linked) in original nested structure.
     *   @property {(number|null)[][][]} finalParamErrors - Estimated standard errors for all parameters (NaN if calculation failed, 0 for fixed).
     *   @property {number} chiSquared - Final chi-squared value.
     *   @property {number[][] | null} covarianceMatrix - Calculated covariance matrix for active parameters (potentially regularized). Null if K=0 or inversion failed.
     *   @property {number[]} parameterErrors - Estimated standard errors for active parameters (NaN if calculation failed).
     *   @property {number} iterations - Number of iterations performed.
     *   @property {boolean} converged - True if the fit converged based on tolerances.
     *   @property {string[]} activeParamLabels - Labels for the active parameters.
     *   @property {string | null} error - Error message if fit failed catastrophically.
     *   @property {number} totalPoints - Total number of data points used (N).
     *   @property {number} degreesOfFreedom - Degrees of freedom (N-K).
     *   @property {number} reducedChiSquared - Chi-squared divided by degrees of freedom.
     *   @property {number} aic - Akaike Information Criterion.
     *   @property {number} aicc - Corrected Akaike Information Criterion.
     *   @property {number} bic - Bayesian Information Criterion.
     *   @property {number[][] | null} residualsPerSeries - Weighted residuals ((y-ymodel)/ye) for each dataset.
     *   @property {Array<{x: number[], y: number[], ci_lower?: number[], ci_upper?: number[]}> | null} fittedModelCurves - Calculated fitted model curves for each dataset, potentially including confidence interval bands (`ci_lower`, `ci_upper`) if `options.confidenceInterval` was valid.
     *   @property {boolean} bootstrapUsed - True if bootstrap fallback was used for CI bands.
     *   @property {Array<Array<{x: number[], y: number[]}>> | null} fittedModelComponentCurves - Nested array of individual component model curves for each dataset and model function. Each dataset contains an array of component curves, where each curve has `x` and `y` arrays representing the independent and dependent variable values, respectively. Null if component curve calculation was not requested or failed.
     */
    async function lmFitGlobal(data, modelFunction, initialParameters, options = {}) {
        const logFn = options.onLog && typeof options.onLog === 'function' ? options.onLog : () => {};
        const logLevelStr = options.logLevel ?? 'info';
        const logLevel = LOG_LEVELS[logLevelStr.toLowerCase()] ?? LOG_LEVELS.info;
        const log = (message, level) => { const messageLevel = LOG_LEVELS[level] ?? LOG_LEVELS.info; if (logLevel >= messageLevel) { logFn(message, level); } };

        // --- Options Processing ---
        const maxIterations = options.maxIterations ?? 100;
        const errorTolerance = options.errorTolerance ?? 1e-6;
        const gradientTolerance = options.gradientTolerance ?? 1e-6;
        const linkMapInput = options.linkMap ?? null;
        const fixMapInput = options.fixMap ?? null;
        const constraints = options.constraints ?? null;
        const constraintFunction = options.constraintFunction ?? null;
        const robustCostFunction = options.robustCostFunction ?? null;
        let lambda = options.lambdaInitial ?? 1e-3;
        const lambdaIncreaseFactor = options.lambdaIncreaseFactor ?? 10;
        const lambdaDecreaseFactor = options.lambdaDecreaseFactor ?? 10;
        const epsilon = options.epsilon ?? 1e-8;
        const calculateFittedOpt = options.calculateFittedModel ?? false;
        const numPointsForCurve = (typeof calculateFittedOpt === 'object' && calculateFittedOpt.numPoints > 1)
                                  ? calculateFittedOpt.numPoints
                                  : 300;
        const shouldCalculateFitted = calculateFittedOpt === true || (typeof calculateFittedOpt === 'object');
        const covarianceLambda = options.covarianceLambda ?? 1e-9;
        const bootstrapFallback = options.bootstrapFallback ?? true;
        const numBootstrapSamples = options.numBootstrapSamples ?? 200;

        // --- Confidence Interval Option ---
        let confidenceLevel = null; // Default: do not calculate
        if (options.confidenceInterval !== null && options.confidenceInterval !== undefined) {
            if (typeof options.confidenceInterval === 'number' && options.confidenceInterval > 0 && options.confidenceInterval < 1) {
                confidenceLevel = options.confidenceInterval;
            } else {
                 log(`Invalid confidenceInterval option (${options.confidenceInterval}). Must be a number between 0 and 1 (exclusive). Confidence intervals will not be calculated.`, 'warn');
            }
        }

        // --- Model X Range List Option ---
        const model_x_range_list = options.model_x_range ?? null;

        log("Starting lmFitGlobal (v1.2.6 - Added CI Bands)..."); // Update version marker

        // --- Calculate Total Data Points (N) ---
        let totalPoints = 0;
        if (data && data.x) { data.x.forEach(xDataset => { if (Array.isArray(xDataset)) totalPoints += xDataset.length; else log("Non-array in data.x.", 'warn'); }); }
        log(`Total data points (N): ${totalPoints}`);
        const baseErrorReturn = { p_active: [], p_reconstructed: initialParameters, finalParamErrors: null, chiSquared: NaN, covarianceMatrix: null, parameterErrors: null, iterations: 0, converged: false, activeParamLabels: [], totalPoints: totalPoints, degreesOfFreedom: NaN, reducedChiSquared: NaN, aic: NaN, aicc: NaN, bic: NaN, residualsPerSeries: null, fittedModelCurves: null };
        if (totalPoints === 0) { const errMsg = "No data points."; log(errMsg, 'error'); return { ...baseErrorReturn, error: errMsg }; }

        // --- Parameter Mapping Setup (K = totalActiveParams) ---
        let setupResults;
         try {
             setupResults = setupParameterMapping(initialParameters, linkMapInput, fixMapInput, log);
         }
         catch (error) { const errMsg = `Parameter setup failed: ${error.message}`; log(errMsg, 'error'); return { ...baseErrorReturn, error: errMsg }; }

        const { activeInitialParams, reconstructParams, activeParamInfo, totalActiveParams, paramStructure, activeParamLabels } = setupResults;
        const K = totalActiveParams;
        const { isFixed, masterMap, paramCoordinates, nTotalParams } = setupParameterMappingDetails(initialParameters, linkMapInput, fixMapInput);

        // --- Handle Case: No Active Parameters ---
        if (K === 0) {
            // ... (same as before, but initialize fittedCurves correctly) ...
             log("No active parameters to fit. Calculating initial stats.", 'warn');
            let initialChiSq = NaN; let dof = totalPoints; let redChiSq = NaN, aic = NaN, aicc = NaN, bic = NaN;
            let finalResiduals = null; let fittedCurves = null; let finalErrors = null;
            const initialReconstructed = reconstructParams([]);
            try {
                initialChiSq = calculateChiSquaredGlobal(data, modelFunction, reconstructParams, [], robustCostFunction, paramStructure, log);
                finalResiduals = calculateFinalResiduals(data, modelFunction, initialReconstructed, log);
                if (shouldCalculateFitted) {
                    // Call with nulls/zeros for CI-related params as they aren't applicable
                    fittedCurves = calculateFittedModelCurves(data, modelFunction, initialReconstructed, numPointsForCurve, log,
                                                            null, [], null, [], reconstructParams, epsilon, dof, 0, model_x_range_list); // K=0
                 }
                finalErrors = []; let currentFlatIdx_err = 0;
                paramStructure.forEach((dsStruct, dsIdx) => { finalErrors[dsIdx] = []; dsStruct.forEach((pLen, pIdx) => { finalErrors[dsIdx][pIdx] = []; for (let vIdx = 0; vIdx < pLen; vIdx++) { finalErrors[dsIdx][pIdx][vIdx] = 0; currentFlatIdx_err++; } }); });
                if (isFinite(initialChiSq) && dof > 0) { redChiSq = initialChiSq / dof; aic = initialChiSq; bic = initialChiSq; if (totalPoints > 1) aicc = aic; else aicc = Infinity; }
                 else if (isFinite(initialChiSq)) { redChiSq = Infinity; aic = initialChiSq; bic = initialChiSq; aicc = Infinity; log("Degrees of freedom is zero or negative. Reduced ChiSq and AICc are Infinity.", 'warn'); }
            } catch(e) { log(`Error calculating initial ChiSq/Residuals: ${e.message}`, 'error'); }
            return { p_active: [], p_reconstructed: initialReconstructed, finalParamErrors: finalErrors, chiSquared: initialChiSq, covarianceMatrix: [], parameterErrors: [], iterations: 0, converged: true, activeParamLabels: [], totalPoints: totalPoints, degreesOfFreedom: dof, reducedChiSquared: redChiSq, aic: aic, aicc: aicc, bic: bic, error: null, residualsPerSeries: finalResiduals, fittedModelCurves: fittedCurves };
        }

        // --- Initial Calculations ---
        let activeParameters = [...activeInitialParams];
        let chiSquared = NaN;
        try { chiSquared = calculateChiSquaredGlobal(data, modelFunction, reconstructParams, activeParameters, robustCostFunction, paramStructure, log); }
        catch (error) { const errMsg = `Initial Chi-Squared calculation failed: ${error.message}`; log(`Non-finite initial Chi-Squared.`, 'error'); return { ...baseErrorReturn, p_reconstructed: reconstructParams(activeParameters), chiSquared: chiSquared, error: errMsg }; }

        let converged = false; let covarianceMatrix = null; let parameterErrors = []; let iterationsPerformed = 0; let finalHessian = null;

        log(`Total active parameters (K): ${K}`, 'info'); log(`Active Parameter Labels: ${activeParamLabels.join(', ')}`, 'info'); log(`Initial Active Parameters: ${activeParameters.map(p=>p.toExponential(3)).join(', ')}`, 'info'); log(`Initial Chi-Squared: ${chiSquared}`, 'info');

        if (!isFinite(chiSquared)) { const errMsg = "Initial Chi-Squared is not finite."; log(`Non-finite initial Chi-Squared.`, 'error'); return { ...baseErrorReturn, p_reconstructed: reconstructParams(activeParameters), chiSquared: chiSquared, error: errMsg }; }

        // --- LM Iteration Loop ---
        let iteration;
        for (iteration = 0; iteration < maxIterations; iteration++) {
            iterationsPerformed = iteration + 1; log(`--- Iteration ${iterationsPerformed} (Lambda: ${lambda.toExponential(3)}) ---`, 'info'); log(`Iter ${iterationsPerformed} - Current Active Params: ${activeParameters.map(p=>p.toExponential(5)).join(', ')}`, 'debug');
            let jacobian, residuals; try { ({ jacobian, residuals } = calculateJacobianAndResidualsGlobal(data, modelFunction, reconstructParams, activeParameters, activeParamInfo, epsilon, paramStructure, log)); } catch (error) { const errMsg = `Jacobian failed: ${error.message}`; log(`Error Jacobian/Resid iter ${iterationsPerformed}: ${errMsg}`, 'error'); return { ...baseErrorReturn, p_reconstructed: reconstructParams(activeParameters), chiSquared: chiSquared, iterations: iterationsPerformed, error: errMsg }; }
            let gradient, currentHessian; try { gradient = calculateGradient(jacobian, residuals); currentHessian = calculateHessian(jacobian); } catch (error) { const errMsg = `Grad/Hess failed: ${error.message}`; log(`Error Grad/Hess iter ${iterationsPerformed}: ${errMsg}`, 'error'); return { ...baseErrorReturn, p_reconstructed: reconstructParams(activeParameters), chiSquared: chiSquared, iterations: iterationsPerformed, error: errMsg }; }
            if (gradient.some(g => !isFinite(g)) || currentHessian.some(row => row.some(h => !isFinite(h)))) { const errMsg = "Non-finite grad/hess."; log(`Non-finite grad/hess iter ${iterationsPerformed}.`, 'error'); return { ...baseErrorReturn, p_reconstructed: reconstructParams(activeParameters), chiSquared: chiSquared, iterations: iterationsPerformed, error: errMsg }; }
            const maxGradient = Math.max(...gradient.map(Math.abs)); if (maxGradient < gradientTolerance) { log(`Converged (grad tol ${gradientTolerance}). Max Grad: ${maxGradient.toExponential(3)}`, 'info'); converged = true; break; }
            log(`Gradient: ${gradient.map(g => g.toExponential(3)).join(', ')}`, 'info');
            let parameterUpdates; let solveSuccess = false; let attempt = 0; const maxSolveAttempts = 5; let currentLambda = lambda;
            while (!solveSuccess && attempt < maxSolveAttempts) {
                const dampedHessian = currentHessian.map((row, i) => row.map((value, j) => (i === j ? value + currentLambda : value)));
                try { const negativeGradient = gradient.map(g => -g); parameterUpdates = solveLinearSystem(dampedHessian, negativeGradient); if (parameterUpdates.some(up => !isFinite(up))) throw new Error("NaN/Inf in updates."); solveSuccess = true; }
                catch (error) { attempt++; log(`Solve failed (Att ${attempt}/${maxSolveAttempts}, Iter ${iterationsPerformed}): ${error.message}. Inc lambda.`, 'warn'); currentLambda = Math.min(currentLambda * lambdaIncreaseFactor * (attempt > 1 ? lambdaIncreaseFactor : 1) , 1e10); log(`Attempting solve with Lambda: ${currentLambda.toExponential(3)}`, 'info'); if (attempt >= maxSolveAttempts) { const errMsg = "Failed solve."; log(`Failed solve after ${maxSolveAttempts} attempts.`, 'error'); return { ...baseErrorReturn, p_reconstructed: reconstructParams(activeParameters), chiSquared: chiSquared, iterations: iterationsPerformed, error: errMsg }; } }
            }
            if (!solveSuccess) continue; lambda = currentLambda; log(`Parameter Updates: ${parameterUpdates.map(pu => pu.toExponential(3)).join(', ')}`, 'info');
            const proposedActiveParams = activeParameters.map((p, i) => p + parameterUpdates[i]);
            let proposedReconstructed = reconstructParams(proposedActiveParams);
            const { constrainedParams, changedActive } = applyConstraintsGlobal( proposedReconstructed, constraints, activeParamInfo, paramStructure, constraintFunction, log );
            proposedReconstructed = constrainedParams;
            const finalProposedActiveParams = changedActive ? activeParamInfo.map(info => proposedReconstructed[info.originalCoord[0][0]][info.originalCoord[0][1]][info.originalCoord[1]]) : proposedActiveParams;
            let newChiSquared = NaN; try { newChiSquared = calculateChiSquaredGlobal(data, modelFunction, reconstructParams, finalProposedActiveParams, robustCostFunction, paramStructure, log); } catch (error) { log(`Error ChiSq proposed step iter ${iterationsPerformed}: ${error.message}`, 'error'); newChiSquared = Infinity; }
            log(`New Chi-Squared: ${newChiSquared}`, 'info');
            if (isFinite(newChiSquared) && newChiSquared < chiSquared) {
                const chiSquaredChange = chiSquared - newChiSquared; activeParameters = finalProposedActiveParams; chiSquared = newChiSquared; lambda = Math.max(lambda / lambdaDecreaseFactor, 1e-12);
                log(`Accepted. ChiSq decreased by ${chiSquaredChange.toExponential(3)}. Lambda decreased to: ${lambda.toExponential(3)}`, 'info');
                try { options.onProgress?.({ iteration: iterationsPerformed, chiSquared: chiSquared, lambda: lambda, activeParameters: [...activeParameters] }); } catch (e) { log(`Error in onProgress callback: ${e.message}`, 'warn'); }
                if (chiSquaredChange < errorTolerance) { log(`Converged (chiSq tol ${errorTolerance}).`, 'info'); converged = true; break; }
            } else { lambda = Math.min(lambda * lambdaIncreaseFactor, 1e10); log(`Rejected (ChiSq ${isNaN(newChiSquared) ? 'NaN' : 'increased/stagnant'}). Lambda increased to: ${lambda.toExponential(3)}`, 'info'); if (lambda >= 1e10) log("Lambda reached maximum limit.", 'warn'); }
        } // --- End of LM Iteration Loop ---


        // --- Post-Loop Processing & Statistics ---
        if (!converged && iteration === maxIterations) { log(`lmFitGlobal did not converge within ${maxIterations} iterations.`, 'warn'); }
        log("Recalculating final Jacobian/Hessian for covariance...", 'info');
        try { const { jacobian: finalJacobian } = calculateJacobianAndResidualsGlobal(data, modelFunction, reconstructParams, activeParameters, activeParamInfo, epsilon, paramStructure, log); finalHessian = calculateHessian(finalJacobian); }
        catch (error) { log(`Failed to recalculate final Hessian: ${error.message}`, 'error'); finalHessian = null; }

        // --- Calculate Statistics & Parameter Errors ---
        let degreesOfFreedom = NaN, reducedChiSquared = NaN, aic = NaN, aicc = NaN, bic = NaN;
         if (isFinite(chiSquared) && totalPoints > 0) {
            degreesOfFreedom = totalPoints - K; // N - K
            // ... (rest of statistics calculation as before) ...
             if (degreesOfFreedom > 0) {
                reducedChiSquared = chiSquared / degreesOfFreedom; aic = chiSquared + 2 * K; const aiccCorrectionDenom = totalPoints - K - 1;
                if (aiccCorrectionDenom > 0) { aicc = aic + (2 * K * (K + 1)) / aiccCorrectionDenom; } else { aicc = Infinity; log("AICc denominator (N-K-1) is zero or negative. AICc set to Infinity.", 'warn'); }
                bic = chiSquared + K * Math.log(totalPoints);
            } else { reducedChiSquared = Infinity; aic = chiSquared + 2 * K; bic = chiSquared + K * Math.log(totalPoints); aicc = Infinity; log(`Degrees of freedom (${degreesOfFreedom}) is zero or negative. Reduced Chi-Squared is Infinity/undefined. Parameter errors may be unreliable or NaN.`, 'warn'); }
        } else { log("Final Chi-Squared is not finite. Cannot calculate statistics reliably.", 'warn'); degreesOfFreedom = totalPoints - K; reducedChiSquared = NaN; aic = NaN; aicc = NaN; bic = NaN; }

        // --- Covariance Matrix and Parameter Errors ---
        parameterErrors = new Array(K).fill(NaN);
        covarianceMatrix = null;
        if (finalHessian && K > 0) {
             try {
                 const regularizedHessian = finalHessian.map((row, i) => row.map((value, j) => (i === j ? value + covarianceLambda : value)));
                 log(`Applying regularization (lambda=${covarianceLambda}) for covariance matrix inversion.`, 'debug');
                 covarianceMatrix = invertMatrixUsingSVD(regularizedHessian); // Invert the regularized matrix

                 const scaleFactor = (reducedChiSquared && isFinite(reducedChiSquared) && reducedChiSquared > 0) ? reducedChiSquared : 1.0;
                 if (scaleFactor === 1.0 && degreesOfFreedom > 0 && isFinite(chiSquared)) { log("Reduced Chi-Squared is invalid or not positive. Using scale factor 1.0 for parameter errors, which might underestimate errors if fit is poor.", 'warn'); }
                 else if (scaleFactor === 1.0 && degreesOfFreedom <= 0) { log("Using scale factor 1.0 for parameter errors due to non-positive degrees of freedom.", 'info'); }

                 parameterErrors = covarianceMatrix.map((row, i) => {
                     if (i >= row.length) { log(`Error accessing covariance matrix diagonal at index ${i}. Matrix might be malformed.`, 'error'); return NaN; }
                     const variance = row[i]; const scaledVariance = variance * scaleFactor; let error = NaN;
                     if (isFinite(scaledVariance)) {
                         if (scaledVariance < 0) { log(`Negative variance (${scaledVariance.toExponential(3)}) encountered for active param ${i}. Returning sqrt(abs(variance)). Error estimate might be unreliable.`, 'warn'); error = Math.sqrt(Math.abs(scaledVariance)); }
                         else { error = Math.sqrt(scaledVariance); }
                     }
                     if (isNaN(error) && isFinite(scaledVariance)) { log(`NaN error calc for param ${i}: variance=${variance}, scaleFactor=${scaleFactor}, scaledVariance=${scaledVariance}`, 'debug'); }
                     return error;
                 });
                 if (parameterErrors.some(isNaN)) { log("NaN encountered in parameter errors (potentially due to non-finite variance/covariance). Check fit quality, model, initial parameters, and data.", 'warn'); }
             } catch (error) {
                 log(`Failed to calculate covariance matrix/parameter errors: ${error.message}`, 'error'); parameterErrors = new Array(K).fill(NaN); covarianceMatrix = new Array(K).fill(0).map(() => new Array(K).fill(NaN));
             }
         } else {
             if (K > 0 && !finalHessian) { log("Could not calculate covariance matrix (no valid final Hessian?). Parameter errors will be NaN.", 'warn'); }
             else if (K === 0) { log("No active parameters (K=0). Parameter errors are not applicable.", 'info'); }
             covarianceMatrix = new Array(K).fill(0).map(() => new Array(K).fill(NaN)); // Ensure it's NaN matrix if not calculated
         }


        // --- Reconstruct Final Parameters and Errors ---
        const finalReconstructedParams = reconstructParams(activeParameters);
        let finalParamErrors = null;
        try { finalParamErrors = []; let currentFlatIdx_err = 0; paramStructure.forEach((dsStruct, dsIdx) => { finalParamErrors[dsIdx] = []; dsStruct.forEach((pLen, pIdx) => { finalParamErrors[dsIdx][pIdx] = []; for (let vIdx = 0; vIdx < pLen; vIdx++) { let errorValue = NaN; if (isFixed[currentFlatIdx_err]) { errorValue = 0; } else if (masterMap[currentFlatIdx_err] !== -1) { const masterFlatIdx = masterMap[currentFlatIdx_err]; const masterInfo = activeParamInfo.find(info => info.flatIndex === masterFlatIdx); if (masterInfo) { if(parameterErrors && masterInfo.activeIndex < parameterErrors.length) { errorValue = parameterErrors[masterInfo.activeIndex]; } else { log(`Invalid index ${masterInfo.activeIndex} for parameterErrors array (length ${parameterErrors?.length}) while processing slave ${currentFlatIdx_err}. Setting error to NaN.`, 'warn'); errorValue = NaN; } } else { log(`Could not find active info for master parameter (flat index ${masterFlatIdx}) of slave (flat index ${currentFlatIdx_err}). Setting error to 0.`, 'warn'); errorValue = 0; } } else { const activeInfo = activeParamInfo.find(info => info.flatIndex === currentFlatIdx_err); if (activeInfo) { if(parameterErrors && activeInfo.activeIndex < parameterErrors.length) { errorValue = parameterErrors[activeInfo.activeIndex]; } else { log(`Invalid index ${activeInfo.activeIndex} for parameterErrors array (length ${parameterErrors?.length}) while processing active param ${currentFlatIdx_err}. Setting error to NaN.`, 'warn'); errorValue = NaN; } } else { log(`Could not find active info for supposedly active parameter at flat index ${currentFlatIdx_err}. Setting error to NaN.`, 'warn'); } } finalParamErrors[dsIdx][pIdx][vIdx] = errorValue; currentFlatIdx_err++; } }); }); }
        catch (e) { log(`Error constructing finalParamErrors structure: ${e.message}`, 'error'); finalParamErrors = null; }

        // --- Calculate Final Residuals ---
        let finalResiduals = null;
        try { finalResiduals = calculateFinalResiduals(data, modelFunction, finalReconstructedParams, log); }
        catch (error) { log(`Failed to calculate final residuals: ${error.message}`, 'error'); }

        // --- Calculate Fitted Model Curves (potentially with CI) ---
        let fittedCurvesResult = null; // Store the object {curves, negativeVarianceEncountered}
        let bootstrapUsed = false;

        if (shouldCalculateFitted) {
             log(`Calculating fitted model curves with ${numPointsForCurve} points...`, 'info');
             if (confidenceLevel !== null) { log(`Confidence interval bands requested (${confidenceLevel*100}%).`, 'info'); }
             try {
                const standardResult = calculateFittedModelCurves(
                    data, modelFunction, finalReconstructedParams, numPointsForCurve, log,
                    confidenceLevel, // The validated level (or null)
                    activeParameters, // Final active parameter values
                    covarianceMatrix, // The calculated (or null/NaN) covariance matrix
                    activeParamInfo, // Needed for mapping active params to perturbations
                    reconstructParams, // Needed for evaluating perturbed models
                    epsilon, // Step size for differentiation
                    degreesOfFreedom, // Needed for t-value
                    K, // Number of active parameters
                    model_x_range_list // Pass the model_x_range_list option
                );
                fittedCurvesResult = standardResult;

                if (bootstrapFallback && confidenceLevel !== null && standardResult.negativeVarianceEncountered) {
                    log("Negative variance detected... Attempting bootstrap fallback...", 'warn');
                    bootstrapUsed = true;

                    const baseOptionsForBootstrap = {
                        maxIterations, errorTolerance, gradientTolerance,
                        linkMap: linkMapInput, fixMap: fixMapInput, constraints, constraintFunction,
                        robustCostFunction, epsilon, covarianceLambda
                    };

                    const bootstrapCurveResult = await calculateBootstrapCIBands(
                        data, modelFunction, finalReconstructedParams, activeParamInfo, reconstructParams, paramStructure,
                        numBootstrapSamples, confidenceLevel, numPointsForCurve, log, baseOptionsForBootstrap,
                        model_x_range_list // Pass the range list
                    );

                    if (bootstrapCurveResult) {
                        log("Bootstrap CI calculation completed. Using bootstrap bands.", 'info');
                        fittedCurvesResult = { curves: bootstrapCurveResult, negativeVarianceEncountered: false };
                    } else {
                        log("Bootstrap CI calculation failed. Retaining standard results.", 'error');
                    }
                }
             } catch (curveError) {
                log(`Failed to calculate fitted model curves: ${curveError.message}`, 'error');
                fittedCurvesResult = { curves: null, negativeVarianceEncountered: false };
             }
        }

        // --- Calculate Component Model Curves ---
        const calculateComponentModelsOpt = options.calculateComponentModels ?? false;
        let fittedModelComponentCurves = null;
        if (calculateComponentModelsOpt) {
            if (!modelFunction || !finalReconstructedParams) {
                log("Skipping component curve calculation due to missing models or final parameters.", 'warn');
            } else {
                try {
                    fittedModelComponentCurves = calculateComponentCurves(
                        modelFunction,
                        finalReconstructedParams,
                        data,
                        numPointsForCurve,
                        log,
                        model_x_range_list // Pass the model_x_range_list option
                    );
                    if (fittedModelComponentCurves === null) {
                        log("Component curve calculation failed or returned null.", 'warn');
                    }
                } catch (compError) {
                    log(`Error during component curve calculation: ${compError.message}`, 'error');
                    console.error("Component Curve Exception:", compError);
                    fittedModelComponentCurves = null;
                }
            }
        } else {
            log("Component curve calculation not requested.", 'info');
        }

        // --- Final Logging ---
        log("--------------------", 'info'); log("lmFitGlobal Finished.", 'info'); log(`Iterations Performed: ${iterationsPerformed}`, 'info'); log(`Total Points (N): ${totalPoints}`, 'info'); log(`Active Parameters (K): ${K}`, 'info'); log(`Degrees of Freedom (N-K): ${degreesOfFreedom}`, 'info'); log(`Final Active Parameters: ${activeParameters.map(p=>p.toExponential(5)).join(', ')}`, 'info'); log(`Final Chi-Squared: ${chiSquared}`, 'info'); log(`Reduced Chi-Squared: ${reducedChiSquared}`, 'info'); log(`AIC: ${aic}`, 'info'); log(`AICc: ${aicc}`, 'info'); log(`BIC: ${bic}`, 'info'); log(`Parameter Errors (Active): ${parameterErrors.map(p=>isNaN(p)?'NaN':p.toExponential(3)).join(', ')}`, 'info'); log(`Converged: ${converged}`, 'info'); log("--------------------", 'info');

        // --- Return Results ---
        return {
            p_active: activeParameters, p_reconstructed: finalReconstructedParams, finalParamErrors,
            chiSquared, covarianceMatrix, parameterErrors, iterations: iterationsPerformed, converged,
            activeParamLabels, error: null, totalPoints, degreesOfFreedom, reducedChiSquared,
            aic, aicc, bic, residualsPerSeries: finalResiduals, fittedModelCurves: fittedCurvesResult?.curves ?? null, // Contains CI bands if calculated
            bootstrapUsed: bootstrapUsed, // <<< Add the flag to the result
            fittedModelComponentCurves: fittedModelComponentCurves // Add component curves to the result
        };
    } // <-- End of lmFitGlobal function definition

    /**
     * Calculates the individual fitted model component curves after fitting.
     *
     * @param {Function[][]} modelFunctionStructure - Array of arrays of the model functions used.
     * @param {number[][][]} finalReconstructedParams - Final fitted parameters in nested structure.
     * @param {object} data - Original data object (used for x-ranges).
     * @param {number} numPoints - Number of points for the curve calculation.
     * @param {Function} onLog - Logging function.
     * @param {Array<number[]>} model_x_range_list - List of optional ranges
     * @returns {Array<Array<{x: number[], y: number[]}>> | null} - Nested array [dsIdx][modelIdx] of curve objects, or null on error.
     * @private
     */
    function calculateComponentCurves(
        modelFunctionStructure, finalReconstructedParams, data, numPoints, onLog,
        model_x_range_list // <<< ADDED
    ) {
        const logFn = (message, level) => { if (onLog) onLog(message, level); else console.log(`[${level}] ${message}`); };
        logFn("Calculating individual component model curves...", 'info');

        // --- Input Validation ---
        if (!modelFunctionStructure || !finalReconstructedParams || !data || !data.x) {
            logFn("Missing required arguments for component curve calculation.", 'error');
            return null;
        }
        if (numPoints <= 1) {
            logFn(`Invalid numPoints (${numPoints}) for component curve calculation.`, 'error');
            return null;
        }
        if (modelFunctionStructure.length !== finalReconstructedParams.length || modelFunctionStructure.length !== data.x.length) {
            logFn("Structure mismatch between models, params, and data for component curves.", 'error');
            return null;
        }

        const allComponentCurves = [];
        let overallSuccess = true;

        try {
            modelFunctionStructure.forEach((datasetModels, dsIdx) => {
                allComponentCurves[dsIdx] = []; // Initialize array for this dataset's components

                if (!finalReconstructedParams[dsIdx] || !data.x[dsIdx]) {
                    logFn(`Skipping component curves for dataset ${dsIdx}: Missing params or x-data.`, 'warn');
                    return;
                }
                const xDataset = data.x[dsIdx];

                // --- Determine X Range using helper ---
                const currentSeriesRangeOpt = model_x_range_list?.[dsIdx] ?? null;
                const xRange = _determineXRange(xDataset, currentSeriesRangeOpt, dsIdx, logFn);
                if (xRange === null) {
                    logFn(`Invalid X range for dataset ${dsIdx}. Skipping component curves.`, 'warn');
                    allComponentCurves[dsIdx] = new Array(datasetModels.length).fill(null); return;
                }
                const { min: xMin, max: xMax } = xRange;
                // --- End Determine X Range ---

                const dx = (xMax === xMin || numPoints <= 1) ? 0 : (xMax - xMin) / (numPoints - 1);

                // --- Iterate through components for this dataset ---
                datasetModels.forEach((modelFunc, modelIdx) => {
                    if (!modelFunc || typeof modelFunc !== 'function') {
                        logFn(`Skipping component curve for ds ${dsIdx}, model ${modelIdx}: Invalid model function.`, 'warn');
                        allComponentCurves[dsIdx][modelIdx] = null;
                        return;
                    }
                    if (!finalReconstructedParams[dsIdx][modelIdx]) {
                        logFn(`Skipping component curve for ds ${dsIdx}, model ${modelIdx}: Missing parameters.`, 'warn');
                        allComponentCurves[dsIdx][modelIdx] = null;
                        return;
                    }

                    const componentParams = finalReconstructedParams[dsIdx][modelIdx];
                    const componentCurve = { x: [], y: [] };
                    let componentError = false;

                    // --- Calculate points over determined range ---
                    for (let i = 0; i < numPoints; i++) {
                        const xCalc = (dx === 0) ? xMin : xMin + i * dx;
                        let yComp = NaN;
                        try {
                            const componentResult = modelFunc(componentParams, [xCalc]);
                            if (!componentResult || componentResult.length !== 1 || !isFinite(componentResult[0])) {
                                throw new Error(`Component model ${dsIdx}-${modelIdx} returned invalid result: ${JSON.stringify(componentResult)}`);
                            }
                            yComp = componentResult[0];
                        } catch (evalError) {
                            logFn(`Error evaluating component model ${dsIdx}-${modelIdx} at x=${xCalc.toFixed(3)}: ${evalError.message}`, 'error');
                            yComp = NaN;
                            componentError = true;
                            overallSuccess = false;
                        }
                        componentCurve.x.push(xCalc);
                        componentCurve.y.push(yComp);
                    }

                    if (componentError) {
                        logFn(`Component curve calculation for ds ${dsIdx}, model ${modelIdx} encountered errors.`, 'warn');
                    }
                    allComponentCurves[dsIdx][modelIdx] = componentCurve;
                });
            });
        } catch (outerError) {
            logFn(`Error processing datasets for component curve calculation: ${outerError.message}`, 'error');
            console.error("Component Curve Calculation Error:", outerError);
            return null;
        }

        return overallSuccess ? allComponentCurves : null;
    }

    // ============================================================================
    // Helper Functions for Wrappers & Wrappers (Unchanged)
    // ============================================================================
     function isSingleDataset(data) { return data && Array.isArray(data.x) && (data.x.length === 0 || !Array.isArray(data.x[0])); }
     function wrapSingleDatasetInput(data, modelFunction, initialParameters, options) {
        const wrappedData = { x: [data.x], y: [data.y], ye: [data.ye] };
        const wrappedModelFunction = [ Array.isArray(modelFunction) ? modelFunction : [modelFunction] ];
        const wrappedInitialParameters = [ initialParameters ];
        const wrappedOptions = { ...options };
        if (options.fixMap && (!Array.isArray(options.fixMap[0]) || !Array.isArray(options.fixMap[0][0]))) { wrappedOptions.fixMap = [options.fixMap]; }
        if (options.constraints && (!Array.isArray(options.constraints[0]) || !Array.isArray(options.constraints[0][0]))) { wrappedOptions.constraints = [options.constraints]; }
        if (options.linkMap && (!Array.isArray(options.linkMap[0]) || !Array.isArray(options.linkMap[0][0]) || !Array.isArray(options.linkMap[0][0][0]))) {
             wrappedOptions.linkMap = [options.linkMap];
        }
        // --- Wrap model_x_range ---
        if (options.model_x_range && Array.isArray(options.model_x_range) && options.model_x_range.length === 2) {
            // If user provided a single tuple [min, max], wrap it in a list
            wrappedOptions.model_x_range = [options.model_x_range];
        } else if (options.model_x_range === null || options.model_x_range === undefined) {
            wrappedOptions.model_x_range = [null]; // Explicitly wrap null
        } else {
            // If it's already a list or invalid, pass it as is (lmFitGlobal will handle/warn)
            wrappedOptions.model_x_range = options.model_x_range;
        }
        // --- End wrap model_x_range ---
        // confidenceInterval option is passed directly in wrappedOptions
        return { data: wrappedData, modelFunction: wrappedModelFunction, initialParameters: wrappedInitialParams, options: wrappedOptions };
    }

    /**
     * Fits a single dataset using the Levenberg-Marquardt algorithm.
     * @param {object} data - {x: number[], y: number[], ye: number[]}.
     * @param {Function | Function[]} modelFunction - A single model function or an array of model functions.
     * @param {number[] | number[][]} initialParameters - Initial guesses. Nested if multiple models.
     * @param {object} [options={}] - Optional configuration (see lmFitGlobal docs).
     *                                 `fixMap`, `constraints` in single-dataset format.
     *                                 `linkMap` in new single-dataset format.
     *                                 `confidenceInterval` as a number (e.g., 0.95).
     * @returns {Promise<object>} The fitting result object.
     */
    async function lmFit(data, modelFunction, initialParameters, options = {}) { // <<< Added async
         if (!isSingleDataset(data)) { throw new Error("lmFit requires single dataset input format."); }
         const modelFuncArray = Array.isArray(modelFunction) ? modelFunction : [modelFunction];
         const initialParamsNested = modelFuncArray.length === 1 && initialParameters.length > 0 && !Array.isArray(initialParameters[0]) ? [initialParameters] : initialParameters;
         const { data: wrappedData, modelFunction: wrappedModelFunc, initialParameters: wrappedInitialParams, options: wrappedOptions } = wrapSingleDatasetInput(data, modelFuncArray, initialParamsNested, options);
         const result = await lmFitGlobal(wrappedData, wrappedModelFunc, wrappedInitialParams, wrappedOptions); // <<< Added await
         // --- Unwrap results for single dataset ---
         if (result && !result.error) {
             // Fitted curves / CIs are already a list, just take the first element
             if (result.fittedModelCurves && Array.isArray(result.fittedModelCurves)) {
                 result.fittedModelCurves = result.fittedModelCurves[0] ?? null; // Get first curve object or null
             }
             // Component curves: unwrap outer list
             if (result.fittedModelComponentCurves && Array.isArray(result.fittedModelComponentCurves)) {
                 result.fittedModelComponentCurves = result.fittedModelComponentCurves[0] ?? null; // Get list of components for the first dataset
             }
             // Residuals: unwrap outer list
             if (result.residualsPerSeries && Array.isArray(result.residualsPerSeries)) {
                 result.residualsPerSeries = result.residualsPerSeries[0] ?? null;
             }
             // Params: unwrap outer list
             if (result.p_reconstructed && Array.isArray(result.p_reconstructed)) {
                 result.p_reconstructed = result.p_reconstructed[0] ?? null;
             }
             if (result.finalParamErrors && Array.isArray(result.finalParamErrors)) {
                 result.finalParamErrors = result.finalParamErrors[0] ?? null;
             }
         }
         return result;
    }

    /**
     * Fits multiple datasets independently (sequentially) using lmFitGlobal.
     * @param {object} data - {x: number[][], y: number[][], ye: number[][]}.
     * @param {Function[][]} modelFunction - Array of arrays of model functions.
     * @param {number[][][]} initialParameters - Nested array of initial parameter guesses.
     * @param {object} [options={}] - Optional configuration. `linkMap` is ignored. `confidenceInterval` applies to each fit.
     * @returns {Promise<object[]>} An array of fitting result objects.
     */
    async function lmFitIndependent(data, modelFunction, initialParameters, options = {}) { // <<< Added async
        if (isSingleDataset(data)) {
            console.warn("lmFitIndependent received single dataset input. Calling lmFit instead.");
            try {
                 const modelFuncArray = Array.isArray(modelFunction) ? modelFunction : [modelFunction];
                 const initialParamsInput = (modelFuncArray.length === 1 && initialParameters.length > 0 && !Array.isArray(initialParameters[0])) ? initialParameters : (initialParameters[0] ?? []);
                 const modelFuncInput = modelFuncArray.length === 1 ? modelFuncArray[0] : (modelFuncArray[0] ?? []);
                 // Need to handle potential nesting difference in options for the single call
                 const singleOptionsForFit = {...options};
                 if (options.fixMap && Array.isArray(options.fixMap[0]) && (!Array.isArray(options.fixMap[0][0]) || !Array.isArray(options.fixMap[0][0][0]))) { singleOptionsForFit.fixMap = options.fixMap[0]; }
                 if (options.constraints && Array.isArray(options.constraints[0]) && (!Array.isArray(options.constraints[0][0]) || !Array.isArray(options.constraints[0][0][0]))) { singleOptionsForFit.constraints = options.constraints[0]; }
                 if (options.linkMap && Array.isArray(options.linkMap[0]) && (!Array.isArray(options.linkMap[0][0]) || !Array.isArray(options.linkMap[0][0][0]))) { singleOptionsForFit.linkMap = options.linkMap[0]; }

                const result = await lmFit(data, modelFuncInput, initialParamsInput, singleOptionsForFit);
                return [result];
            } catch (e) { return [{ error: `Error processing single dataset input for lmFitIndependent: ${e.message}`, converged: false }]; }
        }
        const numDatasets = data.x?.length ?? 0; if (numDatasets === 0) return [];
        if (modelFunction?.length !== numDatasets || initialParameters?.length !== numDatasets || data.y?.length !== numDatasets || data.ye?.length !== numDatasets) { throw new Error("Input array lengths must match for lmFitIndependent."); }
        const allResults = []; const rootOnLog = options.onLog || (() => {}); const rootOnProgress = options.onProgress || (() => {}); const rootLogLevel = LOG_LEVELS[options.logLevel?.toLowerCase() ?? 'info'] ?? LOG_LEVELS.info;
        for (let i = 0; i < numDatasets; i++) {
            const datasetIndex = i;
            const logFnLoop = (message, level) => { const messageLevel = LOG_LEVELS[level] ?? LOG_LEVELS.info; if (rootLogLevel >= messageLevel) { rootOnLog(message, level, datasetIndex); } };
            logFnLoop(`--- Starting Independent Fit for Dataset ${datasetIndex} ---`, 'info');
            const singleData = { x: data.x[i], y: data.y[i], ye: data.ye[i] }; const singleModelFunc = modelFunction[i]; const singleInitialParams = initialParameters[i];
            const singleOptions = { ...options }; // Pass confidenceInterval option through
            if (options.fixMap && Array.isArray(options.fixMap[i])) { singleOptions.fixMap = options.fixMap[i]; } else { delete singleOptions.fixMap; }
            if (options.constraints && Array.isArray(options.constraints[i])) { singleOptions.constraints = options.constraints[i]; } else { delete singleOptions.constraints; }
            delete singleOptions.linkMap; // Link map doesn't apply to independent fits
            delete singleOptions.onLog; delete singleOptions.onProgress;
            singleOptions.onLog = (message, level) => { rootOnLog(message, level, datasetIndex); }; singleOptions.onProgress = (progressData) => { rootOnProgress(progressData, datasetIndex); };
            // Wrap the single dataset inputs for lmFitGlobal
            const { data: wrappedData, modelFunction: wrappedModelFunc, initialParameters: wrappedInitialParams, options: wrappedOptions } = wrapSingleDatasetInput(singleData, singleModelFunc, singleInitialParams, singleOptions);
            // --- Extract model_x_range for this dataset ---
            if (options.model_x_range && Array.isArray(options.model_x_range) && options.model_x_range.length > i) {
                singleOptions.model_x_range = options.model_x_range[i]; // Pass the specific tuple or null
            } else {
                delete singleOptions.model_x_range; // Remove if structure is wrong
            }
            // ---
            try { const result = await lmFitGlobal(wrappedData, wrappedModelFunc, wrappedInitialParams, wrappedOptions); allResults.push(result); } // <<< Added await
            catch (error) { logFnLoop(`Error fitting dataset ${datasetIndex}: ${error.message}`, 'error'); allResults.push({ error: `Fit failed: ${error.message}`, converged: false, p_active: [], p_reconstructed: singleInitialParams, finalParamErrors: null, chiSquared: NaN, covarianceMatrix: null, parameterErrors: [], iterations: 0, activeParamLabels: [], totalPoints: singleData.x?.length ?? 0, degreesOfFreedom: NaN, reducedChiSquared: NaN, aic: NaN, aicc: NaN, bic: NaN, residualsPerSeries: null, fittedModelCurves: null }); }
            logFnLoop(`--- Finished Independent Fit for Dataset ${datasetIndex} ---`, 'info');
        }
        return allResults;
    }

    // gaussianRandom (assuming Box-Muller is correct as provided in example)
    function gaussianRandom(mean = 0, stdev = 1) {
        let u = 0, v = 0;
        while(u === 0) u = Math.random(); //Converting [0,1) to (0,1)
        while(v === 0) v = Math.random();
        let num = Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
        return num * stdev + mean;
    }

    /**
     * Generates a Poisson-distributed random number.
     * Uses Knuth's algorithm for smaller means and Normal approximation for larger means.
     * @param {number} lambda - The mean (rate) of the Poisson distribution (must be >= 0).
     * @param {number} [gaussianThreshold=50] - Lambda value above which to switch to Gaussian approximation.
     * @returns {number} A non-negative integer drawn from the Poisson distribution.
     * @private
     */
    function poissonRandom(lambda, gaussianThreshold = 50) {
        if (lambda < 0) {
            console.warn(`poissonRandom: lambda cannot be negative (got ${lambda}). Returning 0.`);
            return 0;
        }
        if (lambda === 0) {
            return 0;
        }

        // Use Normal approximation for large lambda (Poisson approx Normal(lambda, sqrt(lambda)))
        if (lambda >= gaussianThreshold) {
            // Generate Gaussian N(lambda, sqrt(lambda)) and round to nearest non-negative integer
            const gaussianVal = gaussianRandom(lambda, Math.sqrt(lambda));
            return Math.max(0, Math.round(gaussianVal)); // Ensure non-negative integer
        }

        // Knuth's algorithm for smaller lambda
        const L = Math.exp(-lambda);
        let k = 0;
        let p = 1;
        do {
            k++;
            p *= Math.random(); // Generate u in [0,1)
        } while (p > L);

        return k - 1;
    }

    /**
 * Simulates y data based on provided x values, model functions, parameters, and noise options.
 * Adheres to the multi-dataset structure used by lmFitGlobal.
 *
 * @param {Array<Array<number>>} dataX - Array of arrays of independent variable values (x points) for each dataset.
 *                                       Defines the points at which to simulate y values.
 * @param {Array<Array<Function>>} modelFunctions - Array of arrays of model functions. Structure must match `parameters`.
 * @param {Array<Array<Array<number>>> parameters - Nested array of parameter values to use for the simulation.
 * @param {object} [options={}] - Optional configuration for noise.
 *   @param {number | Array<number> | null} [options.noiseStdDev=null] - Standard deviation for Gaussian noise (mean=0). Used if noiseType is 'gaussian'.
 *   @param {string | Array<string>} [options.noiseType='gaussian'] - Type of noise model ('gaussian', 'poisson', 'none').
 *   @param {Function} [options.logFn=console.log] - Function for logging warnings or errors during simulation.
 * @returns {{x: number[][], y: number[][]}|null} - An object containing the original `x` arrays and the simulated dependent variable `y`.
 */
function simulateFromParams(dataX, modelFunctions, parameters, options = {}) {
    const logFn = options.logFn && typeof options.logFn === 'function' ? options.logFn : console.log;

    // --- Input Validation ---
    try {
        if (!dataX || !modelFunctions || !parameters || !Array.isArray(dataX) || !Array.isArray(modelFunctions) || !Array.isArray(parameters)) {
            throw new Error("Missing or invalid required arguments (dataX, modelFunctions, parameters).");
        }
        const numDatasets = dataX.length;
        if (modelFunctions.length !== numDatasets || parameters.length !== numDatasets) {
            throw new Error("Input arrays (dataX, modelFunctions, parameters) must have the same outer length.");
        }
        for (let i = 0; i < numDatasets; i++) {
            if (!Array.isArray(dataX[i])) throw new Error(`dataX[${i}] is not an array.`);
            if (!Array.isArray(modelFunctions[i])) throw new Error(`modelFunctions[${i}] is not an array.`);
            if (!Array.isArray(parameters[i])) throw new Error(`parameters[${i}] is not an array.`);
            if (modelFunctions[i].length !== parameters[i].length) throw new Error(`Mismatch between number of models (${modelFunctions[i].length}) and parameter groups (${parameters[i].length}) for dataset ${i}.`);
        }
    } catch (validationError) {
        logFn(`Simulation Input Validation Error: ${validationError.message}`, 'error');
        return null;
    }

    // --- Noise Configuration Processing ---
    const noiseSettings = [];
    const numDatasets = dataX.length;
    const noiseStdDevOpt = options.noiseStdDev ?? null;
    const noiseTypeOpt = options.noiseType ?? null;

    for (let dsIdx = 0; dsIdx < numDatasets; dsIdx++) {
        let currentStdDev = 0;
        let currentType = 'none';

        if (typeof noiseStdDevOpt === 'number' && noiseStdDevOpt > 0) {
            currentStdDev = noiseStdDevOpt;
        } else if (Array.isArray(noiseStdDevOpt) && typeof noiseStdDevOpt[dsIdx] === 'number' && noiseStdDevOpt[dsIdx] > 0) {
            currentStdDev = noiseStdDevOpt[dsIdx];
        }

        let explicitType = null;
        if (typeof noiseTypeOpt === 'string') {
            explicitType = noiseTypeOpt.toLowerCase();
        } else if (Array.isArray(noiseTypeOpt) && typeof noiseTypeOpt[dsIdx] === 'string') {
            explicitType = noiseTypeOpt[dsIdx].toLowerCase();
        }

        if (explicitType === 'poisson') {
            currentType = 'poisson';
            currentStdDev = NaN;
        } else if (explicitType === 'gaussian') {
            currentType = (currentStdDev > 0) ? 'gaussian' : 'none';
        } else if (explicitType === 'none') {
            currentType = 'none';
        } else {
            currentType = (currentStdDev > 0) ? 'gaussian' : 'none';
        }

        noiseSettings[dsIdx] = { type: currentType, stdDev: currentStdDev };
        logFn(`Dataset ${dsIdx} Noise: type='${noiseSettings[dsIdx].type}', stdDev=${noiseSettings[dsIdx].stdDev}`, 'debug');
    }

    // --- Simulation Loop ---
    const simulatedY = [];
    try {
        for (let dsIdx = 0; dsIdx < numDatasets; dsIdx++) {
            const currentX = dataX[dsIdx];
            const currentModels = modelFunctions[dsIdx];
            const currentParams = parameters[dsIdx];
            const currentNoiseSetting = noiseSettings[dsIdx];
            const currentY = [];

            if (!currentX || currentX.length === 0) {
                logFn(`Skipping simulation for empty dataset ${dsIdx}.`, 'debug');
                simulatedY.push([]);
                continue;
            }

            for (let ptIdx = 0; ptIdx < currentX.length; ptIdx++) {
                const x = currentX[ptIdx];
                let y_true = 0;
                let modelEvalOk = true;

                try {
                    for (let modelIdx = 0; modelIdx < currentModels.length; modelIdx++) {
                        const func = currentModels[modelIdx];
                        const params = currentParams[modelIdx];
                        if (!func || !params) throw new Error(`Missing function or params for model ${modelIdx} in dataset ${dsIdx}.`);

                        const componentResult = func(params, [x]);
                        if (!componentResult || componentResult.length !== 1 || !isFinite(componentResult[0])) {
                            throw new Error(`Model ${modelIdx} returned non-finite result: ${JSON.stringify(componentResult)}`);
                        }
                        y_true += componentResult[0];
                    }
                    if (!isFinite(y_true)) throw new Error("Summed y_true is not finite.");
                } catch (modelError) {
                    logFn(`Model evaluation error at ds ${dsIdx}, point ${ptIdx} (x=${x}): ${modelError.message}. Setting y to NaN.`, 'warn');
                    y_true = NaN;
                    modelEvalOk = false;
                }

                let y_noisy = y_true;
                if (modelEvalOk) {
                    switch (currentNoiseSetting.type) {
                        case 'gaussian':
                            const noise = gaussianRandom(0, currentNoiseSetting.stdDev);
                            y_noisy = y_true + noise;
                            break;
                        case 'poisson':
                            const lambda = Math.max(0, y_true);
                            if (y_true < 0) {
                                logFn(`Warning: Clamping negative y_true (${y_true.toFixed(3)}) to 0 for Poisson noise calculation at ds ${dsIdx}, point ${ptIdx} (x=${x}).`, 'warn');
                            }
                            y_noisy = poissonRandom(lambda);
                            break;
                        case 'none':
                        default:
                            break;
                    }
                } else {
                    y_noisy = NaN;
                }

                currentY.push(y_noisy);
            }
            simulatedY.push(currentY);
        }
    } catch (simulationError) {
        logFn(`Simulation loop error: ${simulationError.message}`, 'error');
        console.error("Simulation Error Details:", simulationError);
        return null;
    }

    return { x: dataX, y: simulatedY };
}

    // Expose public functions
    global.lmFitGlobal = lmFitGlobal;
    global.lmFit = lmFit;
    global.lmFitIndependent = lmFitIndependent;
    global.simulateFromParams = simulateFromParams;

// Establish the root object, `window` in the browser, or `global` on the server.
})(typeof window !== 'undefined' ? window : global); // <-- End of the IIFE wrapper