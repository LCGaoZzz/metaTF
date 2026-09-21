//! rust_genie3 — PyO3 extension.
//!
//! Entry semantics mirror the R wrapper .GENIE3 (nCores==1 branch):
//!   * per target gene, in targetNames order:
//!       theseRegulatorNames <- setdiff(regulatorNames, targetName)
//!       mtry <- .setMtry(K, numRegulators)
//!       x <- exprMatrixT[, theseRegulatorNames]; y <- exprMatrixT[, targetName]
//!       im <- BuildTreeEns(...)[[12]];  im <- im/sum(im)
//!   * parity mode: ONE R Mersenne-Twister stream seeded by set_seed(seed)
//!     shared sequentially across all targets — replicates R's global RNG
//!     state threaded through successive .C calls bit-for-bit.
//!   * fast mode: rayon across targets; each target gets an independent MT
//!     stream derived from (seed, target ordinal) via SplitMix64. The
//!     deviation from parity is exactly the bootstrap draw sequence;
//!     documented in MANIFEST.json "fast_mode".

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray2, ToPyArray};
use rayon::prelude::*;

mod genie3;
mod rng;

use genie3::{set_mtry, BuildResult, EnsParams, KSpec};
use rng::{r_long_sum, splitmix64_target_seed, RMersenneTwister};

fn parse_k(k: &str) -> PyResult<KSpec> {
    match k {
        "sqrt" => Ok(KSpec::Sqrt),
        "all" => Ok(KSpec::All),
        other => other
            .parse::<usize>()
            .map(KSpec::K)
            .map_err(|_| PyValueError::new_err(format!(
                "K must be 'sqrt', 'all' or a positive integer, got {other:?}"
            ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_one_target(
    flat: &[f32],
    g: usize,
    n: usize,
    regs: &[i64],
    tgt: i64,
    n_trees: usize,
    ks: KSpec,
    et: bool,
    rng: &mut RMersenneTwister,
) -> Vec<f64> {
    let n_att = regs.len();
    // materialize the att-major core table (C core_table = c(x), column-major)
    let mut core_x = vec![0f32; n_att * n];
    for (a, &r) in regs.iter().enumerate() {
        let r = r as usize;
        for o in 0..n {
            core_x[a * n + o] = flat[o * g + r];
        }
    }
    let tp = tgt as usize;
    let y: Vec<f32> = (0..n).map(|o| flat[o * g + tp]).collect();
    let mtry = set_mtry(&ks, n_att);
    let params = EnsParams {
        n_trees,
        rf_k: mtry,
        et,
        bootstrap: !et, // RF: bootstrap_sampling=1; ET: 0
        min_node_size: 1, // R wrapper: nmin <- 1
    };
    let BuildResult { importance } = genie3::Core::build(&core_x, &y, n, n_att, params, rng);
    importance
}

/// Run BuildTreeEns for a list of targets.
///
/// expr_t: (n_samples, n_genes) float32, C-contiguous — the R exprMatrixT
///         after as.single(); sample o, gene g at expr_t[o*G + g].
/// regs_flat + offsets: CSR-style per-target regulator gene positions
///         (already setdiff'd against the target, order preserved).
/// tgt_pos: target gene position per target.
/// Returns (raw_importance_flat, normalized_flat); normalized is
/// raw / R-long-double-sum(raw) — the wrapper's im/sum(im).
#[pyfunction]
#[pyo3(signature = (expr_t, regs_flat, offsets, tgt_pos, n_trees, k, tree_method, mode, seed, n_threads=None))]
#[allow(clippy::too_many_arguments)]
fn genie3_targets(
    py: Python<'_>,
    expr_t: PyReadonlyArray2<f32>,
    regs_flat: Vec<i64>,
    offsets: Vec<i64>,
    tgt_pos: Vec<i64>,
    n_trees: usize,
    k: &str,
    tree_method: &str,
    mode: &str,
    seed: u32,
    n_threads: Option<usize>,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    if offsets.len() != tgt_pos.len() + 1 {
        return Err(PyValueError::new_err("offsets must have len(targets)+1 entries"));
    }
    if regs_flat.len() as i64 != *offsets.last().unwrap() {
        return Err(PyValueError::new_err("offsets[-1] must equal len(regs_flat)"));
    }
    let ks = parse_k(k)?;
    let et = match tree_method {
        "RF" => false,
        "ET" => true,
        other => {
            return Err(PyValueError::new_err(format!(
                "tree_method must be 'RF' or 'ET', got {other:?}"
            )))
        }
    };
    if et && offsets.windows(2).any(|w| w[1] == w[0]) {
        return Err(PyValueError::new_err(
            "ET mode with 0 regulators for a target is undefined in the C code (UB); refusing",
        ));
    }

    let mat = expr_t.as_array();
    let (n, g) = (mat.shape()[0], mat.shape()[1]);
    let flat = mat
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("expr_t must be C-contiguous (pass np.ascontiguousarray)"))?;
    let n_targets = tgt_pos.len();

    let raw: Vec<Vec<f64>> = match mode {
        "parity" => {
            // single stream, strictly sequential — n_threads must not matter
            let mut rng = RMersenneTwister::from_set_seed(seed);
            (0..n_targets)
                .map(|t| {
                    let regs = &regs_flat[offsets[t] as usize..offsets[t + 1] as usize];
                    run_one_target(flat, g, n, regs, tgt_pos[t], n_trees, ks, et, &mut rng)
                })
                .collect()
        }
        "fast" => {
            let run = |t: usize| -> Vec<f64> {
                let tseed = splitmix64_target_seed(seed, t);
                let mut rng = RMersenneTwister::from_set_seed(tseed);
                let regs = &regs_flat[offsets[t] as usize..offsets[t + 1] as usize];
                run_one_target(flat, g, n, regs, tgt_pos[t], n_trees, ks, et, &mut rng)
            };
            match n_threads {
                None => (0..n_targets).into_par_iter().map(run).collect(),
                Some(1) => (0..n_targets).map(run).collect(),
                Some(nt) => {
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(nt)
                        .build()
                        .map_err(|e| PyValueError::new_err(format!("rayon pool: {e}")))?;
                    pool.install(|| (0..n_targets).into_par_iter().map(run).collect())
                }
            }
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "mode must be 'parity' or 'fast', got {other:?}"
            )))
        }
    };

    // wrapper-level normalization: im <- im / sum(im) with R's long-double sum
    let mut out_raw = Vec::with_capacity(regs_flat.len());
    let mut out_norm = Vec::with_capacity(regs_flat.len());
    for v in raw {
        let s = r_long_sum(&v);
        for &x in &v {
            out_raw.push(x);
            out_norm.push(x / s); // R: im/sum(im); 0/0 -> NaN like R
        }
    }
    Ok((
        out_raw.to_pyarray_bound(py).into(),
        out_norm.to_pyarray_bound(py).into(),
    ))
}

/// First n unif_rand() values after set.seed(seed) — for validating the RNG
/// port against a real R session.
#[pyfunction]
fn mt_first_unifs(py: Python<'_>, seed: u32, n: usize) -> Py<PyArray1<f64>> {
    let mut rng = RMersenneTwister::from_set_seed(seed);
    let v: Vec<f64> = (0..n).map(|_| rng.unif_rand()).collect();
    v.to_pyarray_bound(py).into()
}

/// R sum() semantics over doubles: sequential x87 long-double accumulation.
/// Exposed for parity testing of the normalization step.
#[pyfunction]
fn r_sum(py: Python<'_>, xs: Vec<f64>) -> Py<PyArray1<f64>> {
    let s = r_long_sum(&xs);
    vec![s].to_pyarray_bound(py).into()
}

#[pymodule]
pub fn rust_genie3(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(genie3_targets, m)?)?;
    m.add_function(wrap_pyfunction!(mt_first_unifs, m)?)?;
    m.add_function(wrap_pyfunction!(r_sum, m)?)?;
    m.add("__version__", "0.1.0")?;
    Ok(())
}
