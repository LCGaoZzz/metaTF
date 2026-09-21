//! rust_puic — PUIC（partial information decomposition and context）GRN 推断，
//! repo/R/PUIC.R 的 PyO3 移植（数值 parity 目标，f64，固定累加序，rayon 并行）。
//!
//! R→Rust 逐步对应表（行号按 LF 归一的 PUIC.R；完整版见 MANIFEST.json）：
//!   PUIC() L44-100 编排：
//!     L49   logScale → expMat <- log2(expMat+1)
//!     L62-63 逐基因离散化（as.list(as.data.frame(t(expMat))) 转置后按基因）
//!     L66/67 regulators/targets <- intersect(..., names(discret_list))（保序去重）
//!     L80-83 pblapply 逐 target → .getPUC（本 crate 的并行粒度同为 target）
//!     L86    do.call(cbind) → regulator(行) × target(列) PUC 矩阵
//!     L89    .FUxy 对称化 + 行/列 ECDF
//!     L91-98 diag auto/zero/one
//!   .discretizeGene L210-227（uniform_width 默认）：
//!     B = floor(sqrt(C))；breaks = seq(min,max,length.out=B+1)（by=(max-min)/B，
//!     端点恰为 min/max，与 R seq 的构造一致）；cut(include.lowest=TRUE) →
//!     第一个满足 v <= b_{k+1} 的箱（左闭右开）
//!   .freqTable L232-255：table(x,y) 计数 → ML 概率 counts/C
//!   .getPUC L260-271：逐 regulator 建 B×B 列联；p_i=rowSums, p_Z=colSums（=靶边际）
//!   .MI（L317 邻近）：H(p_i)+H(p_z)-H(p_ij)，自然对数，正项求和
//!   .specific.information L317-331：
//!     S_i[z] = Σ_b p(b|z)·(ln(p(z|b)) - ln(p_z[z]))，计数=0 的项按 R 的
//!     colSums(na.rm=TRUE) 语义丢弃（0·Inf=NaN 被剔除的等价跳过）
//!   .puc_per_target L273-315：
//!     pos bins = {z: Σ_i S_i[z] > 0} ∪ {z: p_z[z] > 0}
//!     每箱：x=S_·[z] 升序稳定排序；val(k0) = cumsum(k0) + x·(R-k0-2)（0 基，
//!           对应 R 1 基的 cumsum(k)+x(k)·(R-k-1)；对全并列块该值与并列序无关）
//!           red_i += val_i · p_z[z]
//!     PUC_i = (R-1) - red_i / MI_i（MI_i=0 时 IEEE 除法产生 ±Inf/NaN，与 R 一致）
//!   .FUxy L347-358：方阵时 mat <- mat + t(mat)（按位置对称化）；
//!     F_x = 行 ECDF（Fn(v)=#{≤v}/n）；F_y = 列 ECDF；输出 (F_x+F_y)*0.5
//!
//! 与 R 的已知偏差（MANIFEST uncertainties 全列）：
//!   * 常数基因（min==max）：R 的 cut() 会因 breaks 不唯一报错；这里全归箱 0。
//!   * NaN 传播：ECDF 排序用 total_cmp，NaN 聚为一组；R 的 ecdf 会产出 NA。

use numpy::{PyArray2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::HashMap;

/// 逐基因等宽离散化（.discretizeGene, uniform_width）。
/// out 长度 = x.len()；返回箱数 nb = floor(sqrt(C))（0 表示 C==0）。
/// 内部箱边界：by = (max-min)/nb；b_{k+1} = min + (k+1)*by，最后一档用 max 本身
/// （等价 R seq 端点恰为 max，保证 v=max 必然入箱、无 NA）。
fn discretize_uniform_width(x: &[f64], out: &mut [u32]) -> usize {
    let n = x.len();
    let nb = (n as f64).sqrt().floor() as usize;
    if nb == 0 {
        return 0;
    }
    let mut mn = f64::INFINITY;
    let mut mx = f64::NEG_INFINITY;
    for &v in x {
        if v < mn {
            mn = v;
        }
        if v > mx {
            mx = v;
        }
    }
    if !(mx > mn) {
        // 常数基因：R 的 cut 会报错（breaks 不唯一）；此处全归箱 0（见 MANIFEST）
        for b in out.iter_mut() {
            *b = 0;
        }
        return nb;
    }
    let by = (mx - mn) / (nb as f64);
    for (i, &v) in x.iter().enumerate() {
        let mut bin = nb - 1; // 最后一箱边界即 max，v <= max 恒真
        for k in 0..(nb - 1) {
            if v <= mn + ((k + 1) as f64) * by {
                bin = k;
                break;
            }
        }
        out[i] = bin as u32;
    }
    nb
}

/// 正项 Shannon 熵（自然对数）：H = -Σ_{p>0} p·ln(p)。
fn neg_entropy_positives(p: &[f64]) -> f64 {
    let mut h = 0f64;
    for &v in p {
        if v > 0.0 {
            h -= v * v.ln();
        }
    }
    h
}

/// 单个 (regulator, target) 对：B×B 列联 → MI 与逐箱 specific information。
/// counts 为 nb*nb 复用缓冲（函数内部清零）。
fn pair_stats(
    bins_r: &[u32],
    bins_t: &[u32],
    pz_target: &[f64],
    nb: usize,
    counts: &mut Vec<u32>,
) -> (f64, Vec<f64>) {
    let n = bins_r.len();
    for v in counts.iter_mut() {
        *v = 0;
    }
    for k in 0..n {
        let idx = (bins_r[k] as usize) * nb + (bins_t[k] as usize);
        counts[idx] += 1;
    }
    let total = n as f64;

    // p_i（行边际）与联合熵按行主序累加
    let mut p_i = vec![0f64; nb];
    let mut h_joint = 0f64;
    for i in 0..nb {
        let mut pi = 0f64;
        for z in 0..nb {
            let p = counts[i * nb + z] as f64 / total;
            pi += p;
            if p > 0.0 {
                h_joint -= p * p.ln();
            }
        }
        p_i[i] = pi;
    }
    // 列边际 == 靶基因边际（对每个 regulator 相同），直接用 pz_target
    let mi = neg_entropy_positives(&p_i) + neg_entropy_positives(pz_target) - h_joint;

    // .specific.information：S[z] = Σ_b p(b|z)·(ln(p(z|b)) - ln(p_z[z]))
    let mut ispec = vec![0f64; nb];
    for z in 0..nb {
        if pz_target[z] <= 0.0 {
            continue; // R：整列 NaN，colSums(na.rm=TRUE) → 0
        }
        let mut s = 0f64;
        for b in 0..nb {
            let cnt = counts[b * nb + z] as f64;
            if cnt <= 0.0 {
                continue; // R：0·Inf=NaN 被 na.rm 丢弃
            }
            let p_bz = cnt / total;
            let p_b_given_z = p_bz / pz_target[z];
            let p_z_given_b = p_bz / p_i[b]; // p_i[b] >= p_bz > 0
            s += p_b_given_z * ((p_z_given_b / pz_target[z]).ln());
        }
        ispec[z] = s;
    }
    (mi, ispec)
}

/// .puc_per_target：一个 target 的 PUC 列（长度 R，顺序 = regulators 顺序）。
fn puc_column(bins: &[u32], c: usize, nb: usize, reg_rows: &[usize], tgt_row: usize) -> Vec<f64> {
    let r = reg_rows.len();
    let tbins = &bins[tgt_row * c..(tgt_row + 1) * c];

    // 靶基因边际 p_Z（= .freqTable colSums，对每个 regulator 相同）
    let mut pz = vec![0f64; nb];
    for &b in tbins {
        pz[b as usize] += 1.0;
    }
    for v in pz.iter_mut() {
        *v /= c as f64;
    }

    let mut mi = vec![0f64; r];
    let mut ispec = vec![0f64; r * nb]; // 行主序 R × nb
    let mut counts = vec![0u32; nb * nb];
    for (ri, &g) in reg_rows.iter().enumerate() {
        let rbins = &bins[g * c..(g + 1) * c];
        let (m, s) = pair_stats(rbins, tbins, &pz, nb, &mut counts);
        mi[ri] = m;
        ispec[ri * nb..(ri + 1) * nb].copy_from_slice(&s);
    }

    // pos bins：union({z: ΣS>0}, {z: p_z>0})
    let mut sum_s = vec![0f64; nb];
    for z in 0..nb {
        let mut acc = 0f64;
        for ri in 0..r {
            acc += ispec[ri * nb + z];
        }
        sum_s[z] = acc;
    }

    let mut red = vec![0f64; r];
    let mut x = vec![0f64; r];
    let mut ord: Vec<u32> = (0..r as u32).collect();
    let mut val = vec![0f64; r];
    for z in 0..nb {
        if !(sum_s[z] > 0.0) && !(pz[z] > 0.0) {
            continue;
        }
        for ri in 0..r {
            x[ri] = ispec[ri * nb + z];
        }
        // sort(x)：升序稳定排序（并列保持原 regulator 顺序；对全并列块
        // val 与并列次序无关，故与 R 的 sort/order 稳定性差异无影响）
        ord.sort_by(|&a, &b| x[a as usize].total_cmp(&x[b as usize]));
        let mut cum = 0f64;
        for (k0, &o) in ord.iter().enumerate() {
            let xv = x[o as usize];
            cum += xv;
            // R 1 基：val(k) = cumsum(k) + x(k)·(R-k-1)；0 基 k0 → (R-k0-2)
            val[o as usize] = cum + xv * (r as f64 - k0 as f64 - 2.0);
        }
        for ri in 0..r {
            red[ri] += val[ri] * pz[z];
        }
    }

    let mut puc = vec![0f64; r];
    for ri in 0..r {
        // PUC = (R-1) - redundancy / I_XiZ；MI=0 时 IEEE ±Inf/NaN（与 R 一致）
        puc[ri] = (r as f64 - 1.0) - red[ri] / mi[ri];
    }
    puc
}

/// 原位 ECDF：vals[i] <- #{vals_j <= vals_i} / n（R ecdf(x)(x)，右连续、并列同值）。
/// 排序用 total_cmp；相等（含 NaN 组）共享该组上界计数。
fn ecdf_apply(vals: &mut [f64]) {
    let n = vals.len();
    if n == 0 {
        return;
    }
    let mut idx: Vec<u32> = (0..n as u32).collect();
    idx.sort_by(|&a, &b| vals[a as usize].total_cmp(&vals[b as usize]));
    let mut i = 0usize;
    while i < n {
        let mut j = i;
        while j + 1 < n && vals[idx[j + 1] as usize].total_cmp(&vals[idx[i] as usize]) == Ordering::Equal
        {
            j += 1;
        }
        let f = (j + 1) as f64 / n as f64;
        for &k in &idx[i..=j] {
            vals[k as usize] = f;
        }
        i = j + 1;
    }
}

/// R intersect(a, b)：保 a 的顺序、去重、要求存在于 b。
fn intersect_order_unique(a: &[String], b: &HashMap<&str, usize>) -> Vec<String> {
    let mut seen = HashMap::new();
    let mut out = Vec::new();
    for s in a {
        if b.contains_key(s.as_str()) && !seen.contains_key(s.as_str()) {
            seen.insert(s.clone(), ());
            out.push(s.clone());
        }
    }
    out
}

#[pyfunction(signature = (exp_mat, row_names, regulators=None, targets=None, log_scale=false, n_threads=None, diag="auto"))]
fn puic<'py>(
    py: Python<'py>,
    exp_mat: &Bound<'py, PyArray2<f64>>,
    row_names: Vec<String>,
    regulators: Option<Vec<String>>,
    targets: Option<Vec<String>>,
    log_scale: bool,
    n_threads: Option<usize>,
    diag: &str,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Vec<String>, Vec<String>)> {
    let dims = exp_mat.dims();
    let (g, c) = (dims[0], dims[1]);
    if g == 0 || c == 0 {
        return Err(PyValueError::new_err("exp_mat must be non-empty genes x cells"));
    }
    if row_names.len() != g {
        return Err(PyValueError::new_err(
            "row_names length must equal the number of rows of exp_mat",
        ));
    }
    let diag_mode: u8 = match diag {
        "auto" => 0,
        "zero" => 1,
        "one" => 2,
        _ => return Err(PyValueError::new_err("diag must be 'auto', 'zero' or 'one'")),
    };

    // 行名 → 行号（要求唯一）
    let mut name_to_row: HashMap<&str, usize> = HashMap::with_capacity(g);
    for (i, n) in row_names.iter().enumerate() {
        if name_to_row.insert(n.as_str(), i).is_some() {
            return Err(PyValueError::new_err("row_names must be unique"));
        }
    }

    // intersect 语义（L66/67）
    let reg_names = match &regulators {
        None => row_names.clone(),
        Some(v) => intersect_order_unique(v, &name_to_row),
    };
    let tgt_names = match &targets {
        None => row_names.clone(),
        Some(v) => intersect_order_unique(v, &name_to_row),
    };
    if reg_names.len() < 2 {
        return Err(PyValueError::new_err(
            "At least two regulators required (intersect against row_names)",
        ));
    }
    if tgt_names.is_empty() {
        return Err(PyValueError::new_err("At least one target required"));
    }
    let reg_rows: Vec<usize> = reg_names.iter().map(|n| name_to_row[n.as_str()]).collect();
    let tgt_rows: Vec<usize> = tgt_names.iter().map(|n| name_to_row[n.as_str()]).collect();

    let threads = n_threads.unwrap_or_else(|| {
        std::cmp::min(
            8,
            std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1),
        )
    });
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|e| PyValueError::new_err(format!("thread pool: {e}")))?;

    // 输入拷贝（可选 log2(x+1)，L49）
    let src = unsafe { exp_mat.as_slice() }.map_err(|_| {
        PyValueError::new_err("input must be a C-contiguous genes x cells float64 array")
    })?;
    let data: Vec<f64> = if log_scale {
        src.iter().map(|v| (v + 1.0).log2()).collect()
    } else {
        src.to_vec()
    };

    // 离散化（L62-63）：B = floor(sqrt(C)) 对所有基因一致
    let nb = (c as f64).sqrt().floor() as usize;
    if nb == 0 {
        return Err(PyValueError::new_err("need at least 1 cell so floor(sqrt(C)) >= 1"));
    }
    let mut bins = vec![0u32; g * c];
    pool.install(|| {
        bins.par_chunks_mut(c)
            .zip(data.par_chunks(c))
            .for_each(|(bo, xr)| {
                discretize_uniform_width(xr, bo);
            });
    });

    // 逐 target PUC 列（L80-83，并行粒度 = target）+ 顺序组装（L86）
    let r = reg_rows.len();
    let t = tgt_rows.len();
    let cols: Vec<Vec<f64>> = pool.install(|| {
        tgt_rows
            .par_iter()
            .map(|&tr| puc_column(&bins, c, nb, &reg_rows, tr))
            .collect()
    });
    let mut mat = vec![0f64; r * t];
    for (ti, col) in cols.iter().enumerate() {
        for (ri, &v) in col.iter().enumerate() {
            mat[ri * t + ti] = v;
        }
    }
    drop(cols);

    // .FUxy（L89 / L347-358）
    let out_arr = PyArray2::<f64>::zeros_bound(py, [r, t], false);
    {
        let out = unsafe { out_arr.as_slice_mut() }.map_err(|_| {
            PyValueError::new_err("failed to obtain output slice")
        })?;
        // 1) 方阵时按位置对称化：new[i,j] = old[i,j] + old[j,i]（对角 = 2·old）
        if r == t {
            for i in 0..r {
                for j in (i + 1)..t {
                    let a = mat[i * t + j];
                    let b = mat[j * t + i];
                    let s = a + b;
                    mat[i * t + j] = s;
                    mat[j * t + i] = s;
                }
                mat[i * t + i] += mat[i * t + i];
            }
        }
        // 2) F_y：列 ECDF（此时 mat 仍是对称化后的原值）。
        //    rayon 逐列计算到缓冲（各列独立），顺序摊平成行主序 fy_flat
        let fy_flat: Vec<f64> = {
            let fy_cols: Vec<Vec<f64>> = pool.install(|| {
                (0..t)
                    .into_par_iter()
                    .map(|j| {
                        let mut col = vec![0f64; r];
                        for (i, cv) in col.iter_mut().enumerate() {
                            *cv = mat[i * t + j];
                        }
                        ecdf_apply(&mut col);
                        col
                    })
                    .collect()
            });
            let mut flat = vec![0f64; r * t];
            for (j, col) in fy_cols.iter().enumerate() {
                for (i, &cv) in col.iter().enumerate() {
                    flat[i * t + j] = cv;
                }
            }
            flat
        };
        // 3) F_x：行 ECDF 原位覆盖 mat
        pool.install(|| {
            mat.par_chunks_mut(t).for_each(|row| ecdf_apply(row));
        });
        // 4) out = (F_x + F_y) * 0.5（按行三路 zip 并行）
        pool.install(|| {
            out.par_chunks_mut(t)
                .zip(mat.par_chunks(t))
                .zip(fy_flat.par_chunks(t))
                .for_each(|((orow, mrow), fyrow)| {
                    for j in 0..t {
                        orow[j] = (mrow[j] + fyrow[j]) * 0.5;
                    }
                });
        });
        // 5) diag zero/one（仅方阵有对角；R 的 diag() 对非方阵报错，这里同样拦截）
        if diag_mode != 0 {
            if r != t {
                return Err(PyValueError::new_err(
                    "diag='zero'/'one' requires a square result (regulators == targets)",
                ));
            }
            for i in 0..r {
                out[i * t + i] = if diag_mode == 1 { 0.0 } else { 1.0 };
            }
        }
    }

    Ok((out_arr, reg_names, tgt_names))
}

#[pyfunction(signature = (exp_mat, row_names, regulators=None, targets=None, log_scale=false, n_threads=None))]
fn puic_raw<'py>(
    py: Python<'py>,
    exp_mat: &Bound<'py, PyArray2<f64>>,
    row_names: Vec<String>,
    regulators: Option<Vec<String>>,
    targets: Option<Vec<String>>,
    log_scale: bool,
    n_threads: Option<usize>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Vec<String>, Vec<String>)> {
    // 与 puic() 相同的前半段（离散化 → 逐 target PUC 列），不做 .FUxy。
    // 用于分阶段数值定位 / bench 中间量对齐。
    let dims = exp_mat.dims();
    let (g, c) = (dims[0], dims[1]);
    if g == 0 || c == 0 {
        return Err(PyValueError::new_err("exp_mat must be non-empty genes x cells"));
    }
    if row_names.len() != g {
        return Err(PyValueError::new_err(
            "row_names length must equal the number of rows of exp_mat",
        ));
    }
    let mut name_to_row: HashMap<&str, usize> = HashMap::with_capacity(g);
    for (i, n) in row_names.iter().enumerate() {
        if name_to_row.insert(n.as_str(), i).is_some() {
            return Err(PyValueError::new_err("row_names must be unique"));
        }
    }
    let reg_names = match &regulators {
        None => row_names.clone(),
        Some(v) => intersect_order_unique(v, &name_to_row),
    };
    let tgt_names = match &targets {
        None => row_names.clone(),
        Some(v) => intersect_order_unique(v, &name_to_row),
    };
    if reg_names.len() < 2 {
        return Err(PyValueError::new_err("At least two regulators required"));
    }
    if tgt_names.is_empty() {
        return Err(PyValueError::new_err("At least one target required"));
    }
    let reg_rows: Vec<usize> = reg_names.iter().map(|n| name_to_row[n.as_str()]).collect();
    let tgt_rows: Vec<usize> = tgt_names.iter().map(|n| name_to_row[n.as_str()]).collect();
    let threads = n_threads.unwrap_or_else(|| {
        std::cmp::min(
            8,
            std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1),
        )
    });
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|e| PyValueError::new_err(format!("thread pool: {e}")))?;
    let src = unsafe { exp_mat.as_slice() }.map_err(|_| {
        PyValueError::new_err("input must be a C-contiguous genes x cells float64 array")
    })?;
    let data: Vec<f64> = if log_scale {
        src.iter().map(|v| (v + 1.0).log2()).collect()
    } else {
        src.to_vec()
    };
    let nb = (c as f64).sqrt().floor() as usize;
    if nb == 0 {
        return Err(PyValueError::new_err("need at least 1 cell"));
    }
    let mut bins = vec![0u32; g * c];
    pool.install(|| {
        bins.par_chunks_mut(c)
            .zip(data.par_chunks(c))
            .for_each(|(bo, xr)| {
                discretize_uniform_width(xr, bo);
            });
    });
    let r = reg_rows.len();
    let t = tgt_rows.len();
    let cols: Vec<Vec<f64>> = pool.install(|| {
        tgt_rows
            .par_iter()
            .map(|&tr| puc_column(&bins, c, nb, &reg_rows, tr))
            .collect()
    });
    let out_arr = PyArray2::<f64>::zeros_bound(py, [r, t], false);
    {
        let out = unsafe { out_arr.as_slice_mut() }.map_err(|_| {
            PyValueError::new_err("failed to obtain output slice")
        })?;
        for (ti, col) in cols.iter().enumerate() {
            for (ri, &v) in col.iter().enumerate() {
                out[ri * t + ti] = v;
            }
        }
    }
    Ok((out_arr, reg_names, tgt_names))
}

#[pymodule]
pub fn rust_puic(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(puic, m)?)?;
    m.add_function(wrap_pyfunction!(puic_raw, m)?)?;
    Ok(())
}
