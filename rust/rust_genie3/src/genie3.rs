//! Bit-level port of GENIE3 1.24.0 src/GENIE3.c (Pierre Geurts' tree code as
//! wrapped by Van Anh Huynh), RF + ET paths, regression forests with
//! variance-reduction importance. See MANIFEST.json for the C-line ->
//! Rust-function map; function names below mirror the C names.
//!
//! Precision contract (all sites audited against the C source):
//!   * CORETABLE_TYPE float: X/Y stored f32 (R: as.single at the .C boundary).
//!   * SCORE_TYPE double: all accumulation arithmetic in f64.
//!   * MIXED SITES (differ per C function — replicated per site):
//!     - summarize_vector_multiregr / find_a_threshold_at_random_multiregr:
//!       `tsm[..][2] += w*y*y`  == (w*y)*y  — pure f64, y promoted once.
//!     - find_the_best_threshold_multiregr:
//!       `tsm[1][2] += w*(y*y)` — y*y rounded in f32 FIRST, then f64 multiply.
//!     - get_vi_multiregr_separate: `tsm[..][2] += y*y` — f32 product, then
//!       promoted to f64 by the +=.
//!     - threshold midpoint: `bt = (float)((old_val+new_val as float)/2.0)`.
//!   * RNG consumers: get_random_integer / get_random_float (see rng.rs);
//!     consumption order is bootstrap -> (per split-node: mtry draws) ->
//!     ... -> one extra bootstrap after the last tree (C calls
//!     make_ls_vector(current_tree) once more inside the ensemble loop).
//!   * attribute_vector permutation PERSISTS across nodes and trees within
//!     one BuildTreeEns call; it is re-sorted only before the importance
//!     computation (GENIE3.c "Re-sort the variables").

use crate::rng::{get_random_float, get_random_integer, RMersenneTwister};

#[derive(Clone, Copy)]
pub struct EnsParams {
    pub n_trees: usize,
    pub rf_k: usize, // mtry (RF) / K (ET)
    pub et: bool,    // Extra-Trees randomisation
    pub bootstrap: bool,
    pub min_node_size: i64, // nmin = 1 in the R wrapper
}

pub struct BuildResult {
    /// raw variable importances (already divided by sum of tree weights, as
    /// compute_ltrees_variable_importance does), f64
    pub importance: Vec<f64>,
}

pub struct Core<'a> {
    x: &'a [f32], // att-major: x[att * n + obj]  (C core_table, column of c(x))
    y: &'a [f32], // C core_table_y
    n: usize,     // nb_obj_in_core_table
    n_att: usize, // nb_attributes
    p: EnsParams,

    // learning set state
    object_weight: Vec<f64>,
    cls: Vec<i64>, // current_learning_set entries are object ids
    cls_size: i64,
    attribute_vector: Vec<usize>,

    // tree tables (shared, preallocated to n_trees*(2n-1) nodes like the C)
    threshold: Vec<f32>,
    tested_att: Vec<i64>,
    left_succ: Vec<i64>,
    right_succ: Vec<i64>,
    ltrees: Vec<i64>,

    // table_score_multiregr[3][3] (single goal: 2*1+1 = 3 columns)
    tsm: [[f64; 3]; 3],
    v_tot: f64,
    info: f64,

    // best-split scratch (C globals best_* / current_threshold_*)
    best_attribute: i64,
    best_threshold: f32,
    best_threshold_score: f64,
    current_threshold: f32,
    current_threshold_score: f64,
    current_threshold_info: f64,

    index_nodes: i64,
    rng: &'a mut RMersenneTwister,
}

impl<'a> Core<'a> {
    pub fn build(
        x: &'a [f32],
        y: &'a [f32],
        n: usize,
        n_att: usize,
        p: EnsParams,
        rng: &'a mut RMersenneTwister,
    ) -> BuildResult {
        let maxnodes = p.n_trees * if n >= 1 { 2 * n - 1 } else { 0 }; // C: nbterms*(2*nbobj-1)
        let ntrees = p.n_trees;
        let mut core = Core {
            x,
            y,
            n,
            n_att,
            p,
            object_weight: vec![1.0; n],
            cls: (0..n as i64).collect(),
            cls_size: n as i64,
            attribute_vector: (0..n_att).collect(),
            threshold: vec![0.0; maxnodes],
            tested_att: vec![-1; maxnodes],
            left_succ: vec![-1; maxnodes],
            right_succ: vec![-1; maxnodes],
            ltrees: vec![0; ntrees],
            tsm: [[0.0; 3]; 3],
            v_tot: 0.0,
            info: 0.0,
            best_attribute: -1,
            best_threshold: 0.0,
            best_threshold_score: -1.0,
            current_threshold: 0.0,
            current_threshold_score: -1.0,
            current_threshold_info: 0.0,
            index_nodes: -1,
            rng,
        };
        core.build_one_tree_ensemble();
        // GENIE3.c BuildTreeEns: "Re-sort the variables" before importance
        core.attribute_vector = (0..n_att).collect();
        let mut vimp = vec![0.0f64; n_att];
        core.compute_ltrees_variable_importance(&mut vimp);
        BuildResult { importance: vimp }
    }

    // ----- data accessors ---------------------------------------------------

    #[inline]
    fn getattval(&self, obj: i64, att: usize) -> f32 {
        self.x[att * self.n + obj as usize]
    }

    #[inline]
    fn getobjy(&self, obj: i64) -> f32 {
        self.y[obj as usize]
    }

    // ----- multiregr scoring (tree-multiregr.c) ------------------------------

    /// summarize_vector_multiregr. NOTE accumulation site A: (w*y)*y in f64.
    fn summarize_vector_multiregr(&mut self, start: i64, end: i64) {
        for i in 0..3 {
            self.tsm[0][i] = 0.0;
        }
        for i in start..=end {
            let obj = self.cls[i as usize];
            let w = self.object_weight[obj as usize];
            self.tsm[0][0] += w;
            let y = self.getobjy(obj) as f64;
            self.tsm[0][1] += w * y;
            self.tsm[0][2] += (w * y) * y; // C: w*y*y == (w*y)*y, all double
        }
    }

    /// stop_splitting_criterio_multiregr (also sets v_tot).
    fn stop_splitting_criterio_multiregr(&mut self) -> bool {
        // single goal i=0: v_tot = t0[2] - t0[1]*t0[1]/t0[0]
        self.v_tot = self.tsm[0][2] - self.tsm[0][1] * self.tsm[0][1] / self.tsm[0][0];
        (self.v_tot / self.tsm[0][0]) <= 0.0
    }

    /// compute_multiregr_score_from_table.
    fn compute_multiregr_score_from_table(&mut self) -> f64 {
        self.tsm[2][0] = self.tsm[0][0] - self.tsm[1][0];
        // single goal i=0
        self.tsm[2][1] = self.tsm[0][1] - self.tsm[1][1];
        self.tsm[2][2] = self.tsm[0][2] - self.tsm[1][2];
        let y_tot_var = (self.tsm[1][2] - self.tsm[1][1] * self.tsm[1][1] / self.tsm[1][0]).abs();
        let n_tot_var = (self.tsm[2][2] - self.tsm[2][1] * self.tsm[2][1] / self.tsm[2][0]).abs();
        self.info = self.v_tot - (y_tot_var + n_tot_var);
        self.info / self.v_tot
    }

    // ----- threshold search ---------------------------------------------------

    /// find_the_best_threshold_multiregr (numerical attribute, RF + ET-symb).
    fn find_the_best_threshold_multiregr(&mut self, att: usize, start: i64, end: i64) {
        let mut best_score: f64 = -1.0;
        let mut best_info: f64 = -1.0;
        let mut best_threshold: f32 = 0.0;

        // init table row 1
        for i in 0..3 {
            self.tsm[1][i] = 0.0;
        }

        // sort the learning-set segment by this attribute
        self.quicksort_ls_vector(att, start, end);

        let mut old_val = self.getattval(self.cls[start as usize], att);
        for st in start..end {
            let obj = self.cls[st as usize];
            let w = self.object_weight[obj as usize];
            self.tsm[1][0] += w;
            let y = self.getobjy(obj);
            self.tsm[1][1] += w * (y as f64);
            // C: tsm[1][2] += w*(y*y) — y*y rounded in FLOAT first
            let yy = y * y;
            self.tsm[1][2] += w * (yy as f64);

            let new_val = self.getattval(self.cls[(st + 1) as usize], att);
            if new_val != old_val {
                let current_score = self.compute_multiregr_score_from_table();
                if current_score > best_score {
                    best_score = current_score;
                    best_info = self.info;
                    // C: best_threshold=(old_val+new_val)/2.0 assigned to float
                    let mid = (old_val + new_val) as f64 / 2.0;
                    let mut bt = mid as f32;
                    if old_val >= bt {
                        bt = new_val;
                    }
                    best_threshold = bt;
                }
                old_val = new_val;
            }
        }
        if best_score >= 0.0 {
            self.current_threshold = best_threshold;
            self.current_threshold_score = best_score;
            self.current_threshold_info = best_info;
        } else {
            self.current_threshold_score = -1.0;
        }
    }

    /// find_a_threshold_at_random_multiregr (ET numerical). RNG: exactly one
    /// get_random_float call when min != max.
    fn find_a_threshold_at_random_multiregr(&mut self, att: usize, start: i64, end: i64) {
        let mut min = self.getattval(self.cls[start as usize], att);
        let mut max = min;
        self.current_threshold_score = -1.0;
        for i in (start + 1)..=end {
            let val = self.getattval(self.cls[i as usize], att);
            if val < min {
                min = val;
            } else if val > max {
                max = val;
            }
        }
        if min == max {
            return;
        }
        let rf = get_random_float(self.rng);
        self.current_threshold = max - (max - min) * rf; // f32 arithmetic
        for i in 0..3 {
            self.tsm[1][i] = 0.0;
        }
        for i in start..=end {
            if self.getattval(self.cls[i as usize], att) < self.current_threshold {
                let obj = self.cls[i as usize];
                let w = self.object_weight[obj as usize];
                self.tsm[1][0] += w;
                let y = self.getobjy(obj) as f64;
                self.tsm[1][1] += w * y;
                self.tsm[1][2] += (w * y) * y; // C: w*y*y == (w*y)*y
            }
        }
        self.current_threshold_score = self.compute_multiregr_score_from_table();
    }

    // ----- sorting (Numerical-Recipes quicksort, verbatim) --------------------

    /// quicksort_ls_vector. ls segment = self.cls[start..=end], VAL(o) =
    /// getattval(o, att). Includes the C's early `return` when jstack > 50
    /// (leaves the segment partially sorted — bug-compatible).
    fn quicksort_ls_vector(&mut self, att: usize, start: i64, end: i64) {
        const M_QS: i64 = 7;
        const STACK_SIZE: usize = 52; // C array is [50]; 52 avoids UB in the
        // unreachable jstack==50 corner while preserving reachable behaviour.
        let mut istack = [0i64; STACK_SIZE];
        let mut jstack: i64 = -1;
        let mut l = start;
        let mut ir = end;

        loop {
            if ir - l < M_QS {
                // insertion sort
                let mut j = l + 1;
                while j <= ir {
                    let o = self.cls[j as usize];
                    let a = self.getattval(o, att);
                    let mut i = j - 1;
                    while i >= l {
                        if self.getattval(self.cls[i as usize], att) <= a {
                            break;
                        }
                        self.cls[(i + 1) as usize] = self.cls[i as usize];
                        i -= 1;
                    }
                    self.cls[(i + 1) as usize] = o;
                    j += 1;
                }
                if jstack == -1 {
                    break;
                }
                ir = istack[jstack as usize];
                jstack -= 1;
                l = istack[jstack as usize];
                jstack -= 1;
            } else {
                let k = (l + ir) >> 1;
                self.cls.swap(k as usize, (l + 1) as usize);
                if self.getattval(self.cls[l as usize], att)
                    > self.getattval(self.cls[ir as usize], att)
                {
                    self.cls.swap(l as usize, ir as usize);
                }
                if self.getattval(self.cls[(l + 1) as usize], att)
                    > self.getattval(self.cls[ir as usize], att)
                {
                    self.cls.swap((l + 1) as usize, ir as usize);
                }
                if self.getattval(self.cls[l as usize], att)
                    > self.getattval(self.cls[(l + 1) as usize], att)
                {
                    self.cls.swap(l as usize, (l + 1) as usize);
                }
                let mut i = l + 1;
                let mut j = ir;
                let o = self.cls[(l + 1) as usize];
                let a = self.getattval(o, att);
                loop {
                    loop {
                        i += 1;
                        if !(self.getattval(self.cls[i as usize], att) < a) {
                            break;
                        }
                    }
                    loop {
                        j -= 1;
                        if !(self.getattval(self.cls[j as usize], att) > a) {
                            break;
                        }
                    }
                    if j < i {
                        break;
                    }
                    self.cls.swap(i as usize, j as usize);
                }
                self.cls[(l + 1) as usize] = self.cls[j as usize];
                self.cls[j as usize] = o;
                jstack += 2;
                if jstack > 50 {
                    // C: "Stack too small in quicksort" -> return
                    return;
                }
                if ir - i + 1 >= j - l {
                    istack[jstack as usize] = ir;
                    istack[(jstack - 1) as usize] = i;
                    ir = j - 1;
                } else {
                    istack[jstack as usize] = j - 1;
                    istack[(jstack - 1) as usize] = l;
                    l = i;
                }
            }
        }
    }

    // ----- split machinery -----------------------------------------------------


    /// find_the_best_split_among_k (RF). RNG: one draw per loop iteration.
    /// attribute_vector permutation persists after this call (C behaviour).
    fn find_the_best_split_among_k(&mut self, start: i64, end: i64) {
        self.best_attribute = -1;
        self.best_threshold_score = -1.0;

        let mut remaining_att = self.n_att;
        let mut i = 0usize;
        while i < self.p.rf_k && remaining_att != 0 {
            let pos = get_random_integer(self.rng, remaining_att);
            let att = self.attribute_vector[pos];
            self.find_the_best_threshold_multiregr(att, start, end);
            if self.current_threshold_score > self.best_threshold_score {
                self.best_threshold_score = self.current_threshold_score;
                self.best_threshold = self.current_threshold;
                self.best_attribute = att as i64;
            }
            remaining_att -= 1;
            if remaining_att != 0 {
                self.attribute_vector.swap(pos, remaining_att);
            }
            i += 1;
        }
    }

    /// find_a_split_at_random_et (ET). RNG: one draw per attempted attribute
    /// (plus one get_random_float per non-constant threshold evaluation).
    fn find_a_split_at_random_et(&mut self, start: i64, end: i64) {
        const RANDOM_SPLIT_SCORE_THRESHOLD: f64 = 10.0; // set_tree_param
        self.best_attribute = -1;
        self.best_threshold_score = -1.0;
        let mut nb_try: i64 = 0;
        let mut remaining_att = self.n_att;
        loop {
            nb_try += 1;
            let pos = get_random_integer(self.rng, remaining_att);
            let att = self.attribute_vector[pos];
            self.find_a_threshold_at_random_multiregr(att, start, end);
            if self.current_threshold_score > self.best_threshold_score {
                self.best_threshold_score = self.current_threshold_score;
                self.best_threshold = self.current_threshold;
                self.best_attribute = att as i64;
            }
            remaining_att -= 1;
            if remaining_att != 0 {
                self.attribute_vector.swap(pos, remaining_att);
            }
            if self.current_threshold_score < 0.0 {
                nb_try -= 1;
            }
            if !((remaining_att != 0)
                && (self.best_threshold_score < RANDOM_SPLIT_SCORE_THRESHOLD)
                && (nb_try < self.p.rf_k as i64))
            {
                break;
            }
        }
    }

    // ----- learning sets ---------------------------------------------------------

    /// make_ls_vector_bagging. RNG: exactly n draws (n = global learning set size).
    fn make_ls_vector_bagging(&mut self) -> f64 {
        let n = self.n;
        for w in self.object_weight.iter_mut() {
            *w = 0.0;
        }
        for _ in 0..n {
            let rn = get_random_integer(self.rng, n);
            self.object_weight[rn] += 1.0;
        }
        self.cls_size = 0;
        for i in 0..n {
            if self.object_weight[i] != 0.0 {
                self.cls[self.cls_size as usize] = i as i64;
                self.cls_size += 1;
            }
        }
        1.0
    }

    /// make_ls_vector_identity (ET): no RNG, learning set untouched.
    fn make_ls_vector_identity(&mut self) -> f64 {
        1.0
    }

    fn make_ls_vector(&mut self) -> f64 {
        if self.p.bootstrap {
            self.make_ls_vector_bagging()
        } else {
            self.make_ls_vector_identity()
        }
    }

    // ----- tree building -----------------------------------------------------------

    /// build_one_tree. Returns root node index.
    fn build_one_tree(&mut self) -> i64 {
        self.index_nodes += 1;
        let tree = self.index_nodes;
        self.left_succ[tree as usize] = -1;
        self.right_succ[tree as usize] = -1;
        self.tested_att[tree as usize] = -1;

        let mut stack: Vec<(i64, i64, i64)> = Vec::with_capacity(1024);
        stack.push((tree, 0, self.cls_size - 1));

        while let Some((node, start, end)) = stack.pop() {
            self.summarize_vector_multiregr(start, end);
            let nodesize = end - start + 1;
            if nodesize == 1 || nodesize < self.p.min_node_size || self.stop_splitting_criterio_multiregr() {
                // leaf (multiregr_savepred == 0: nothing stored)
                continue;
            }
            if self.p.et {
                self.find_a_split_at_random_et(start, end);
            } else {
                self.find_the_best_split_among_k(start, end);
            }
            // not_significant_test_multiregr(): leaf when !(score >= 0.0)
            if !(self.best_threshold_score >= 0.0) {
                continue;
            }
            let att = self.best_attribute as usize;
            let thr = self.best_threshold;
            let borne = separate_ls(att, thr, self.x, self.n, &mut self.cls, start, end);

            self.index_nodes += 1;
            let left = self.index_nodes;
            self.index_nodes += 1;
            let right = self.index_nodes;
            self.left_succ[left as usize] = -1;
            self.right_succ[left as usize] = -1;
            self.tested_att[left as usize] = -1;
            self.left_succ[right as usize] = -1;
            self.right_succ[right as usize] = -1;
            self.tested_att[right as usize] = -1;

            self.threshold[node as usize] = thr;
            self.tested_att[node as usize] = self.best_attribute;
            self.left_succ[node as usize] = left - node;
            self.right_succ[node as usize] = right - node;

            // C pushes left, then right on top (right processed first)
            stack.push((left, start, borne - 1));
            stack.push((right, borne, end));
        }
        tree
    }

    /// build_one_tree_ensemble. RNG consumption: initial make_ls_vector, one
    /// per tree inside the loop (so n_trees+1 bootstrap vectors total — the
    /// last one is drawn and discarded, matching the C loop structure).
    fn build_one_tree_ensemble(&mut self) {
        self.index_nodes = -1;
        self.make_ls_vector(); // make_ls_vector(-1): bootstrap before tree 0
        for t in 0..self.p.n_trees {
            let current_tree = self.build_one_tree();
            let _current_weight = self.make_ls_vector(); // 1.0; consumes RNG
            self.ltrees[t] = current_tree;
        }
    }

    // ----- variable importance ------------------------------------------------------

    /// compute_multiregr_score_from_table_for_varimp (single goal).
    fn compute_multiregr_score_from_table_for_varimp(&mut self, vi: &mut [f64]) {
        self.tsm[2][0] = self.tsm[0][0] - self.tsm[1][0];
        // i = 0
        self.v_tot = self.tsm[0][2] - self.tsm[0][1] * self.tsm[0][1] / self.tsm[0][0];
        self.tsm[2][1] = self.tsm[0][1] - self.tsm[1][1];
        self.tsm[2][2] = self.tsm[0][2] - self.tsm[1][2];
        let y_tot_var = (self.tsm[1][2] - self.tsm[1][1] * self.tsm[1][1] / self.tsm[1][0]).abs();
        let n_tot_var = (self.tsm[2][2] - self.tsm[2][1] * self.tsm[2][1] / self.tsm[2][0]).abs();
        vi[0] = self.v_tot - (y_tot_var + n_tot_var);
    }

    /// get_vi_multiregr_separate. Accumulation site C: y*y in f32, promoted.
    fn get_vi_multiregr_separate(&mut self, ts: &[i64], start: i64, end: i64, borne: i64, vi: &mut [f64]) {
        for i in 0..3 {
            self.tsm[0][i] = 0.0;
            self.tsm[1][i] = 0.0;
        }
        for i in start..=end {
            let obj = ts[i as usize];
            self.tsm[0][0] += 1.0; // C: table++ on double
            let y = self.getobjy(obj);
            self.tsm[0][1] += y as f64;
            let yy = y * y; // f32 product
            self.tsm[0][2] += yy as f64;
        }
        if start >= borne || borne > end {
            vi[0] = 0.0;
            return;
        }
        for i in start..borne {
            let obj = ts[i as usize];
            self.tsm[1][0] += 1.0;
            let y = self.getobjy(obj);
            self.tsm[1][1] += y as f64;
            let yy = y * y;
            self.tsm[1][2] += yy as f64;
        }
        self.compute_multiregr_score_from_table_for_varimp(vi);
    }

    /// compute_one_tree_variable_importance_multiregr_separate (obj == -1 path).
    fn compute_one_tree_variable_importance(
        &mut self,
        tree: i64,
        ts: &mut Vec<i64>,
        weight: f64,
        attribute_importance: &mut [f64],
    ) {
        let length = self.n as i64;
        let mut stack: Vec<(i64, i64, i64)> = Vec::with_capacity(1024);
        stack.push((tree, 0, length - 1));
        while let Some((node, start, end)) = stack.pop() {
            let nodesize = end - start + 1;
            if self.left_succ[node as usize] == -1 || nodesize == 1 {
                continue;
            }
            let att = self.tested_att[node as usize] as usize;
            let thr = self.threshold[node as usize];
            let borne = separate_ls(att, thr, self.x, self.n, ts, start, end);
            let mut vi = [0.0f64; 1];
            self.get_vi_multiregr_separate(ts, start, end, borne, &mut vi);
            // attribute_position is identity here (attribute_vector re-sorted)
            attribute_importance[att] += weight * vi[0];
            if start < borne {
                stack.push((
                    node + self.left_succ[node as usize],
                    start,
                    borne - 1,
                ));
            }
            if borne <= end {
                stack.push((
                    node + self.right_succ[node as usize],
                    borne,
                    end,
                ));
            }
        }
    }

    /// compute_ltrees_variable_importance_multiregr_separate (obj == -1).
    fn compute_ltrees_variable_importance(&mut self, attribute_importance: &mut [f64]) {
        // ts_vector <- identity (C reuses current_learning_set for this)
        let mut ts: Vec<i64> = (0..self.n as i64).collect();
        for v in attribute_importance.iter_mut() {
            *v = 0.0;
        }
        let mut sum_weight = 0.0f64;
        for t in 0..self.p.n_trees {
            self.compute_one_tree_variable_importance(self.ltrees[t], &mut ts, 1.0, attribute_importance);
            sum_weight += 1.0;
        }
        // average_predictions_ltrees == 1
        for v in attribute_importance.iter_mut() {
            *v /= sum_weight;
        }
    }
}

/// R wrapper `.setMtry(K, numRegulators)`:
/// numeric K -> K; "sqrt" -> round(sqrt(p)); else p.
/// round-half-even never matters here (sqrt of an int is never exactly x.5).
pub fn set_mtry(k: &KSpec, num_regulators: usize) -> usize {
    match k {
        KSpec::Sqrt => ((num_regulators as f64).sqrt().round()) as usize,
        KSpec::All => num_regulators,
        KSpec::K(v) => *v,
    }
}

#[derive(Clone, Copy)]
pub enum KSpec {
    Sqrt,
    All,
    K(usize),
}

/// Free-standing separate_ls_vector_local (C original is a plain function
/// over globals; this form avoids the &self / &mut self.cls aliasing while
/// keeping identical semantics: two-pointer partition by val < threshold).
fn separate_ls(
    att: usize,
    threshold: f32,
    x: &[f32],
    n: usize,
    ls: &mut [i64],
    start: i64,
    end: i64,
) -> i64 {
    let val = |obj: i64| x[att * n + obj as usize];
    let mut start = start;
    let mut end = end;
    while start != end {
        while start != end && val(ls[start as usize]) < threshold {
            start += 1;
        }
        while start != end && !(val(ls[end as usize]) < threshold) {
            end -= 1;
        }
        if start != end {
            ls.swap(start as usize, end as usize);
            start += 1;
        }
    }
    if val(ls[start as usize]) < threshold {
        start + 1
    } else {
        start
    }
}
