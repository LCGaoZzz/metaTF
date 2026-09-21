//! metatf_rust — aggregate extension for the metatf package.
//!
//! Wraps the three campaign-winner crates as SUBMODULES of one cdylib so a
//! single wheel (metatf-rust) provides all accelerators:
//!   metatf_rust.rust_pcor_v2.partial_corr_spearman
//!   metatf_rust.rust_puic.puic / puic_raw
//!   metatf_rust.rust_genie3.genie3_targets / mt_first_unifs / r_sum
//! The inner crates are the verbatim round-3/round-4 winners (the only edit
//! is a `pub` on their #[pymodule] fns so the facade can call them); this
//! facade contains no numeric code of its own.

use pyo3::prelude::*;
use pyo3::types::PyModule;

fn attach_submodule(
    m: &Bound<'_, PyModule>,
    name: &str,
    init: fn(&Bound<'_, PyModule>) -> PyResult<()>,
) -> PyResult<()> {
    let py = m.py();
    let sub = PyModule::new_bound(py, name)?;
    init(&sub)?;
    m.add_submodule(&sub)
}

#[pymodule]
fn metatf_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    attach_submodule(m, "rust_pcor_v2", rust_pcor_v2::rust_pcor_v2)?;
    attach_submodule(m, "rust_puic", rust_puic::rust_puic)?;
    attach_submodule(m, "rust_genie3", rust_genie3::rust_genie3)?;
    Ok(())
}
