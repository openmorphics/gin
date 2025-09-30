//! tonic_python: PyO3 bindings for tonic-rs core kernels.
//!
//! Exposes NumPy-compatible functions:
//! - voxel_grid -> f64 array shape (T,1,H,W)
//! - frames (time_window, event_count, n_time_bins, n_event_bins) -> i16 array shape (T,P,H,W)
//! - time_surface -> f64 array shape (T,P,H,W)
//! - denoise/decimate -> tuple of (x: u16, y: u16, t: i64, p: i8)
//!
//! Inputs: x(u16), y(u16), t(i64), p(i8) as 1D NumPy arrays with equal lengths.
//! Event order is preserved in filtered outputs.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use numpy::{ndarray, PyArray1, PyArray4, PyReadonlyArray1};
use tonic_core::Event;

fn build_events<'py>(
    x: PyReadonlyArray1<'py, u16>,
    y: PyReadonlyArray1<'py, u16>,
    t: PyReadonlyArray1<'py, i64>,
    p: PyReadonlyArray1<'py, i8>,
) -> PyResult<Vec<Event>> {
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;
    let ts = t.as_slice()?;
    let ps = p.as_slice()?;

    let n = xs.len();
    if ys.len() != n || ts.len() != n || ps.len() != n {
        return Err(PyValueError::new_err(
            "Input arrays x, y, t, p must have equal length.",
        ));
    }

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(Event { t_ns: ts[i], x: xs[i], y: ys[i], p: ps[i], stream_id: 0 });
    }
    Ok(out)
}

#[pyfunction(text_signature = "(x, y, t, p, sensor_w, sensor_h, n_time_bins, /)")]
#[pyo3(name = "voxel_grid")]
/// Build a voxel grid with bilinear interpolation over time.
/// Returns a NumPy f64 array with shape (T, 1, H, W) in C-order.
fn voxel_grid_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u16>,
    y: PyReadonlyArray1<'py, u16>,
    t: PyReadonlyArray1<'py, i64>,
    p: PyReadonlyArray1<'py, i8>,
    sensor_w: usize,
    sensor_h: usize,
    n_time_bins: usize,
) -> PyResult<Py<PyArray4<f64>>> {
    let events = build_events(x, y, t, p)?;
    let buf = tonic_core::kernels::voxel_grid::to_voxel_grid(
        &events,
        sensor_w,
        sensor_h,
        n_time_bins,
    );
    let shape = (n_time_bins, 1, sensor_h, sensor_w);
    let arr = ndarray::Array4::from_shape_vec(shape, buf)
        .map_err(|_| PyValueError::new_err("Failed to reshape voxel grid into (T,1,H,W)."))?;
    Ok(PyArray4::from_owned_array(py, arr).to_owned())
}

#[pyfunction(text_signature = "(x, y, t, p, sensor_w, sensor_h, n_polarities, time_window, overlap, include_incomplete, start_time=None, end_time=None, /)")]
#[pyo3(name = "frames_time_window")]
/// Frame conversion using fixed time windows.
/// Returns a NumPy i16 array with shape (T, P, H, W) in C-order.
fn frames_time_window_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u16>,
    y: PyReadonlyArray1<'py, u16>,
    t: PyReadonlyArray1<'py, i64>,
    p: PyReadonlyArray1<'py, i8>,
    sensor_w: usize,
    sensor_h: usize,
    n_polarities: usize,
    time_window: i64,
    overlap: i64,
    include_incomplete: bool,
    start_time: Option<i64>,
    end_time: Option<i64>,
) -> PyResult<Py<PyArray4<i16>>> {
    let events = build_events(x, y, t, p)?;
    let (buf, t_slices) = tonic_core::kernels::frame::to_frame_time_window(
        &events,
        sensor_w,
        sensor_h,
        n_polarities,
        time_window,
        overlap,
        include_incomplete,
        start_time,
        end_time,
    )
    .map_err(PyValueError::new_err)?;
    let shape = (t_slices, n_polarities, sensor_h, sensor_w);
    let arr = ndarray::Array4::from_shape_vec(shape, buf)
        .map_err(|_| PyValueError::new_err("Failed to reshape frames into (T,P,H,W)."))?;
    Ok(PyArray4::from_owned_array(py, arr).to_owned())
}

#[pyfunction(text_signature = "(x, y, t, p, sensor_w, sensor_h, n_polarities, event_count, overlap, include_incomplete, /)")]
#[pyo3(name = "frames_event_count")]
/// Frame conversion using fixed event counts per slice.
/// Returns a NumPy i16 array with shape (T, P, H, W) in C-order.
fn frames_event_count_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u16>,
    y: PyReadonlyArray1<'py, u16>,
    t: PyReadonlyArray1<'py, i64>,
    p: PyReadonlyArray1<'py, i8>,
    sensor_w: usize,
    sensor_h: usize,
    n_polarities: usize,
    event_count: usize,
    overlap: usize,
    include_incomplete: bool,
) -> PyResult<Py<PyArray4<i16>>> {
    let events = build_events(x, y, t, p)?;
    let (buf, t_slices) = tonic_core::kernels::frame::to_frame_event_count(
        &events,
        sensor_w,
        sensor_h,
        n_polarities,
        event_count,
        overlap,
        include_incomplete,
    )
    .map_err(PyValueError::new_err)?;
    let shape = (t_slices, n_polarities, sensor_h, sensor_w);
    let arr = ndarray::Array4::from_shape_vec(shape, buf)
        .map_err(|_| PyValueError::new_err("Failed to reshape frames into (T,P,H,W)."))?;
    Ok(PyArray4::from_owned_array(py, arr).to_owned())
}

#[pyfunction(text_signature = "(x, y, t, p, sensor_w, sensor_h, n_polarities, n_time_bins, overlap_frac, /)")]
#[pyo3(name = "frames_n_time_bins")]
/// Frame conversion with a fixed number of time bins.
/// Returns a NumPy i16 array with shape (T, P, H, W) in C-order.
fn frames_n_time_bins_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u16>,
    y: PyReadonlyArray1<'py, u16>,
    t: PyReadonlyArray1<'py, i64>,
    p: PyReadonlyArray1<'py, i8>,
    sensor_w: usize,
    sensor_h: usize,
    n_polarities: usize,
    n_time_bins: usize,
    overlap_frac: f64,
) -> PyResult<Py<PyArray4<i16>>> {
    let events = build_events(x, y, t, p)?;
    let (buf, t_slices) = tonic_core::kernels::frame::to_frame_n_time_bins(
        &events,
        sensor_w,
        sensor_h,
        n_polarities,
        n_time_bins,
        overlap_frac,
    )
    .map_err(PyValueError::new_err)?;
    let shape = (t_slices, n_polarities, sensor_h, sensor_w);
    let arr = ndarray::Array4::from_shape_vec(shape, buf)
        .map_err(|_| PyValueError::new_err("Failed to reshape frames into (T,P,H,W)."))?;
    Ok(PyArray4::from_owned_array(py, arr).to_owned())
}

#[pyfunction(text_signature = "(x, y, t, p, sensor_w, sensor_h, n_polarities, n_event_bins, overlap_frac, /)")]
#[pyo3(name = "frames_n_event_bins")]
/// Frame conversion with a fixed number of event bins.
/// Returns a NumPy i16 array with shape (T, P, H, W) in C-order.
fn frames_n_event_bins_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u16>,
    y: PyReadonlyArray1<'py, u16>,
    t: PyReadonlyArray1<'py, i64>,
    p: PyReadonlyArray1<'py, i8>,
    sensor_w: usize,
    sensor_h: usize,
    n_polarities: usize,
    n_event_bins: usize,
    overlap_frac: f64,
) -> PyResult<Py<PyArray4<i16>>> {
    let events = build_events(x, y, t, p)?;
    let (buf, t_slices) = tonic_core::kernels::frame::to_frame_n_event_bins(
        &events,
        sensor_w,
        sensor_h,
        n_polarities,
        n_event_bins,
        overlap_frac,
    )
    .map_err(PyValueError::new_err)?;
    let shape = (t_slices, n_polarities, sensor_h, sensor_w);
    let arr = ndarray::Array4::from_shape_vec(shape, buf)
        .map_err(|_| PyValueError::new_err("Failed to reshape frames into (T,P,H,W)."))?;
    Ok(PyArray4::from_owned_array(py, arr).to_owned())
}

#[pyfunction(text_signature = "(x, y, t, p, sensor_w, sensor_h, n_polarities, dt, tau, overlap, include_incomplete, start_time=None, end_time=None, /)")]
#[pyo3(name = "time_surface")]
/// Compute time surfaces over fixed delta_t windows.
/// Returns a NumPy f64 array with shape (T, P, H, W) in C-order.
fn time_surface_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u16>,
    y: PyReadonlyArray1<'py, u16>,
    t: PyReadonlyArray1<'py, i64>,
    p: PyReadonlyArray1<'py, i8>,
    sensor_w: usize,
    sensor_h: usize,
    n_polarities: usize,
    dt: i64,
    tau: f64,
    overlap: i64,
    include_incomplete: bool,
    start_time: Option<i64>,
    end_time: Option<i64>,
) -> PyResult<Py<PyArray4<f64>>> {
    let events = build_events(x, y, t, p)?;
    let (buf, t_slices) = tonic_core::kernels::time_surface::to_time_surface(
        &events,
        sensor_w,
        sensor_h,
        n_polarities,
        dt,
        tau,
        overlap,
        include_incomplete,
        start_time,
        end_time,
    )
    .map_err(PyValueError::new_err)?;
    let shape = (t_slices, n_polarities, sensor_h, sensor_w);
    let arr = ndarray::Array4::from_shape_vec(shape, buf)
        .map_err(|_| PyValueError::new_err("Failed to reshape time surfaces into (T,P,H,W)."))?;
    Ok(PyArray4::from_owned_array(py, arr).to_owned())
}

#[pyfunction(text_signature = "(x, y, t, p, sensor_w, sensor_h, filter_time, /)")]
#[pyo3(name = "denoise")]
/// Denoise the event stream using a temporal 4-neighborhood filter.
/// Returns a tuple of NumPy arrays (x: u16, y: u16, t: i64, p: i8).
fn denoise_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u16>,
    y: PyReadonlyArray1<'py, u16>,
    t: PyReadonlyArray1<'py, i64>,
    p: PyReadonlyArray1<'py, i8>,
    sensor_w: usize,
    sensor_h: usize,
    filter_time: i64,
) -> PyResult<(Py<PyArray1<u16>>, Py<PyArray1<u16>>, Py<PyArray1<i64>>, Py<PyArray1<i8>>)> {
    let events = build_events(x, y, t, p)?;
    let kept = tonic_core::kernels::denoise::denoise(&events, sensor_w, sensor_h, filter_time);

    let mut xs = Vec::with_capacity(kept.len());
    let mut ys = Vec::with_capacity(kept.len());
    let mut ts = Vec::with_capacity(kept.len());
    let mut ps = Vec::with_capacity(kept.len());
    for ev in kept {
        xs.push(ev.x);
        ys.push(ev.y);
        ts.push(ev.t_ns);
        ps.push(ev.p);
    }

    let x_arr = PyArray1::from_vec(py, xs).to_owned();
    let y_arr = PyArray1::from_vec(py, ys).to_owned();
    let t_arr = PyArray1::from_vec(py, ts).to_owned();
    let p_arr = PyArray1::from_vec(py, ps).to_owned();
    Ok((x_arr, y_arr, t_arr, p_arr))
}

#[pyfunction(text_signature = "(x, y, t, p, sensor_w, sensor_h, n, /)")]
#[pyo3(name = "decimate")]
/// Decimate the event stream by keeping every n-th event per pixel.
/// Returns a tuple of NumPy arrays (x: u16, y: u16, t: i64, p: i8).
fn decimate_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u16>,
    y: PyReadonlyArray1<'py, u16>,
    t: PyReadonlyArray1<'py, i64>,
    p: PyReadonlyArray1<'py, i8>,
    sensor_w: usize,
    sensor_h: usize,
    n: usize,
) -> PyResult<(Py<PyArray1<u16>>, Py<PyArray1<u16>>, Py<PyArray1<i64>>, Py<PyArray1<i8>>)> {
    if n == 0 {
        return Err(PyValueError::new_err("n must be an integer greater than zero."));
    }
    let events = build_events(x, y, t, p)?;
    let kept = tonic_core::kernels::decimate::decimate(&events, sensor_w, sensor_h, n);

    let mut xs = Vec::with_capacity(kept.len());
    let mut ys = Vec::with_capacity(kept.len());
    let mut ts = Vec::with_capacity(kept.len());
    let mut ps = Vec::with_capacity(kept.len());
    for ev in kept {
        xs.push(ev.x);
        ys.push(ev.y);
        ts.push(ev.t_ns);
        ps.push(ev.p);
    }

    let x_arr = PyArray1::from_vec(py, xs).to_owned();
    let y_arr = PyArray1::from_vec(py, ys).to_owned();
    let t_arr = PyArray1::from_vec(py, ts).to_owned();
    let p_arr = PyArray1::from_vec(py, ps).to_owned();
    Ok((x_arr, y_arr, t_arr, p_arr))
}

#[pymodule]
fn tonic_python(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add("__doc__", "Python bindings for tonic-rs core kernels.")?;
    m.add_function(wrap_pyfunction!(voxel_grid_py, m)?)?;
    m.add_function(wrap_pyfunction!(frames_time_window_py, m)?)?;
    m.add_function(wrap_pyfunction!(frames_event_count_py, m)?)?;
    m.add_function(wrap_pyfunction!(frames_n_time_bins_py, m)?)?;
    m.add_function(wrap_pyfunction!(frames_n_event_bins_py, m)?)?;
    m.add_function(wrap_pyfunction!(time_surface_py, m)?)?;
    m.add_function(wrap_pyfunction!(denoise_py, m)?)?;
    m.add_function(wrap_pyfunction!(decimate_py, m)?)?;
    Ok(())
}