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
use numpy::{ndarray, PyArray1, PyArray2, PyArray4, PyReadonlyArray1};
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

#[pyfunction(text_signature = "(x, y, t, p, sensor_w, sensor_h, n_polarities, cell_size, surface_size, time_window, tau, decay, /)")]
#[pyo3(name = "averaged_time_surface")]
/// Compute HATS averaged time surfaces per cell and polarity.
/// Returns a NumPy f32 array with shape (C, P, S, S) in C-order.
fn averaged_time_surface_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u16>,
    y: PyReadonlyArray1<'py, u16>,
    t: PyReadonlyArray1<'py, i64>,
    p: PyReadonlyArray1<'py, i8>,
    sensor_w: usize,
    sensor_h: usize,
    n_polarities: usize,
    cell_size: usize,
    surface_size: usize,
    time_window: i64,
    tau: f32,
    decay: &str,
) -> PyResult<Py<PyArray4<f32>>> {
    let events = build_events(x, y, t, p)?;
    let dk = match decay {
        "exp" | "EXP" | "Exp" => tonic_core::kernels::averaged_time_surface::DecayKind::Exp,
        "lin" | "LIN" | "Lin" => tonic_core::kernels::averaged_time_surface::DecayKind::Lin,
        _ => return Err(PyValueError::new_err("decay must be 'exp' or 'lin'")),
    };
    let wgrid = if cell_size == 0 { 0 } else { (sensor_w + cell_size - 1) / cell_size };
    let hgrid = if cell_size == 0 { 0 } else { (sensor_h + cell_size - 1) / cell_size };
    let n_cells = wgrid.saturating_mul(hgrid);
    let buf = tonic_core::kernels::averaged_time_surface::to_averaged_time_surface(
        &events,
        sensor_w,
        sensor_h,
        n_polarities,
        cell_size,
        surface_size,
        time_window as f32,
        tau,
        dk,
    ).map_err(PyValueError::new_err)?;
    let shape = (n_cells, n_polarities, surface_size, surface_size);
    let arr = ndarray::Array4::from_shape_vec(shape, buf)
        .map_err(|_| PyValueError::new_err("Failed to reshape averaged time surface into (C,P,S,S)."))?;
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

#[pyfunction(text_signature = "(x, y, t, p, sensor_w, sensor_h, delta, /)")]
#[pyo3(name = "refractory_period")]
/// Apply a per-pixel refractory period (keep at most one event within delta).
/// Returns a tuple of NumPy arrays (x: u16, y: u16, t: i64, p: i8).
fn refractory_period_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u16>,
    y: PyReadonlyArray1<'py, u16>,
    t: PyReadonlyArray1<'py, i64>,
    p: PyReadonlyArray1<'py, i8>,
    sensor_w: usize,
    sensor_h: usize,
    delta: i64,
) -> PyResult<(Py<PyArray1<u16>>, Py<PyArray1<u16>>, Py<PyArray1<i64>>, Py<PyArray1<i8>>)> {
    let events = build_events(x, y, t, p)?;
    let kept = tonic_core::kernels::refractory_period::refractory_period(&events, sensor_w, sensor_h, delta);

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

#[pyfunction(text_signature = "(x, y, t, p, coef, offset, /)")]
#[pyo3(name = "time_skew")]
/// Apply linear skew to timestamps: t' = t * coef + offset.
/// Returns a tuple of NumPy arrays (x: u16, y: u16, t: i64, p: i8).
fn time_skew_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u16>,
    y: PyReadonlyArray1<'py, u16>,
    t: PyReadonlyArray1<'py, i64>,
    p: PyReadonlyArray1<'py, i8>,
    coef: f64,
    offset: i64,
) -> PyResult<(Py<PyArray1<u16>>, Py<PyArray1<u16>>, Py<PyArray1<i64>>, Py<PyArray1<i8>>)> {
    let events = build_events(x, y, t, p)?;
    let out_evs = tonic_core::kernels::time_ops::time_skew(&events, coef, offset);

    let mut xs = Vec::with_capacity(out_evs.len());
    let mut ys = Vec::with_capacity(out_evs.len());
    let mut ts = Vec::with_capacity(out_evs.len());
    let mut ps = Vec::with_capacity(out_evs.len());
    for ev in out_evs {
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

#[pyfunction(text_signature = "(x, y, t, p, std, clip_negative=False, sort_timestamps=False, /)")]
#[pyo3(name = "time_jitter")]
/// Add Gaussian jitter to timestamps: t' = t + N(0, std^2).
/// Returns a tuple of NumPy arrays (x: u16, y: u16, t: i64, p: i8).
fn time_jitter_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u16>,
    y: PyReadonlyArray1<'py, u16>,
    t: PyReadonlyArray1<'py, i64>,
    p: PyReadonlyArray1<'py, i8>,
    std: f64,
    clip_negative: bool,
    sort_timestamps: bool,
) -> PyResult<(Py<PyArray1<u16>>, Py<PyArray1<u16>>, Py<PyArray1<i64>>, Py<PyArray1<i8>>)> {
    let events = build_events(x, y, t, p)?;
    let out_evs = tonic_core::kernels::time_ops::time_jitter(&events, std, clip_negative, sort_timestamps);

    let mut xs = Vec::with_capacity(out_evs.len());
    let mut ys = Vec::with_capacity(out_evs.len());
    let mut ts = Vec::with_capacity(out_evs.len());
    let mut ps = Vec::with_capacity(out_evs.len());
    for ev in out_evs {
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

#[pyfunction(text_signature = "(x, y, t, p, sensor_w, sensor_h, n, p_channels, /)")]
#[pyo3(name = "uniform_noise")]
/// Append uniform noise events across sensor and time range of input, then sort by t.
/// Returns a tuple of NumPy arrays (x: u16, y: u16, t: i64, p: i8).
fn uniform_noise_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u16>,
    y: PyReadonlyArray1<'py, u16>,
    t: PyReadonlyArray1<'py, i64>,
    p: PyReadonlyArray1<'py, i8>,
    sensor_w: usize,
    sensor_h: usize,
    n: usize,
    p_channels: usize,
) -> PyResult<(Py<PyArray1<u16>>, Py<PyArray1<u16>>, Py<PyArray1<i64>>, Py<PyArray1<i8>>)> {
    let events = build_events(x, y, t, p)?;
    let out_evs = tonic_core::kernels::time_ops::uniform_noise(&events, sensor_w, sensor_h, n, p_channels);

    let mut xs = Vec::with_capacity(out_evs.len());
    let mut ys = Vec::with_capacity(out_evs.len());
    let mut ts = Vec::with_capacity(out_evs.len());
    let mut ps = Vec::with_capacity(out_evs.len());
    for ev in out_evs {
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

#[pyfunction(text_signature = "(x, y, t, p, drop_probability, /)")]
#[pyo3(name = "drop_event")]
/// Randomly drop events with probability; returns structured arrays of kept events.
fn drop_event_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u16>,
    y: PyReadonlyArray1<'py, u16>,
    t: PyReadonlyArray1<'py, i64>,
    p: PyReadonlyArray1<'py, i8>,
    drop_probability: f64,
) -> PyResult<(Py<PyArray1<u16>>, Py<PyArray1<u16>>, Py<PyArray1<i64>>, Py<PyArray1<i8>>)> {
    let events = build_events(x, y, t, p)?;
    let kept = tonic_core::kernels::drop_ops::drop_event(&events, drop_probability);

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

#[pyfunction(text_signature = "(x, y, t, p, duration_ratio, /)")]
#[pyo3(name = "drop_by_time")]
/// Drop events within a randomly positioned time interval with the given ratio of total duration.
fn drop_by_time_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u16>,
    y: PyReadonlyArray1<'py, u16>,
    t: PyReadonlyArray1<'py, i64>,
    p: PyReadonlyArray1<'py, i8>,
    duration_ratio: f64,
) -> PyResult<(Py<PyArray1<u16>>, Py<PyArray1<u16>>, Py<PyArray1<i64>>, Py<PyArray1<i8>>)> {
    let events = build_events(x, y, t, p)?;
    let kept = tonic_core::kernels::drop_ops::drop_by_time(&events, duration_ratio);

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

#[pyfunction(text_signature = "(x, y, t, p, sensor_w, sensor_h, area_ratio, /)")]
#[pyo3(name = "drop_by_area")]
/// Drop events inside a randomly chosen axis-aligned box; box size is area_ratio of sensor.
fn drop_by_area_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u16>,
    y: PyReadonlyArray1<'py, u16>,
    t: PyReadonlyArray1<'py, i64>,
    p: PyReadonlyArray1<'py, i8>,
    sensor_w: usize,
    sensor_h: usize,
    area_ratio: f64,
) -> PyResult<(Py<PyArray1<u16>>, Py<PyArray1<u16>>, Py<PyArray1<i64>>, Py<PyArray1<i8>>)> {
    let events = build_events(x, y, t, p)?;
    let kept = tonic_core::kernels::drop_ops::drop_by_area(&events, sensor_w, sensor_h, area_ratio);

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

#[pyfunction(text_signature = "(x, y, t, p, sensor_w, sensor_h, var_x, var_y, sigma_xy, clip_outliers=False, /)")]
#[pyo3(name = "spatial_jitter")]
/// Add multivariate Gaussian jitter to (x,y) with covariance [[var_x, sigma_xy],[sigma_xy, var_y]].
/// If clip_outliers is True, events falling outside the sensor are dropped; otherwise they are clamped.
fn spatial_jitter_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u16>,
    y: PyReadonlyArray1<'py, u16>,
    t: PyReadonlyArray1<'py, i64>,
    p: PyReadonlyArray1<'py, i8>,
    sensor_w: usize,
    sensor_h: usize,
    var_x: f64,
    var_y: f64,
    sigma_xy: f64,
    clip_outliers: bool,
) -> PyResult<(Py<PyArray1<u16>>, Py<PyArray1<u16>>, Py<PyArray1<i64>>, Py<PyArray1<i8>>)> {
    let events = build_events(x, y, t, p)?;
    let kept = tonic_core::kernels::spatial::spatial_jitter(
        &events,
        sensor_w,
        sensor_h,
        var_x,
        var_y,
        sigma_xy,
        clip_outliers,
    );

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

#[pyfunction(text_signature = "(x, y, t, p, sensor_w, sensor_h, hot_pixel_frequency, /)")]
#[pyo3(name = "identify_hot_pixel")]
/// Identify hot pixels by frequency threshold over the recording duration.
/// Returns a NumPy uint16 array of shape (K, 2) with columns [x, y].
fn identify_hot_pixel_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u16>,
    y: PyReadonlyArray1<'py, u16>,
    t: PyReadonlyArray1<'py, i64>,
    p: PyReadonlyArray1<'py, i8>,
    sensor_w: usize,
    sensor_h: usize,
    hot_pixel_frequency: f64,
) -> PyResult<Py<PyArray2<u16>>> {
    let events = build_events(x, y, t, p)?;
    let coords = tonic_core::kernels::drop_ops::identify_hot_pixel(
        &events,
        sensor_w,
        sensor_h,
        hot_pixel_frequency,
    );
    let k = coords.len();
    let mut arr = ndarray::Array2::<u16>::zeros((k, 2));
    for (i, (cx, cy)) in coords.iter().enumerate() {
        arr[(i, 0)] = *cx;
        arr[(i, 1)] = *cy;
    }
    Ok(PyArray2::from_owned_array(py, arr).to_owned())
}

#[pyfunction(text_signature = "(x, y, t, p, sensor_w, sensor_h, x_coords, y_coords, /)")]
#[pyo3(name = "drop_pixel")]
/// Drop events whose (x,y) matches any provided coordinate; preserves order.
fn drop_pixel_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u16>,
    y: PyReadonlyArray1<'py, u16>,
    t: PyReadonlyArray1<'py, i64>,
    p: PyReadonlyArray1<'py, i8>,
    sensor_w: usize,
    sensor_h: usize,
    x_coords: PyReadonlyArray1<'py, u16>,
    y_coords: PyReadonlyArray1<'py, u16>,
) -> PyResult<(Py<PyArray1<u16>>, Py<PyArray1<u16>>, Py<PyArray1<i64>>, Py<PyArray1<i8>>)> {
    let events = build_events(x, y, t, p)?;
    let xs = x_coords.as_slice()?;
    let ys = y_coords.as_slice()?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x_coords and y_coords must have the same length."));
    }
    let coords: Vec<(u16, u16)> = xs.iter().zip(ys.iter()).map(|(a, b)| (*a, *b)).collect();
    let kept = tonic_core::kernels::drop_ops::drop_pixel(&events, sensor_w, sensor_h, &coords);

    let mut xo = Vec::with_capacity(kept.len());
    let mut yo = Vec::with_capacity(kept.len());
    let mut to = Vec::with_capacity(kept.len());
    let mut po = Vec::with_capacity(kept.len());
    for ev in kept {
        xo.push(ev.x);
        yo.push(ev.y);
        to.push(ev.t_ns);
        po.push(ev.p);
    }

    let x_arr = PyArray1::from_vec(py, xo).to_owned();
    let y_arr = PyArray1::from_vec(py, yo).to_owned();
    let t_arr = PyArray1::from_vec(py, to).to_owned();
    let p_arr = PyArray1::from_vec(py, po).to_owned();
    Ok((x_arr, y_arr, t_arr, p_arr))
}

#[pyfunction(text_signature = "(x, y, t, p, sensor_w, sensor_h, target_w, target_h, /)")]
#[pyo3(name = "center_crop")]
fn center_crop_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u16>,
    y: PyReadonlyArray1<'py, u16>,
    t: PyReadonlyArray1<'py, i64>,
    p: PyReadonlyArray1<'py, i8>,
    sensor_w: usize,
    sensor_h: usize,
    target_w: usize,
    target_h: usize,
) -> PyResult<(Py<PyArray1<u16>>, Py<PyArray1<u16>>, Py<PyArray1<i64>>, Py<PyArray1<i8>>)> {
    let events = build_events(x, y, t, p)?;
    let kept = tonic_core::kernels::crop::center_crop(&events, sensor_w, sensor_h, target_w, target_h);

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

#[pyfunction(text_signature = "(x, y, t, p, sensor_w, sensor_h, x0, y0, target_w, target_h, /)")]
#[pyo3(name = "crop_box")]
fn crop_box_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u16>,
    y: PyReadonlyArray1<'py, u16>,
    t: PyReadonlyArray1<'py, i64>,
    p: PyReadonlyArray1<'py, i8>,
    sensor_w: usize,
    sensor_h: usize,
    x0: usize,
    y0: usize,
    target_w: usize,
    target_h: usize,
) -> PyResult<(Py<PyArray1<u16>>, Py<PyArray1<u16>>, Py<PyArray1<i64>>, Py<PyArray1<i8>>)> {
    let events = build_events(x, y, t, p)?;
    let kept = tonic_core::kernels::crop::crop_box(&events, sensor_w, sensor_h, x0, y0, target_w, target_h);

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

#[pyfunction(text_signature = "(x, y, t, p, sensor_w, sensor_h, /)")]
#[pyo3(name = "flip_lr")]
fn flip_lr_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u16>,
    y: PyReadonlyArray1<'py, u16>,
    t: PyReadonlyArray1<'py, i64>,
    p: PyReadonlyArray1<'py, i8>,
    sensor_w: usize,
    sensor_h: usize,
) -> PyResult<(Py<PyArray1<u16>>, Py<PyArray1<u16>>, Py<PyArray1<i64>>, Py<PyArray1<i8>>)> {
    let events = build_events(x, y, t, p)?;
    let kept = tonic_core::kernels::flip::flip_lr(&events, sensor_w, sensor_h);

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

#[pyfunction(text_signature = "(x, y, t, p, sensor_w, sensor_h, /)")]
#[pyo3(name = "flip_ud")]
fn flip_ud_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u16>,
    y: PyReadonlyArray1<'py, u16>,
    t: PyReadonlyArray1<'py, i64>,
    p: PyReadonlyArray1<'py, i8>,
    sensor_w: usize,
    sensor_h: usize,
) -> PyResult<(Py<PyArray1<u16>>, Py<PyArray1<u16>>, Py<PyArray1<i64>>, Py<PyArray1<i8>>)> {
    let events = build_events(x, y, t, p)?;
    let kept = tonic_core::kernels::flip::flip_ud(&events, sensor_w, sensor_h);

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

#[pyfunction(text_signature = "(x, y, t, p, /)")]
#[pyo3(name = "flip_polarity")]
fn flip_polarity_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u16>,
    y: PyReadonlyArray1<'py, u16>,
    t: PyReadonlyArray1<'py, i64>,
    p: PyReadonlyArray1<'py, i8>,
) -> PyResult<(Py<PyArray1<u16>>, Py<PyArray1<u16>>, Py<PyArray1<i64>>, Py<PyArray1<i8>>)> {
    let events = build_events(x, y, t, p)?;
    let kept = tonic_core::kernels::flip::flip_polarity(&events);

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

#[pyfunction(text_signature = "(x, y, t, p, /)")]
#[pyo3(name = "time_reverse")]
/// Reflect timestamps over max(t) and reverse order; x,y,p unchanged.
/// Returns a tuple of NumPy arrays (x: u16, y: u16, t: i64, p: i8).
fn time_reverse_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u16>,
    y: PyReadonlyArray1<'py, u16>,
    t: PyReadonlyArray1<'py, i64>,
    p: PyReadonlyArray1<'py, i8>,
) -> PyResult<(Py<PyArray1<u16>>, Py<PyArray1<u16>>, Py<PyArray1<i64>>, Py<PyArray1<i8>>)> {
    let events = build_events(x, y, t, p)?;
    let kept = tonic_core::kernels::time_ops::time_reverse(&events);

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

#[pyfunction(text_signature = "(x, y, t, p, sensor_size, downsample_factor, noise_threshold, differentiate, /)")]
#[pyo3(name = "event_downsample")]
/// EventDownsampling fast path: integrator/differentiator over spatial neighborhoods.
/// Returns a tuple of NumPy arrays (x: u16, y: u16, t: i64, p: i8).
fn event_downsample_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u16>,
    y: PyReadonlyArray1<'py, u16>,
    t: PyReadonlyArray1<'py, i64>,
    p: PyReadonlyArray1<'py, i8>,
    sensor_size: (u16, u16),
    downsample_factor: u8,
    noise_threshold: u32,
    differentiate: bool,
) -> PyResult<(Py<PyArray1<u16>>, Py<PyArray1<u16>>, Py<PyArray1<i64>>, Py<PyArray1<i8>>)> {
    let events = build_events(x, y, t, p)?;
    let kept = tonic_core::kernels::downsample::event_downsample_integrator(
        &events,
        sensor_size,
        downsample_factor,
        noise_threshold,
        differentiate,
    );

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
    m.add_function(wrap_pyfunction!(averaged_time_surface_py, m)?)?;
    m.add_function(wrap_pyfunction!(denoise_py, m)?)?;
    m.add_function(wrap_pyfunction!(decimate_py, m)?)?;
    m.add_function(wrap_pyfunction!(refractory_period_py, m)?)?;
    m.add_function(wrap_pyfunction!(time_skew_py, m)?)?;
    m.add_function(wrap_pyfunction!(time_jitter_py, m)?)?;
    m.add_function(wrap_pyfunction!(uniform_noise_py, m)?)?;
    m.add_function(wrap_pyfunction!(drop_event_py, m)?)?;
    m.add_function(wrap_pyfunction!(drop_by_time_py, m)?)?;
    m.add_function(wrap_pyfunction!(drop_by_area_py, m)?)?;
    m.add_function(wrap_pyfunction!(spatial_jitter_py, m)?)?;
    m.add_function(wrap_pyfunction!(identify_hot_pixel_py, m)?)?;
    m.add_function(wrap_pyfunction!(drop_pixel_py, m)?)?;
    m.add_function(wrap_pyfunction!(center_crop_py, m)?)?;
    m.add_function(wrap_pyfunction!(crop_box_py, m)?)?;
    m.add_function(wrap_pyfunction!(flip_lr_py, m)?)?;
    m.add_function(wrap_pyfunction!(flip_ud_py, m)?)?;
    m.add_function(wrap_pyfunction!(flip_polarity_py, m)?)?;
    m.add_function(wrap_pyfunction!(time_reverse_py, m)?)?;
    m.add_function(wrap_pyfunction!(event_downsample_py, m)?)?;
    Ok(())
}