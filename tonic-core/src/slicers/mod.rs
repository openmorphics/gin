// Public slicers API
// Re-exports helper functions for computing contiguous [start, end) event index pairs.
pub mod slicers;

pub use slicers::{
    slice_by_time,
    slice_by_count,
    slice_by_time_bins,
    slice_by_event_bins,
};