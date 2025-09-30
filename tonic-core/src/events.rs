#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Event {
    /// Timestamp in nanoseconds
    pub t_ns: i64,
    /// X-coordinate (column)
    pub x: u16,
    /// Y-coordinate (row)
    pub y: u16,
    /// Polarity: expected values 0/1 or -1/1. Kernel maps 0 -> -1.0.
    pub p: i8,
    /// Stream identifier for multi-sensor or sharded streams
    pub stream_id: u16,
}
