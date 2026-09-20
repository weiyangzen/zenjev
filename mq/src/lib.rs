//! ZenJev external message-queue bridge library.
//!
//! The binary (`jev-mq-bridge`) is a thin wrapper around this library so the
//! protocol and validation contract can be exercised by integration tests and
//! by the Python trainer-side client through the same Unix-socket framing.

pub mod canonical;
pub mod config;
pub mod dedup;
pub mod envelope;
pub mod metrics;
pub mod sink;
pub mod source;
pub mod wal;
