use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Bounded local write-ahead spool.
///
/// Every record is appended as a `u32` big-endian length prefix followed by the
/// raw bytes and `sync_data` before it may be sent to the trainer. The spool is
/// replayed on restart, which is why a broker ack is only produced after the
/// trainer confirms durable acceptance.
pub struct Wal {
    path: PathBuf,
    file: File,
    max_bytes: usize,
    pending: Vec<Vec<u8>>,
    bytes: usize,
}

#[derive(Debug)]
pub struct WalFull;

impl Wal {
    pub fn open(path: &Path, max_bytes: usize) -> Result<Self, String> {
        if max_bytes == 0 {
            return Err("wal max_bytes must be positive".to_string());
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("cannot create WAL directory: {error}"))?;
        }
        let pending = replay(path)?;
        let bytes = encoded_size(&pending);
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|error| format!("cannot open WAL {}: {error}", path.display()))?;
        Ok(Self {
            path: path.to_path_buf(),
            file,
            max_bytes,
            pending,
            bytes,
        })
    }

    pub fn pending(&self) -> &[Vec<u8>] {
        &self.pending
    }

    pub fn len(&self) -> usize {
        self.pending.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn append(&mut self, record: &[u8]) -> Result<(), WalFull> {
        let frame = encoded_size(std::slice::from_ref(&record.to_vec()));
        if self.bytes + frame > self.max_bytes {
            return Err(WalFull);
        }
        if self
            .file
            .write_all(&(record.len() as u32).to_be_bytes())
            .is_err()
        {
            return Err(WalFull);
        }
        if self.file.write_all(record).is_err() {
            return Err(WalFull);
        }
        if let Err(error) = self.file.sync_data() {
            eprintln!("jev-mq-bridge: WAL sync failed: {error}");
            return Err(WalFull);
        }
        self.pending.push(record.to_vec());
        self.bytes += frame;
        Ok(())
    }

    /// Remove a frame after the trainer acknowledged durable acceptance and
    /// compact the file when half of the budget is free again.
    pub fn acknowledge(&mut self, record: &[u8]) {
        if let Some(index) = self
            .pending
            .iter()
            .position(|item| item.as_slice() == record)
        {
            let removed = encoded_size(std::slice::from_ref(&self.pending.remove(index)));
            self.bytes = self.bytes.saturating_sub(removed);
            if self.bytes * 2 < self.max_bytes {
                if let Err(error) = self.rewrite() {
                    eprintln!("jev-mq-bridge: WAL compaction failed: {error}");
                }
            }
        }
    }

    fn rewrite(&mut self) -> Result<(), String> {
        let temporary = self.path.with_extension("rewrite");
        {
            let mut file = File::create(&temporary)
                .map_err(|error| format!("cannot create WAL rewrite file: {error}"))?;
            for record in &self.pending {
                file.write_all(&(record.len() as u32).to_be_bytes())
                    .map_err(|error| error.to_string())?;
                file.write_all(record).map_err(|error| error.to_string())?;
            }
            file.sync_data().map_err(|error| error.to_string())?;
        }
        std::fs::rename(&temporary, &self.path)
            .map_err(|error| format!("cannot replace WAL: {error}"))?;
        self.file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|error| error.to_string())?;
        self.bytes = encoded_size(&self.pending);
        Ok(())
    }
}

fn encoded_size(records: &[Vec<u8>]) -> usize {
    records.iter().map(|record| record.len() + 4).sum()
}

pub fn replay(path: &Path) -> Result<Vec<Vec<u8>>, String> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let mut file = File::open(path).map_err(|error| format!("cannot open WAL: {error}"))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|error| format!("cannot read WAL: {error}"))?;
    let mut records = Vec::new();
    let mut cursor = 0usize;
    while cursor + 4 <= bytes.len() {
        let length = u32::from_be_bytes(
            bytes[cursor..cursor + 4]
                .try_into()
                .expect("length prefix is four bytes"),
        ) as usize;
        cursor += 4;
        if cursor + length > bytes.len() {
            // A torn tail is dropped: the broker will redeliver unacked records.
            eprintln!("jev-mq-bridge: dropping torn WAL tail at offset {cursor}");
            break;
        }
        records.push(bytes[cursor..cursor + length].to_vec());
        cursor += length;
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_ack_replay_and_bound() {
        let directory = std::env::temp_dir().join(format!("jev-wal-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("wal.bin");
        let mut wal = Wal::open(&path, 64).unwrap();
        wal.append(b"first").unwrap();
        wal.append(b"second").unwrap();
        assert!(wal.append(b"x".repeat(64).as_slice()).is_err());
        wal.acknowledge(b"first");
        // Acked frames are compacted away; only pending frames remain.
        let replayed = replay(&path).unwrap();
        assert_eq!(replayed, vec![b"second".to_vec()]);
        assert!(wal.bytes() <= 64);
        let _ = std::fs::remove_dir_all(&directory);
    }
}
