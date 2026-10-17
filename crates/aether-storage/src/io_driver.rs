//! Project Aether-Manifold: Physical 4KB Page Direct I/O Driver (`io_driver.rs`)
//!
//! Provides hardware-sympathetic direct I/O for 4096-byte `TensorGraphPage` blocks.
//! - On Linux (`target_os = "linux"`): Directly leverages `io_uring` with registered fixed
//!   buffers and `O_DIRECT | O_DSYNC` to achieve zero-deserialization kernel DMA.
//! - On non-Linux / test environments: Uses unbuffered random-access file I/O or pinned
//!   memory mappings with strictly 0 hot-path heap allocations.

use aether_core::error::AetherError;
use aether_core::page::TensorGraphPage;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

/// Asynchronous Direct Page I/O Completion.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct IoCompletion {
    /// User data identifier originally submitted with the request.
    pub user_data: u64,
    /// Completion result (e.g., number of bytes transferred or negative error code).
    pub result: i32,
}

#[cfg(target_os = "linux")]
/// Linux kernel `io_uring` direct I/O driver.
pub mod linux_uring {
    use super::*;
    use io_uring::{opcode, types, IoUring};
    use std::os::unix::fs::OpenOptionsExt;
    use std::os::unix::io::{AsRawFd, RawFd};

    /// Linux `io_uring` driver managing SQPOLL and fixed registered buffer arenas.
    pub struct UringDirectDriver {
        ring: IoUring,
        _fd: RawFd,
        _file: File,
    }

    impl UringDirectDriver {
        /// Open a block device or run file with `O_DIRECT | O_DSYNC` and pin page arena.
        pub fn new(
            path: &Path,
            ring_entries: u32,
            arena_pages: &[TensorGraphPage],
        ) -> Result<Self, AetherError> {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .custom_flags(libc::O_DIRECT | libc::O_DSYNC | libc::O_NOATIME)
                .open(path)
                .map_err(|e| AetherError::IoFault {
                    code: e.raw_os_error().unwrap_or(-1),
                })?;

            let fd = file.as_raw_fd();
            let ring = IoUring::builder()
                .setup_sqpoll(1000)
                .build(ring_entries)
                .map_err(|e| AetherError::IoFault {
                    code: e.raw_os_error().unwrap_or(-2),
                })?;

            // Pin 4096-byte aligned TensorGraphPage arena as fixed kernel buffers
            let iovecs: Vec<libc::iovec> = arena_pages
                .iter()
                .map(|page| libc::iovec {
                    iov_base: (page as *const TensorGraphPage) as *mut libc::c_void,
                    iov_len: core::mem::size_of::<TensorGraphPage>(),
                })
                .collect();

            unsafe {
                ring.submitter()
                    .register_buffers(&iovecs)
                    .map_err(|e| AetherError::IoFault {
                        code: e.raw_os_error().unwrap_or(-3),
                    })?;
                ring.submitter()
                    .register_files(&[fd])
                    .map_err(|e| AetherError::IoFault {
                        code: e.raw_os_error().unwrap_or(-4),
                    })?;
            }

            Ok(Self {
                ring,
                _fd: fd,
                _file: file,
            })
        }

        /// Submit a 4KB fixed-buffer NVMe read/write with zero heap allocations.
        #[inline]
        pub fn submit_fixed_page_io(
            &mut self,
            arena_slot: u16,
            disk_page_offset: u64,
            is_write: bool,
            user_data: u64,
        ) -> Result<(), AetherError> {
            let byte_offset = disk_page_offset * 4096;
            let ptr = (arena_slot as usize) as *mut u8;

            let sqe = if is_write {
                opcode::WriteFixed::new(types::Fixed(0), ptr, 4096, arena_slot)
                    .offset(byte_offset)
                    .build()
                    .user_data(user_data)
            } else {
                opcode::ReadFixed::new(types::Fixed(0), ptr, 4096, arena_slot)
                    .offset(byte_offset)
                    .build()
                    .user_data(user_data)
            };

            unsafe {
                self.ring.submission().push(&sqe).map_err(|_| {
                    AetherError::ScratchpadExhausted {
                        requested: 1,
                        available: 0,
                    }
                })?;
            }
            self.ring
                .submitter()
                .submit()
                .map_err(|e| AetherError::IoFault {
                    code: e.raw_os_error().unwrap_or(-5),
                })?;
            Ok(())
        }

        /// Reaps completions into caller-provided stack buffer with 0 heap allocations.
        #[inline]
        pub fn reap_completions(&mut self, out: &mut [IoCompletion]) -> usize {
            let mut count = 0;
            let mut cq = self.ring.completion();
            while count < out.len() {
                if let Some(cqe) = cq.next() {
                    if let Some(slot) = out.get_mut(count) {
                        *slot = IoCompletion {
                            user_data: cqe.user_data(),
                            result: cqe.result(),
                        };
                        count += 1;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            count
        }
    }
}

/// Cross-platform direct file driver for unbuffered 4KB block persistence.
pub struct FileDirectDriver {
    file: Option<File>,
}

impl FileDirectDriver {
    /// Creates a driver without an open file (operates in memory-only mode).
    pub fn memory_only() -> Self {
        Self { file: None }
    }

    /// Open or create a physical block run file for 4KB direct reads and writes.
    pub fn open(path: &Path) -> Result<Self, AetherError> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .map_err(|e| AetherError::IoFault {
                code: e.raw_os_error().unwrap_or(-1),
            })?;

        Ok(Self { file: Some(file) })
    }

    /// Read a 4096-byte page directly from disk into `target_page`.
    ///
    /// # Invariants
    /// - Strictly 0 heap allocations on the hot path (AGENTS.md Rule 3.1).
    pub fn read_page(
        &mut self,
        page_index: u64,
        target_page: &mut TensorGraphPage,
    ) -> Result<usize, AetherError> {
        let file = match self.file.as_mut() {
            Some(f) => f,
            None => return Ok(4096), // Memory-only simulation mode
        };

        let offset = page_index * 4096;
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| AetherError::IoFault {
                code: e.raw_os_error().unwrap_or(-2),
            })?;

        let bytes: &mut [u8; 4096] = target_page.as_aligned_bytes_mut();
        file.read_exact(bytes).map_err(|e| AetherError::IoFault {
            code: e.raw_os_error().unwrap_or(-3),
        })?;

        Ok(4096)
    }

    /// Write a 4096-byte page directly from `source_page` to disk.
    ///
    /// # Invariants
    /// - Strictly 0 heap allocations on the hot path (AGENTS.md Rule 3.1).
    pub fn write_page(
        &mut self,
        page_index: u64,
        source_page: &TensorGraphPage,
    ) -> Result<usize, AetherError> {
        let file = match self.file.as_mut() {
            Some(f) => f,
            None => return Ok(4096), // Memory-only simulation mode
        };

        let offset = page_index * 4096;
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| AetherError::IoFault {
                code: e.raw_os_error().unwrap_or(-4),
            })?;

        let bytes: &[u8; 4096] = source_page.as_aligned_bytes();
        file.write_all(bytes).map_err(|e| AetherError::IoFault {
            code: e.raw_os_error().unwrap_or(-5),
        })?;

        Ok(4096)
    }
}

impl Default for FileDirectDriver {
    fn default() -> Self {
        Self::memory_only()
    }
}
