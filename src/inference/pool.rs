//! Session pool for ONNX inference triplets.
//!
//! Generic [`Pool<T>`] backed by an async-channel. The only production
//! instantiation is [`SessionPool`] = `Pool<SessionTriplet>`.

use ort::session::Session;
use std::ops::{Deref, DerefMut};

/// A set of ONNX sessions for one inference pipeline (encoder + decoder + joiner).
///
/// Moved out of the pool on checkout and returned on checkin.
/// Each triplet is independent and can run inference concurrently with others.
pub struct SessionTriplet {
    pub(crate) encoder: Session,
    pub(crate) decoder: Session,
    pub(crate) joiner: Session,
}

/// Errors returned by [`Pool::checkout`].
#[derive(Debug)]
pub enum PoolError {
    /// The pool was closed (graceful shutdown). All current and future
    /// waiters resolve to this variant; the caller should respond with a
    /// 503 / `pool_closed` to the client.
    Closed,
}

impl std::fmt::Display for PoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PoolError::Closed => write!(f, "session pool is closed"),
        }
    }
}

impl std::error::Error for PoolError {}

/// Pool of pre-loaded items of type `T` backed by an MPMC `async-channel`.
///
/// `SessionPool = Pool<SessionTriplet>` is the only public instantiation
/// outside this module. Generic `T` exists so the pool semantics can be
/// unit-tested without ONNX models.
///
/// Checkout = `recv` from the channel, checkin = `send` back via the
/// [`PoolGuard`] returned by [`checkout`](Self::checkout). The pool size acts
/// as the concurrency limit — no separate semaphore needed. FIFO ordering is
/// intrinsic to the underlying channel, and `close()` flips all current and
/// future waiters into [`PoolError::Closed`] so graceful shutdown can drain
/// without panicking.
pub struct Pool<T> {
    sender: async_channel::Sender<T>,
    receiver: async_channel::Receiver<T>,
    total: usize,
}

/// Public alias for the production pool: holds [`SessionTriplet`] instances.
pub type SessionPool = Pool<SessionTriplet>;

impl<T> Pool<T> {
    /// Create a pool pre-filled with the given items.
    pub fn new(items: Vec<T>) -> Self {
        let total = items.len();
        // Bounded channel with capacity == total: send is always immediate
        // (try_send never returns Full while we own the only sender for
        // checked-out items), and the channel's internal queue holds the
        // available pool inventory.
        let (sender, receiver) = async_channel::bounded(total.max(1));
        for item in items {
            sender
                .try_send(item)
                .expect("channel capacity matches item count");
        }
        Self {
            sender,
            receiver,
            total,
        }
    }

    /// Checkout an item from the pool. Awaits FIFO if none available.
    ///
    /// Returns [`PoolError::Closed`] if the pool was shut down via
    /// [`close`](Self::close) before an item became available.
    pub async fn checkout(&self) -> Result<PoolGuard<'_, T>, PoolError> {
        match self.receiver.recv().await {
            Ok(item) => Ok(PoolGuard {
                pool: self,
                item: Some(item),
            }),
            Err(_) => Err(PoolError::Closed),
        }
    }

    /// Checkout an item from the pool synchronously (blocks until one is
    /// available).  This is the FFI-friendly counterpart to [`checkout`](Self::checkout).
    ///
    /// Returns [`PoolError::Closed`] if the pool was shut down.
    pub fn checkout_blocking(&self) -> Result<PoolGuard<'_, T>, PoolError> {
        match self.receiver.recv_blocking() {
            Ok(item) => Ok(PoolGuard {
                pool: self,
                item: Some(item),
            }),
            Err(_) => Err(PoolError::Closed),
        }
    }

    /// Close the pool: all current and future [`checkout`](Self::checkout)
    /// callers resolve to [`PoolError::Closed`]. Used by graceful shutdown.
    /// Idempotent.
    pub fn close(&self) {
        self.sender.close();
        self.receiver.close();
    }

    /// Total number of items the pool was created with.
    pub fn total(&self) -> usize {
        self.total
    }

    /// Number of currently available (not checked-out) items. O(1).
    pub fn available(&self) -> usize {
        self.receiver.len()
    }
}

/// RAII guard that auto-checks-in an item when dropped.
///
/// Returned by [`Pool::checkout`]. Deref to access the inner item.
/// On drop (including panic unwind) the item is returned to the pool;
/// if the pool was closed in the meantime the item is silently dropped.
pub struct PoolGuard<'a, T> {
    pool: &'a Pool<T>,
    item: Option<T>,
}

impl<T> PoolGuard<'_, T> {
    /// Strip the lifetime so the guard can be moved into a `'static`
    /// context (e.g. `tokio::task::spawn_blocking`). Returns the owned
    /// item together with an [`OwnedReservation`] that must receive the
    /// item back via [`OwnedReservation::checkin`] when the blocking task
    /// is done. Forgets the original guard so the inner Drop does not also
    /// try to check-in.
    pub fn into_owned(mut self) -> (T, OwnedReservation<T>) {
        let item = self
            .item
            .take()
            .expect("PoolGuard::into_owned called after drop");
        let reservation = OwnedReservation {
            sender: self.pool.sender.clone(),
        };
        (item, reservation)
    }
}

impl<T> Deref for PoolGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.item
            .as_ref()
            .expect("PoolGuard accessed after item taken")
    }
}

impl<T> DerefMut for PoolGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.item
            .as_mut()
            .expect("PoolGuard accessed after item taken")
    }
}

impl<T> Drop for PoolGuard<'_, T> {
    fn drop(&mut self) {
        if let Some(item) = self.item.take() {
            // Best-effort checkin. `try_send` is non-blocking and the
            // channel capacity equals total items, so it can only fail
            // if the pool was closed — in which case dropping the item
            // is the right thing.
            let _ = self.pool.sender.try_send(item);
        }
    }
}

/// Owned counterpart to [`PoolGuard`] for `'static` contexts (e.g.
/// `spawn_blocking`). The item is returned to the pool automatically on Drop
/// via [`Self::checkin`], or if the guard is forgotten, via the Drop impl.
///
/// After a panic the guard is dropped during unwind, so the item is recovered
/// without requiring the caller to manually invoke `checkin`.
pub struct OwnedReservation<T> {
    sender: async_channel::Sender<T>,
}

impl<T> OwnedReservation<T> {
    /// Return the item to the pool from a synchronous (blocking) context.
    /// Silently drops the item if the pool has been closed.
    pub fn checkin(self, item: T) {
        let _ = self.sender.try_send(item);
    }

    /// Create an RAII guard that holds both the item and the reservation.
    /// On drop (including during panic unwind) the item is returned to the pool.
    pub fn guard(self, item: T) -> PoolItemGuard<T> {
        PoolItemGuard {
            reservation: self,
            item: Some(item),
        }
    }
}

/// RAII guard that couples an owned pool item with its reservation.
///
/// On drop the item is automatically checked back into the pool. This is the
/// recommended pattern for `spawn_blocking` tasks where a panic would otherwise
/// leak the pool slot.
pub struct PoolItemGuard<T> {
    reservation: OwnedReservation<T>,
    item: Option<T>,
}

impl<T> PoolItemGuard<T> {
    /// Mutable access to the inner item.
    pub fn item_mut(&mut self) -> &mut T {
        self.item
            .as_mut()
            .expect("PoolItemGuard item already taken")
    }

    /// Immutable access to the inner item.
    pub fn item(&self) -> &T {
        self.item
            .as_ref()
            .expect("PoolItemGuard item already taken")
    }

    /// Consume the guard and return the item, **without** checking it back in.
    /// The caller is responsible for returning the item via `checkin`.
    pub fn into_inner(mut self) -> T {
        self.item.take().expect("PoolItemGuard item already taken")
    }
}

impl<T> Deref for PoolItemGuard<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.item()
    }
}

impl<T> DerefMut for PoolItemGuard<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.item_mut()
    }
}

impl<T> Drop for PoolItemGuard<T> {
    fn drop(&mut self) {
        if let Some(item) = self.item.take() {
            let _ = self.reservation.sender.try_send(item);
        }
    }
}
