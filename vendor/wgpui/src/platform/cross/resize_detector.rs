#[cfg(not(target_family = "wasm"))]
use device_query::{DeviceQuery, DeviceState};
use std::cell::Cell;
use std::time::Duration;
use crate::time_ext::Instant;

/// Detects whether a window is actively being resized by watching the stream
/// of [`winit::event::WindowEvent::Resized`] events.
///
/// **Primary strategy:** poll global mouse button state via `device_query`.
/// While the left button is held after a resize event started, the user is
/// still dragging. The moment it releases the resize is done instantly.
/// On macOS, this is available only with Accessibility permission.
///
/// **Fallback:** idle-threshold timer for keyboard/programmatic resizes where
/// no mouse button is held (e.g. resize via keyboard shortcut, snap, tiling WM).
pub struct ResizeDetector {
    /// Set when a resize event has been seen and we haven't confirmed the end yet.
    active: Cell<bool>,
    /// Deadline for the timer-based fallback (programmatic / non-mouse resizes).
    deadline: Cell<Option<Instant>>,
    /// Cached device_query state — cheap to call, avoids re-initializing each poll.
    #[cfg(not(target_family = "wasm"))]
    device_state: Option<DeviceState>,
}

/// Fallback idle period used when the resize wasn't driven by a held mouse button
/// (e.g. keyboard shortcut, tiling WM, programmatic resize).
const IDLE_THRESHOLD: Duration = Duration::from_millis(150);

impl ResizeDetector {
    pub fn new() -> Self {
        Self {
            active: Cell::new(false),
            deadline: Cell::new(None),
            #[cfg(target_os = "macos")]
            device_state: DeviceState::checked_new(),
            #[cfg(all(not(target_family = "wasm"), not(target_os = "macos")))]
            device_state: Some(DeviceState::new()),
        }
    }

    /// Call on every `WindowEvent::Resized` from the winit event loop.
    pub fn on_resize_event(&self) {
        self.active.set(true);
        self.deadline.set(Some(Instant::now() + IDLE_THRESHOLD));
    }

    /// Returns `true` while the resize is in progress.
    pub fn is_resizing(&self) -> bool {
        if !self.active.get() {
            return false;
        }

        #[cfg(not(target_family = "wasm"))]
        {
        // Poll the real global left-button state. If it's held the user is
        // still dragging the resize handle; keep deferring and push the
        // fallback deadline out so it doesn't fire during the drag.
        if let Some(device_state) = &self.device_state {
            let mouse = device_state.get_mouse();
            // button_pressed is 1-based; index 1 = left button.
            let left_held = mouse.button_pressed.get(1).copied().unwrap_or(false);

            if left_held {
                self.deadline.set(Some(Instant::now() + IDLE_THRESHOLD));
                return true;
            }
        }
        }

        // Left button is up. The user finished a mouse-driven resize; done immediately.
        // Also covers the timer path: if the button was never down (keyboard/programmatic)
        // we fall through to the deadline check below.
        //
        // Check deadline in case this is a non-mouse resize that still needs the grace period.
        match self.deadline.get() {
            Some(deadline) if Instant::now() < deadline => {
                // Could be a split-second between event and button release detection;
                // keep active until the deadline just to be safe.
                true
            }
            _ => {
                self.active.set(false);
                self.deadline.set(None);
                false
            }
        }
    }
}

impl Default for ResizeDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_starts_inactive_and_activates_on_resize() {
        let detector = ResizeDetector::new();

        assert!(!detector.is_resizing());
        detector.on_resize_event();
        assert!(detector.is_resizing());
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn timer_fallback_works_without_device_state() {
        let detector = ResizeDetector {
            active: Cell::new(false),
            deadline: Cell::new(None),
            device_state: None,
        };

        assert!(!detector.is_resizing());
        detector.on_resize_event();
        assert!(detector.is_resizing());
        detector
            .deadline
            .set(Some(Instant::now() - Duration::from_millis(1)));
        assert!(!detector.is_resizing());
        assert!(!detector.is_resizing());
    }
}
