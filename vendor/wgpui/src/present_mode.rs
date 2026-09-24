//! Mode de presentation de la swapchain, pilotable a chaud. Distinct de
//! `flamegraph::PresentMode`, qui n'existe que sous sa feature et ne sert
//! qu'au format de capture.

use std::sync::atomic::{AtomicU8, Ordering};

/// Cadence de presentation de la fenetre.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum WindowPresentMode {
    /// Vsync : une trame par rafraichissement d'ecran, jamais de dechirure.
    #[default]
    Fifo,
    /// Vsync sans blocage : la derniere trame prete remplace la precedente.
    /// Pas de dechirure, latence plus basse, mais le GPU travaille en continu.
    Mailbox,
    /// Presentation immediate : pas de plafond, dechirure possible.
    Immediate,
}

impl WindowPresentMode {
    /// Cle stable pour les reglages et `/stats`.
    pub fn key(self) -> &'static str {
        match self {
            Self::Fifo => "fifo",
            Self::Mailbox => "mailbox",
            Self::Immediate => "immediate",
        }
    }

    /// Inverse de [`Self::key`]. `None` sur une cle inconnue.
    pub fn from_key(key: &str) -> Option<Self> {
        match key {
            "fifo" => Some(Self::Fifo),
            "mailbox" => Some(Self::Mailbox),
            "immediate" => Some(Self::Immediate),
            _ => None,
        }
    }

    fn bit(self) -> u8 {
        1 << (self as u8)
    }
}

static CURRENT: AtomicU8 = AtomicU8::new(0);
static SUPPORTED: AtomicU8 = AtomicU8::new(0);

fn decode(raw: u8) -> WindowPresentMode {
    match raw {
        1 => WindowPresentMode::Mailbox,
        2 => WindowPresentMode::Immediate,
        _ => WindowPresentMode::Fifo,
    }
}

/// Le mode configure sur la swapchain de la fenetre.
pub fn window_present_mode() -> WindowPresentMode {
    decode(CURRENT.load(Ordering::Relaxed))
}

/// Les modes que la surface accepte reellement. `Fifo` est garanti par
/// WebGPU ; `Mailbox` manque sur beaucoup de pilotes, `Immediate` sur
/// certains compositeurs.
pub fn supported_present_modes() -> Vec<WindowPresentMode> {
    let mask = SUPPORTED.load(Ordering::Relaxed);
    [
        WindowPresentMode::Fifo,
        WindowPresentMode::Mailbox,
        WindowPresentMode::Immediate,
    ]
    .into_iter()
    .filter(|m| mask == 0 && *m == WindowPresentMode::Fifo || mask & m.bit() != 0)
    .collect()
}

pub(crate) fn set_window_present_mode(mode: WindowPresentMode) {
    CURRENT.store(mode as u8, Ordering::Relaxed);
}

pub(crate) fn set_supported_present_modes(modes: impl IntoIterator<Item = WindowPresentMode>) {
    SUPPORTED.store(
        modes.into_iter().fold(0, |acc, m| acc | m.bit()),
        Ordering::Relaxed,
    );
}
