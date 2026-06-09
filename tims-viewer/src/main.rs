//! tims-viewer — native GPU viewer for Bruker timsTOF raw data.
//!
//! Usage:
//!   tims-viewer <path/to/run.d>      open a real Bruker .d dataset
//!   tims-viewer DEMO                 synthetic point cloud (no Bruker data needed)
//!   tims-viewer DEMO --budget 25000000

mod app;
mod camera;
mod data;
mod render;
mod state;
mod ui;

use anyhow::Result;
use clap::Parser;
use winit::event_loop::{ControlFlow, EventLoop};

use app::{App, Plan};
use data::meta::MetaIndex;

/// Command-line arguments.
#[derive(Parser, Debug)]
#[command(name = "tims-viewer", about = "Native GPU viewer for Bruker timsTOF raw data")]
struct Args {
    /// Path to a Bruker `.d` folder, or the literal `DEMO` for a synthetic cloud.
    input: String,
    /// On-GPU point budget (resident capacity). Larger = more detail, more VRAM.
    #[arg(long)]
    budget: Option<usize>,
    /// Number of synthetic frames for DEMO.
    #[arg(long, default_value_t = 800)]
    demo_frames: usize,
    /// Total synthetic points for DEMO (before downsampling to the budget).
    #[arg(long, default_value_t = 30_000_000)]
    demo_points: u64,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    let (meta, is_demo) = if args.input.eq_ignore_ascii_case("DEMO") {
        log::info!(
            "DEMO mode: {} frames, {} synthetic points",
            args.demo_frames,
            args.demo_points
        );
        (MetaIndex::demo(args.demo_frames, args.demo_points), true)
    } else {
        log::info!("loading metadata from {}", args.input);
        let meta = MetaIndex::load(&args.input)?;
        log::info!(
            "{} frames, ~{} points, RT [{:.1}, {:.1}] s, m/z [{:.1}, {:.1}], 1/K0 [{:.3}, {:.3}]",
            meta.frames.len(),
            meta.total_points_estimate,
            meta.bounds.rt.min,
            meta.bounds.rt.max,
            meta.bounds.mz.min,
            meta.bounds.mz.max,
            meta.bounds.im.min,
            meta.bounds.im.max,
        );
        (meta, false)
    };

    let plan = Plan::new(meta, is_demo, args.budget);
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new(plan);
    event_loop.run_app(&mut app)?;
    Ok(())
}
