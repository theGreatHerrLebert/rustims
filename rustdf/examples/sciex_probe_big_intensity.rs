//! Probe: how does a REAL .wiff.scan encode intensities above 65,535?
//!
//! The sciexwiff codec guesses a 3-byte `0x7e` escape, which the vendor reader mis-decodes
//! (peaks read back as ~4.1e9). The ZenoTOF 8600 CC0 template has ~20k spectra with base peaks
//! above 65,535 (pwiz reads them fine), so their blocks carry the true encoding. This tool finds
//! blocks whose decoded base-peak m/z matches a target (from the mzML), and dumps the raw token
//! bytes around the largest peak, plus every token the codec cannot classify.
//!
//!   cargo run --release --example sciex_probe_big_intensity --features sciex -- \
//!       TEMPLATE.wiff.scan 414.7145 0.02 <block_lo> <block_hi> <pwiz_bp_intensity>
#[cfg(feature = "sciex")]
fn main() -> Result<(), String> {
    use rustdf::sim::sciex_dispatch::{read_hdr, seed_cut_n};
    use sciexwiff::wiffscan::{decode_tracked, n_to_mz, scan_blocks};
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 7 {
        return Err("args: SCAN target_mz min_peaks block_lo block_hi expected_bp_intensity".into());
    }
    let scan = std::fs::read(&a[1]).map_err(|e| e.to_string())?;
    let (target, min_peaks) = (a[2].parse::<f64>().unwrap(), a[3].parse::<usize>().unwrap());
    let (lo, hi) = (a[4].parse::<usize>().unwrap(), a[5].parse::<usize>().unwrap());
    let expected = a[6].parse::<u32>().unwrap();
    let blocks = scan_blocks(&scan);
    eprintln!("{} blocks; probing [{lo}, {hi})", blocks.len());
    let mut hits = 0;
    for (bi, b) in blocks.iter().enumerate().take(hi).skip(lo) {
        let start = b.ff + 9;
        if start >= b.end || b.end > scan.len() { continue; }
        let cut_n = match read_hdr(&scan, b.ff) { Ok(h) => seed_cut_n(h, b.cal_a, b.cal_b), Err(_) => continue };
        let body = &scan[start..b.end];
        let (peaks, spans, _) = decode_tracked(body, 0, cut_n, usize::MAX, false);
        if peaks.len() < min_peaks { continue; }
        // base peak by decoded intensity
        let (mi, &(n, it)) = peaks.iter().enumerate().max_by_key(|(_, p)| p.1).unwrap();
        let mz = n_to_mz(n, b.cal_a, b.cal_b);
        hits += 1;
        let tic: u64 = peaks.iter().map(|p| p.1 as u64).sum();
        println!("\n== block {bi}: peaks={} tic={tic} cut_n={cut_n} cal=({:.6e},{:.4}) decoded max: n={n} mz={mz:.4} it={it}", peaks.len(), b.cal_a, b.cal_b);
        // the peak at the target m/z (pwiz says it is the base peak with `expected` intensity)
        let (ti, _) = peaks.iter().enumerate().min_by(|(_, x), (_, y)| {
            (n_to_mz(x.0, b.cal_a, b.cal_b) - target).abs().partial_cmp(&(n_to_mz(y.0, b.cal_a, b.cal_b) - target).abs()).unwrap()
        }).unwrap();
        for (label, idx) in [("target-mz peak", ti), ("decoded-max peak", mi)] {
            let (s, e) = spans[idx];
            let ctx_s = s.saturating_sub(12); let ctx_e = (e + 12).min(body.len());
            let hex = |r: std::ops::Range<usize>| body[r].iter().map(|x| format!("{x:02x}")).collect::<Vec<_>>().join(" ");
            println!("  {label}: peak#{idx} n={} mz={:.4} decoded_it={} (pwiz bp={expected}) span=[{s},{e}) bytes: [{}] {} [{}]",
                peaks[idx].0, n_to_mz(peaks[idx].0, b.cal_a, b.cal_b), peaks[idx].1, hex(ctx_s..s), hex(s..e), hex(e..ctx_e));
        }
        // token census: prefixes 0x7c/0x7d/0x7e/0x7f seen in this block, with the raw bytes after 0x7e/0x7f
        let mut census = [0usize; 4];
        for &(s, e) in &spans {
            let t = &body[s..e];
            // intensity field starts after the delta prefix (0xfc: 2 bytes, 0xfd: 3 bytes, 0x80..0xff: 1 byte)
            let off = match t[0] { 0xfc => 2, 0xfd => 3, 0x80..=0xff => 1, _ => 0 };
            if off < t.len() {
                match t[off] { 0x7c => census[0] += 1, 0x7d => census[1] += 1, 0x7e => { census[2] += 1; println!("    0x7e token: {}", t.iter().map(|x| format!("{x:02x}")).collect::<Vec<_>>().join(" ")); }, 0x7f => { census[3] += 1; println!("    0x7f token: {}", t.iter().map(|x| format!("{x:02x}")).collect::<Vec<_>>().join(" ")); }, _ => {} }
            }
        }
        println!("  prefix census: 0x7c={} 0x7d={} 0x7e={} 0x7f={}", census[0], census[1], census[2], census[3]);
        if hits >= 14 { break; }
    }
    eprintln!("{hits} matching blocks");
    Ok(())
}
#[cfg(not(feature = "sciex"))]
fn main() { eprintln!("build with --features sciex"); }
