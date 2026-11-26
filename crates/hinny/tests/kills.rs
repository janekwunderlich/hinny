//! Real crashes: a child process (this test binary, run again) writes batches with background
//! flushes and merges going on, and is killed with SIGKILL at a random moment. Its writes reached
//! the operating system, which a process crash doesn't lose, so whatever it acknowledged must
//! survive: the reopened index must hold exactly the state after a prefix of the batches, each
//! whole or not at all, that includes every batch the child reported as written.

#![cfg(unix)]

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use hinny::index::{Doc, Index, Op};

const CHILD: &str = "HINNY_KILL_CHILD";
const KILLS: u64 = 40;

fn doc(key: u64, version: u64) -> Doc {
    Doc { key, version, passages: vec![vec![format!("batch {version} key {key} with some words to index")]], attrs: Vec::new(), vector: None }
}

/// Batch `b`: key 0 at version `b`, eight new keys, and the first key of batch `b - 2` deleted.
fn batch(b: u64) -> (Vec<Doc>, Vec<u64>) {
    let mut docs = vec![doc(0, b)];
    docs.extend((1..=8).map(|i| doc(b * 10 + i, b)));
    let deletes = if b >= 3 { vec![(b - 2) * 10 + 1] } else { Vec::new() };
    (docs, deletes)
}

fn model(m: u64) -> HashMap<u64, u64> {
    let mut state = HashMap::new();
    for b in 1..=m {
        let (docs, deletes) = batch(b);
        for d in docs {
            state.insert(d.key, d.version);
        }
        for k in deletes {
            state.remove(&k);
        }
    }
    state
}

/// The child: writes batches until it's killed, printing each one as it's acknowledged. Small
/// memtables keep flushes and merges running in the background; a sync every 50 batches and a
/// checkpoint every 120 are rare enough that a kill usually lands among plain writes, where a
/// write that hadn't reached the OS would be lost. (In a normal test run, without the variable, it does nothing.)
#[test]
fn kill_child() {
    let Ok(dir) = std::env::var(CHILD) else { return };
    let mut ix = Index::open(std::path::Path::new(&dir), 1).unwrap();
    ix.flush_passages = 150;
    ix.merge_factor = 2;
    let mut out = std::io::stdout().lock();
    for b in 1.. {
        let (docs, deletes) = batch(b);
        let mut ops: Vec<Op> = docs.iter().map(Op::Put).collect();
        ops.extend(deletes.iter().map(|&k| Op::Delete(k)));
        ix.write(&ops).unwrap();
        use std::io::Write;
        writeln!(out, "ack {b}").unwrap();
        out.flush().unwrap();
        if b % 120 == 0 {
            ix.checkpoint().unwrap();
        } else if b % 50 == 0 {
            ix.sync().unwrap();
        }
    }
}

#[test]
fn killed_writers_keep_every_acknowledged_batch() {
    let exe = std::env::current_exe().unwrap();
    let mut seed = 0x5eed_u64;
    let mut survived = Vec::new();
    for kill in 0..KILLS {
        let dir = std::env::temp_dir().join(format!("hinny-kill-{kill}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut child = Command::new(&exe)
            .args(["--exact", "kill_child", "--nocapture", "--test-threads=1"])
            .env(CHILD, &dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let acked = Arc::new(Mutex::new(0u64));
        let reader = {
            let (stdout, acked) = (child.stdout.take().unwrap(), acked.clone());
            std::thread::spawn(move || {
                for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                    if let Some(b) = line.strip_prefix("ack ").and_then(|b| b.parse::<u64>().ok()) {
                        *acked.lock().unwrap() = b;
                    }
                }
            })
        };
        // Wait for the first batch, then kill somewhere in the next 1.5 s: early on, and amid the
        // flushes and merges that come later.
        while *acked.lock().unwrap() == 0 {
            if child.try_wait().unwrap().is_some() {
                panic!("kill {kill}: the child exited before writing");
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        std::thread::sleep(Duration::from_micros((seed >> 33) % 1_500_000));
        child.kill().unwrap();
        child.wait().unwrap();
        reader.join().unwrap();
        let reported = *acked.lock().unwrap();

        let ix = Index::open(&dir, 1).unwrap_or_else(|e| panic!("kill {kill} after batch {reported}: reopening failed: {e}"));
        let state = ix.versions();
        drop(ix);
        // Key 0 carries the number of the last batch applied: that names the only prefix to check.
        let m = state.get(&0).copied().unwrap_or(0);
        assert!(model(m) == state, "kill {kill} after batch {reported}: the index isn't the state after batches 1..={m}");
        assert!(m <= reported + 1, "kill {kill}: {m} batches survived, more than were written");
        assert!(m >= reported, "kill {kill}: batch {reported} was acknowledged but only {m} survived");
        survived.push(m);
        std::fs::remove_dir_all(&dir).unwrap();
    }
    survived.sort();
    println!("{KILLS} kills; batches surviving from {} to {}", survived[0], survived[survived.len() - 1]);
}
