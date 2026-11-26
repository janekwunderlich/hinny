//! `hinny-server --dir <database directory> [--listen 127.0.0.1:7878] [--token <token>]
//! [--max-body-mb 256] [--replica-interval 1s]`: serves a hinny database over HTTP until SIGINT
//! or SIGTERM. With `--replica-of <primary url> [--primary-token <t>] [--sync-every 1s]`, serves
//! a copy of another server's, kept current, and refuses writes.

use std::net::SocketAddr;
use std::process::ExitCode;
use std::time::Duration;

use hinny_server::{DEFAULT_BODY_LIMIT, DEFAULT_REPLICA_INTERVAL, Options, Server};

const USAGE: &str = "usage: hinny-server --dir <database directory> [--listen 127.0.0.1:7878] [--token <token>] [--max-body-mb 256]
                    [--replica-interval 1s] [--replica-of <primary url> [--primary-token <token>] [--sync-every 1s]]

  --dir               the database: a directory of namespaces (created if there is none)
  --listen            the address to listen on (default 127.0.0.1:7878)
  --token             every request but /healthz must carry it as Authorization: Bearer <token>;
                      also read from HINNY_TOKEN. Without one, only a loopback address is allowed.
  --max-body-mb       the largest request body, in megabytes (default 256)
  --replica-interval  how long a namespace's listing for replicas is handed out again before it
                      is made anew, which writes what is in memory as a segment (default 1s)

A replica (--replica-of): --dir holds copies of the primary's namespaces, synced on a timer;
reads are answered from them, writes refused.
  --replica-of        the primary's URL, such as http://10.0.0.5:7878
  --primary-token     the primary's token, if it has one; also read from HINNY_PRIMARY_TOKEN
  --sync-every        how often to sync every namespace (default 1s)

Durations are a number with ms, s or m (500ms, 2s, 1m); a bare number is seconds.";

struct Args {
    dir: String,
    listen: SocketAddr,
    options: Options,
    replica_interval: Duration,
    replica_of: Option<String>,
    primary_token: Option<String>,
    sync_every: Duration,
}

/// `500ms`, `2s`, `1m`, or a bare number of seconds.
fn duration(flag: &str, value: &str) -> Result<Duration, String> {
    let bad = || format!("{flag} takes a duration such as 500ms, 2s or 1m, not {value:?}");
    let (number, unit) = match value.find(|c: char| !(c.is_ascii_digit() || c == '.')) {
        Some(i) => value.split_at(i),
        None => (value, "s"),
    };
    let n: f64 = number.parse().map_err(|_| bad())?;
    let seconds = match unit {
        "ms" => n / 1000.0,
        "s" => n,
        "m" => n * 60.0,
        _ => return Err(bad()),
    };
    Duration::try_from_secs_f64(seconds).map_err(|_| bad())
}

fn args() -> Result<Args, String> {
    let mut dir = None;
    let mut listen = "127.0.0.1:7878".to_string();
    // An empty HINNY_TOKEN is one that isn't set (as `HINNY_TOKEN= hinny-server ...` means).
    let mut options = Options { token: std::env::var("HINNY_TOKEN").ok().filter(|t| !t.is_empty()), body_limit: DEFAULT_BODY_LIMIT };
    let mut replica_interval = DEFAULT_REPLICA_INTERVAL;
    let mut replica_of = None;
    let mut primary_token = std::env::var("HINNY_PRIMARY_TOKEN").ok().filter(|t| !t.is_empty());
    let mut sync_every = Duration::from_secs(1);
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        if flag == "-h" || flag == "--help" {
            return Err(String::new());
        }
        let value = it.next().ok_or_else(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--dir" => dir = Some(value),
            "--listen" => listen = value,
            "--token" => options.token = Some(value),
            "--max-body-mb" => options.body_limit = value.parse::<usize>().ok().filter(|n| *n > 0).ok_or_else(|| format!("--max-body-mb takes a positive number of megabytes, not {value:?}"))? << 20,
            "--replica-interval" => replica_interval = duration(&flag, &value)?,
            "--replica-of" => replica_of = Some(value),
            "--primary-token" => primary_token = Some(value),
            "--sync-every" => sync_every = duration(&flag, &value)?.max(Duration::from_millis(10)),
            other => return Err(format!("no option {other:?}")),
        }
    }
    let dir = dir.ok_or("--dir is required: the database directory to serve")?;
    let listen: SocketAddr = listen.parse().map_err(|_| format!("--listen takes an address and port such as 127.0.0.1:7878, not {listen:?}"))?;
    if options.token.as_deref() == Some("") {
        return Err("the token is empty; give a real one, or none to listen on loopback only".to_string());
    }
    // Without a token anyone who can reach the port can read and delete everything, so only
    // this machine may.
    if options.token.is_none() && !listen.ip().is_loopback() {
        return Err(format!("refusing to listen on {listen} without a token: give --token (or HINNY_TOKEN), or listen on 127.0.0.1"));
    }
    Ok(Args { dir, listen, options, replica_interval, replica_of, primary_token, sync_every })
}

async fn shutdown() {
    let interrupt = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = interrupt => {}
        _ = terminate => {}
    }
    eprintln!("hinny-server: shutting down: finishing requests, closing namespaces");
}

/// Prints a failed sync, but not the same failure every second: once, until it changes.
fn report() -> impl FnMut(&std::io::Error) + Send + 'static {
    let mut last = String::new();
    move |e| {
        let message = e.to_string();
        if last != message {
            eprintln!("hinny-server: sync failed (retrying): {message}");
            last = message;
        }
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    let args = match args() {
        Ok(a) => a,
        Err(e) if e.is_empty() => {
            println!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Err(e) => {
            eprintln!("hinny-server: {e}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let opened = match &args.replica_of {
        Some(primary) => Server::replica_of(&args.dir, primary, args.primary_token.as_deref()),
        None => Server::open(&args.dir),
    };
    let server = match opened {
        Ok(s) => s,
        Err(e) => {
            let what = if args.replica_of.is_some() { "a replica" } else { "the database" };
            eprintln!("hinny-server: can't open {what} in {}: {e}", args.dir);
            return ExitCode::FAILURE;
        }
    };
    server.set_replica_interval(args.replica_interval);
    // A replica syncs once before it answers, so it doesn't answer from nothing (or from a
    // copy days old) when the primary is there; when it isn't, it answers from what it has.
    let _syncing = match server.replica() {
        Some(replica) => {
            let first = replica.clone();
            match tokio::task::spawn_blocking(move || first.sync_all()).await {
                Ok(Ok(_)) => {}
                Ok(Err(e)) => eprintln!("hinny-server: the first sync failed; serving what {} has: {e}", args.dir),
                Err(e) => eprintln!("hinny-server: the first sync failed: {e}"),
            }
            Some(replica.sync_every(args.sync_every, report()))
        }
        None => None,
    };
    let listener = match tokio::net::TcpListener::bind(args.listen).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("hinny-server: can't listen on {}: {e}", args.listen);
            return ExitCode::FAILURE;
        }
    };
    let what = match &args.replica_of {
        Some(primary) => format!("a replica of {primary} in {}", args.dir),
        None => args.dir.clone(),
    };
    eprintln!("hinny-server: serving {what} on http://{}{}", args.listen, if args.options.token.is_some() { " (with a token)" } else { "" });
    match hinny_server::serve(listener, server, &args.options, shutdown()).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("hinny-server: {e}");
            ExitCode::FAILURE
        }
    }
}
