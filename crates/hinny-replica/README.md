# hinny-replica

An embedded read-replica library that mirrors a [`hinny-server`](../hinny-server/README.md) primary's namespaces into a local directory over HTTP and queries them in-process—local disk read latency with writes directed to the single primary. `hinny-server --replica-of` wraps this crate.

```rust
use std::sync::Arc;
use std::time::Duration;

let replica = Arc::new(hinny_replica::Replica::new("http://10.0.0.5:7878", Some("secret"), "copy")?);
replica.sync_all()?; // initial sync of all namespaces on the primary
let _syncing = replica.sync_every(Duration::from_secs(1), |e| eprintln!("sync failed: {e}"));

let docs = replica.namespace("docs")?; // read-only hinny::Namespace
let hits = docs.query().text("apartment lease").limit(10).run()?;
```

## API

- `Replica::sync(name)`: brings one namespace up to date and returns whether anything changed; removes the local copy if deleted on the primary.
- `Replica::sync_all()`: syncs every namespace on the primary and prunes deleted ones; an error on one namespace does not abort the rest.
- `Replica::namespace(name)`: returns the local read-only handle. Subsequent syncs refresh the handle automatically (reopening if the schema changed).
- `Replica::sync_every(interval, report)`: spawns a background sync thread that runs until the returned guard is dropped.

Syncing downloads only missing immutable files, writes the schema, atomically updates the manifest last, and removes unreferenced files. Interrupted syncs leave the prior local state intact. See [`hinny-server`'s Replicas section](../hinny-server/README.md#replicas) for freshness bounds and primary flushing behavior.

Uses blocking HTTP via [`ureq`](https://crates.io/crates/ureq) with TLS powered by `rustls`.
