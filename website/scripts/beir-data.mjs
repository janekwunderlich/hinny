// The BEIR rows the site shows, from benchmarks/beir/results.jsonl (one JSON object a line),
// into src/data/beir.json. The JSON is committed, so a build with only this folder (Vercel's root
// directory set to website/) still works: without the results file, it's left as it is.
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const legacyProtocol = "legacy-unversioned";
const currentProtocol = "beir-v2";

export function selectResults(input, requestedProtocol) {
  const successful = input.filter((r) => !r.error);
  const protocols = new Set(successful.map((r) => r.protocol ?? legacyProtocol));
  if (!requestedProtocol && protocols.size !== 1) {
    throw new Error("BEIR results contain multiple protocols or no results. Set BEIR_PROTOCOL explicitly; protocols cannot be mixed.");
  }
  const protocol = requestedProtocol ?? [...protocols][0];
  if (![legacyProtocol, currentProtocol].includes(protocol)) {
    throw new Error(`Unsupported BEIR protocol: ${protocol}`);
  }
  const selected = successful.filter((r) => (r.protocol ?? legacyProtocol) === protocol);
  if (!selected.length) throw new Error(`No results for BEIR protocol ${protocol}`);
  return selected.map((r) => {
    const { dataset, system, mode, ndcg10, recall100, p50_ms, p99_ms, build_s, size_mb, docs, queries } = r;
    let query_config = r.query_config;
    if (protocol === legacyProtocol) {
      query_config = {
        ndcg10: { metric_cutoff: 10, request_limit: 100, self_exclusion: "remove_query_id" },
        recall100: { metric_cutoff: 100, request_limit: 100, self_exclusion: "remove_query_id" },
        latency: { request_limit: 10 },
      };
    } else if (query_config?.quality_and_latency?.request_limit !== 11 ||
               query_config?.quality_and_latency?.metric_cutoff !== 10 ||
               query_config?.quality_and_latency?.self_exclusion !== "remove_query_id_then_trim" ||
               query_config?.recall100?.request_limit !== 101 ||
               query_config?.recall100?.metric_cutoff !== 100 ||
               query_config?.recall100?.self_exclusion !== "remove_query_id_then_trim") {
      throw new Error(`Invalid ${currentProtocol} query configuration for ${system}/${mode}/${dataset}`);
    }
    return { dataset, system, mode, ndcg10, recall100, p50_ms, p99_ms, build_s, size_mb, docs, queries,
             protocol, query_config, ...(r.timing ? { timing: r.timing } : {}) };
  });
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const source = fileURLToPath(new URL("../../benchmarks/beir/results.jsonl", import.meta.url));
  const target = fileURLToPath(new URL("../src/data/beir.json", import.meta.url));
  if (!existsSync(source)) {
    console.log("beir-data: no benchmarks/beir/results.jsonl here, keeping src/data/beir.json");
  } else {
    const input = readFileSync(source, "utf8").split("\n").filter(Boolean).map((line) => JSON.parse(line));
    const rows = selectResults(input, process.env.BEIR_PROTOCOL);
    writeFileSync(target, JSON.stringify(rows, null, 1) + "\n");
    console.log(`beir-data: ${rows.length} rows, protocol ${rows[0].protocol}`);
  }
}
