import * as stylex from '@stylexjs/stylex'
import { announceOnly } from '#/announce'
import { createFileRoute } from '@tanstack/react-router'
import { pageMeta } from '#/seo'
import type { ReactNode } from 'react'
import { Button, Disclosure, DisclosurePanel, Heading } from 'react-aria-components'
import { Caret } from '#/components/Button'
import { CodeTabs, InlineCode } from '#/components/CodeBlock'
import { InstallNotes } from '#/components/Install'
import { ExternalLink } from '#/components/Link'
import { PageHead } from '#/components/Page'
import { Note, P } from '#/components/Prose'
import { SearchDemo } from '#/components/SearchDemo'
import { focus, frame, link, text } from '#/components/styles'
import { firstQuery } from '#/data/samples'
import { GITHUB_URL, NAME, RELEASED } from '#/data/site'
import { colors, fonts, media, radii, sizes, space, type } from '../tokens.stylex'

export const Route = createFileRoute('/docs')({
  beforeLoad: announceOnly,
  head: () => ({ meta: pageMeta(`Docs · ${NAME}`, `Core concepts, Python API reference, and examples for text, vector, sparse, and hybrid queries in ${NAME}.`, '/docs') }),
  component: Docs,
})

const parts = [
  { label: 'Full text', text: 'Block-max MaxScore BM25 over bit-packed postings. Snowball stemming in 18 languages.' },
  { label: 'Vectors', text: '1-bit RaBitQ codes over shared clusters, reranked from f16 rows only when the error bound requires it. Cosine, Euclidean, and dot product.' },
  { label: 'Learned sparse', text: 'SPLADE-style term weights scored by dot product on the same posting lists as BM25.' },
  { label: 'Filters', text: 'Ranges, sets, prefixes, globs, tags, and boolean logic evaluated inside each retriever before a doc enters top k.' },
  { label: 'Writes', text: 'Atomic batches via a CRC32 write-ahead log. Searchable on return, crash-safe on SIGKILL, explicit sync() and flush() for disk durability. Readers never lock writers.' },
  { label: 'Storage', text: 'Immutable memory-mapped segments merged in the background. Versioned formats (each release reads the prior format) tested with 344 fault-injection cases and 40 SIGKILLs.' },
]

const writes: [string, string][] = [
  ['Batch', 'Atomic upserts & deletes, one seq'],
  ['Log', 'CRC32 record, all or nothing'],
  ['Memtable', 'Live immediately, vectors in f16'],
  ['Segment', 'Postings, codes, columns flushed in background'],
]

const reads: [string, string][] = [
  ['Snapshot', 'Lock-free view of segments + memtable'],
  ['Retrievers', 'Block-max BM25, 1-bit RaBitQ + f16 rerank'],
  ['Fusion', '0–1 min-max blend (default 50/50) or RRF'],
  ['Hits', 'Ranked docs with text and snippets'],
]

const rag = String.raw`chunks = db.namespace("chunks", text="text", ints=["page"], strings=["source"], vector=768)
chunks.upsert_many(
    {"key": hinny.key(f"{path}#{i}"), "text": chunk, "attrs": {"source": path, "page": page}, "vector": vec}
    for i, (chunk, page, vec) in enumerate(split_and_embed(path))
)

hits = chunks.query(text=question, vector=embed(question), filter=attr("source").prefix("handbook/"), limit=8)
context = "\n\n".join(h.text["text"] for h in hits)`

const memory = String.raw`memory = db.namespace("memory", text="note", strings=["user", "kind"], ints=["at"], vector=768)

def remember(user, note, kind="fact"):
    key = hinny.key(f"{user}:{note}")
    memory.upsert(key, text=note, attrs={"user": user, "kind": kind, "at": int(time.time())}, vector=embed(note))

def recall(user, question, since=0):
    scope = (attr("user") == user) & (attr("at") >= since)
    return memory.query(text=question, vector=embed(question), filter=scope, limit=8)`

const sparse = String.raw`ns = db.namespace("docs", text="body", sparse=["terms"], vector=256, language="english")
ns.upsert(1, text="...", sparse={"terms": {"lease": 1.4, "rent": 0.6}}, vector=v)   # weights from a sparse model

hits = ns.query(sparse={"terms": query_weights}, vector=query_vector)    # sparse + vectors`

const batches = String.raw`snap = ns.snapshot()                          # the namespace as of now
ns.apply(upsert=[{"key": 7, "text": "moved"}], delete=[3])   # all of it or none
snap.get(3)                                   # still there: the snapshot is from before`

const contents: [string, string][] = [
  ['install', 'Install'],
  ['first', 'A first query'],
  ['why', 'Words and meaning'],
  ['parts', "What's in it"],
  ['api', 'The Python API'],
  ['rag', 'For RAG'],
  ['memory', 'For agent memory'],
  ['sparse', 'Learned sparse fields'],
  ['batches', 'Batches and snapshots'],
  ['how', 'How it works'],
]

const styles = stylex.create({
  // The contents at the left, staying in view; the text in a column beside it.
  layout: {
    display: 'grid',
    gridTemplateColumns: { default: '220px minmax(0, 1fr)', [media.tablet]: 'minmax(0, 1fr)' },
    columnGap: space.x16,
    borderTopWidth: sizes.hair,
    borderTopStyle: 'solid',
    borderTopColor: colors.line,
  },
  side: { display: { default: 'block', [media.tablet]: 'none' }, paddingTop: space.block },
  toc: { position: 'sticky', top: '96px', display: 'grid', gap: space.x2 },
  tocLabel: { marginBottom: space.x2, fontSize: type.meta, lineHeight: '20px', fontWeight: 500, color: colors.fg },
  tocLink: { fontSize: type.small, lineHeight: '22px' },
  // On narrow windows the contents fold away above the text.
  fold: { display: { default: 'none', [media.tablet]: 'block' }, borderBottomWidth: sizes.hair, borderBottomStyle: 'solid', borderBottomColor: colors.line },
  foldHeading: { margin: 0 },
  foldButton: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    width: '100%',
    paddingBlock: space.x4,
    paddingInline: 0,
    borderWidth: 0,
    backgroundColor: 'transparent',
    fontSize: type.small,
    lineHeight: '20px',
    fontWeight: 500,
    color: colors.fg,
    cursor: 'pointer',
  },
  caret: { transitionProperty: 'transform', transitionDuration: '150ms' },
  caretOpen: { transform: 'rotate(180deg)' },
  foldList: { display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))', columnGap: space.x6, rowGap: space.x2, paddingBottom: space.x5 },

  main: { minWidth: 0, maxWidth: '760px' },
  section: {
    display: 'grid',
    gridTemplateColumns: 'minmax(0, 1fr)',
    gap: space.x5,
    paddingBlock: space.block,
    borderTopWidth: sizes.hair,
    borderTopStyle: 'solid',
    borderTopColor: colors.line,
    scrollMarginTop: '72px',
  },
  first: { borderTopWidth: 0 },

  // What's in it: a name in the margin of each line.
  parts: { display: 'grid' },
  part: {
    display: 'grid',
    gridTemplateColumns: { default: '9rem minmax(0, 1fr)', [media.phone]: 'minmax(0, 1fr)' },
    columnGap: space.x5,
    rowGap: space.x1,
    paddingBlock: space.x4,
    borderTopWidth: sizes.hair,
    borderTopStyle: 'solid',
    borderTopColor: colors.line,
  },
  partFirst: { borderTopWidth: 0, paddingTop: 0 },

  // The API: a signature, then what it does.
  api: { display: 'grid' },
  sig: { paddingTop: space.x5, fontFamily: fonts.mono, fontSize: type.code, lineHeight: 1.7, fontWeight: 500, color: colors.fg, overflowWrap: 'anywhere', fontVariantLigatures: 'none' },
  sigFirst: { paddingTop: 0 },
  does: { paddingTop: space.x2, paddingBottom: space.x5, borderBottomWidth: sizes.hair, borderBottomStyle: 'solid', borderBottomColor: colors.line },

  // How it works: a write and a query, each a row of steps.
  flow: { display: 'grid', gap: space.x6 },
  lane: { display: 'grid', gap: space.x3 },
  steps: { display: 'grid', gridTemplateColumns: { default: 'repeat(4, minmax(0, 1fr))', [media.phone]: 'minmax(0, 1fr)' }, gap: space.x2 },
  step: { display: 'grid', gap: space.x1, alignContent: 'start', padding: space.x4, borderRadius: radii.card, backgroundColor: colors.surface },
  stepNumber: { fontFamily: fonts.mono, fontSize: type.micro, lineHeight: '16px', color: colors.muted },
})

function DocSection({ id, title, first = false, children }: { id: string; title: ReactNode; first?: boolean; children: ReactNode }) {
  return (
    <section id={id} aria-labelledby={`${id}-h`} {...stylex.props(styles.section, first && styles.first)}>
      <h2 id={`${id}-h`} {...stylex.props(text.h3)}>
        {title}
      </h2>
      {children}
    </section>
  )
}

function Steps({ steps }: { steps: [string, string][] }) {
  return (
    <ol {...stylex.props(styles.steps)}>
      {steps.map(([name, what], i) => (
        <li key={name} {...stylex.props(styles.step)}>
          <span {...stylex.props(styles.stepNumber)}>{i + 1}</span>
          <span {...stylex.props(text.title)}>{name}</span>
          <span {...stylex.props(text.meta)}>{what}</span>
        </li>
      ))}
    </ol>
  )
}

function Docs() {
  const toc = contents.map(([id, name]) => (
    <a key={id} href={`#${id}`} {...stylex.props(link.quiet, styles.tocLink, focus.ring)}>
      {name}
    </a>
  ))
  return (
    <>
      <PageHead label="Docs" title={`Using ${NAME}`}>
        A database is a directory of namespaces. Each namespace has a schema of text fields, typed attributes, optional sparse fields, and a vector. Writes are searchable on return. Source and language docs live on{' '}
        {GITHUB_URL ? (
          <>
            <ExternalLink href={GITHUB_URL} {...stylex.props(link.inline, focus.ring)}>
              GitHub
            </ExternalLink>
            : Rust (<ExternalLink href={`${GITHUB_URL}/tree/main/crates/hinny`} {...stylex.props(link.inline, focus.ring)}><InlineCode>crates/hinny</InlineCode></ExternalLink>), Python (<ExternalLink href={`${GITHUB_URL}/tree/main/bindings/python`} {...stylex.props(link.inline, focus.ring)}><InlineCode>bindings/python</InlineCode></ExternalLink>), Node.js (<ExternalLink href={`${GITHUB_URL}/tree/main/bindings/node`} {...stylex.props(link.inline, focus.ring)}><InlineCode>bindings/node</InlineCode></ExternalLink>), C (<ExternalLink href={`${GITHUB_URL}/tree/main/crates/hinny-c`} {...stylex.props(link.inline, focus.ring)}><InlineCode>crates/hinny-c</InlineCode></ExternalLink>), HTTP server (<ExternalLink href={`${GITHUB_URL}/tree/main/crates/hinny-server`} {...stylex.props(link.inline, focus.ring)}><InlineCode>crates/hinny-server</InlineCode></ExternalLink>), and architecture (<ExternalLink href={`${GITHUB_URL}/blob/main/DESIGN.md`} {...stylex.props(link.inline, focus.ring)}><InlineCode>DESIGN.md</InlineCode></ExternalLink>).
          </>
        ) : (
          <>
            the repository: Rust (<InlineCode>crates/hinny</InlineCode>), Python (<InlineCode>bindings/python</InlineCode>), Node.js (<InlineCode>bindings/node</InlineCode>), C (<InlineCode>crates/hinny-c</InlineCode>), HTTP server (<InlineCode>crates/hinny-server</InlineCode>), and <InlineCode>DESIGN.md</InlineCode>.
          </>
        )}
      </PageHead>

      <div {...stylex.props(frame.page)}>
        <div {...stylex.props(styles.layout)}>
          <aside {...stylex.props(styles.side)}>
            <nav aria-label="On this page" {...stylex.props(styles.toc)}>
              <p {...stylex.props(styles.tocLabel)}>On this page</p>
              {toc}
            </nav>
          </aside>

          <div {...stylex.props(styles.main)}>
            <div {...stylex.props(styles.fold)}>
              <Disclosure>
                {({ isExpanded }) => (
                  <>
                    <Heading level={2} {...stylex.props(styles.foldHeading)}>
                      <Button slot="trigger" {...stylex.props(styles.foldButton, focus.ring)}>
                        On this page
                        <span {...stylex.props(styles.caret, isExpanded && styles.caretOpen)}>
                          <Caret size={12} />
                        </span>
                      </Button>
                    </Heading>
                    <DisclosurePanel>
                      <nav aria-label="On this page" {...stylex.props(styles.foldList)}>
                        {toc}
                      </nav>
                    </DisclosurePanel>
                  </>
                )}
              </Disclosure>
            </div>

            <DocSection id="install" title="Install" first>
              <InstallNotes />
            </DocSection>

            <DocSection id="first" title="A first query">
              <P>Open a namespace, upsert a document, and run a filtered hybrid query.</P>
              <CodeTabs tabs={firstQuery} note={RELEASED ? undefined : 'API preview'} />
            </DocSection>

            <DocSection id="why" title="Why hybrid search">
              <P>BM25 matches exact names, IDs, and error codes. Vectors catch paraphrases when the wording differs. One call runs both under the same filter and blends their scores. Sample data below:</P>
              <SearchDemo />
            </DocSection>

            <DocSection id="parts" title="One index for words, meaning, and attributes">
              <dl {...stylex.props(styles.parts)}>
                {parts.map((p, i) => (
                  <div key={p.label} {...stylex.props(styles.part, i === 0 && styles.partFirst)}>
                    <dt {...stylex.props(text.title)}>{p.label}</dt>
                    <dd {...stylex.props(text.body)}>{p.text}</dd>
                  </div>
                ))}
              </dl>
            </DocSection>

            <DocSection id="api" title="The Python API">
              <dl {...stylex.props(styles.api)}>
                <dt {...stylex.props(styles.sig, styles.sigFirst)}>Database(path), db.namespace(name, text=…, ints=…, floats=…, strings=…, tags=…, vector=dims, metric="cosine"), db.open(name)</dt>
                <dd {...stylex.props(text.body, styles.does)}>
                  Opens or creates a namespace. Attributes support integers, floats, strings, and tags (multi-string lists). Schemas can grow over time. <InlineCode>metric</InlineCode> selects <InlineCode>"cosine"</InlineCode> (default), <InlineCode>"euclidean"</InlineCode>, or <InlineCode>"dot"</InlineCode> and is fixed once set. <InlineCode>language="english"</InlineCode> (or any of 18 Snowball languages) enables stemming and drops stopwords by default for English and German (<InlineCode>stopwords=False</InlineCode> keeps them); analyzers are fixed once the namespace holds data. <InlineCode>db.open(name)</InlineCode> opens an existing namespace with its stored schema; <InlineCode>db.delete(name)</InlineCode> removes a closed namespace and all its files.
                </dd>
                <dt {...stylex.props(styles.sig)}>ns.upsert(key, text=…, attrs=…, vector=…), ns.upsert_many(docs)</dt>
                <dd {...stylex.props(text.body, styles.does)}>
                  Writes are immediately searchable and survive a process crash on return. The WAL is OS-buffered; call <InlineCode>ns.sync()</InlineCode> for an explicit storage durability boundary. NumPy <InlineCode>float32</InlineCode> arrays are read in place. <InlineCode>upsert_many</InlineCode> commits an iterable as one atomic batch.
                </dd>
                <dt {...stylex.props(styles.sig)}>ns.query(text=…, vector=…, sparse=…, filter=…, limit=10, order_by=…)</dt>
                <dd {...stylex.props(text.body, styles.does)}>Runs text, vector, sparse, or any combination together. With only a filter, lists matching documents in <InlineCode>order_by</InlineCode> order. Hits include stored text and a highlighted snippet by default.</dd>
                <dt {...stylex.props(styles.sig)}>ns.query(text="tax ret", typing=32)</dt>
                <dd {...stylex.props(text.body, styles.does)}>Search-as-you-type: treats the final word as an incomplete prefix and expands it to the top 32 matching terms, so "tax ret" matches "tax return".</dd>
                <dt {...stylex.props(styles.sig)}>text_weight=0.5, fusion="rrf"</dt>
                <dd {...stylex.props(text.body, styles.does)}>By default, each retriever's scores are min-max scaled to 0–1 and blended 50/50. Lower <InlineCode>text_weight</InlineCode> when your embedding model is strong on the domain; raise it when exact names and codes matter more. Pass <InlineCode>fusion="rrf"</InlineCode> for Reciprocal Rank Fusion.</dd>
                <dt {...stylex.props(styles.sig)}>text_candidates=200, vector_candidates=100, sparse_candidates=200, ns.explain(…)</dt>
                <dd {...stylex.props(text.body, styles.does)}>
                  Controls how many candidates each retriever gathers before fusion (fixed regardless of <InlineCode>limit</InlineCode>, raised automatically if <InlineCode>limit</InlineCode> is larger). <InlineCode>explain()</InlineCode> returns the hits alongside fusion diagnostics: each retriever's depth, match count, score range, and overlap.
                </dd>
                <dt {...stylex.props(styles.sig)}>attr("kind") == "pdf", &gt;=, .between(), .one_of(), .prefix(), .contains(), .glob(), .iglob(), .has(), .has_any(), .has_all(), &amp; | ~</dt>
                <dd {...stylex.props(text.body, styles.does)}>
                  Filters evaluated inside the search so filtered results stay exact. Numeric filters accept ints or floats. <InlineCode>.glob()</InlineCode> and <InlineCode>.iglob()</InlineCode> match full strings with <InlineCode>*</InlineCode> and <InlineCode>?</InlineCode>. On tags, use <InlineCode>.has("tax")</InlineCode>, <InlineCode>.has_any([…])</InlineCode>, and <InlineCode>.has_all([…])</InlineCode>; string predicates match when any tag on the document matches.
                </dd>
                <dt {...stylex.props(styles.sig)}>ns.count(filter), ns.count_by("kind", filter, limit)</dt>
                <dd {...stylex.props(text.body, styles.does)}>Counts documents matching a filter directly from attribute columns, or groups counts by attribute value. Documents with tags are counted once per tag.</dd>
                <dt {...stylex.props(styles.sig)}>ns.apply(upsert=…, delete=…), ns.delete_many(keys), ns.snapshot()</dt>
                <dd {...stylex.props(text.body, styles.does)}>Atomic mixed write/delete batches, and point-in-time snapshots that remain isolated from later writes and merges.</dd>
                <dt {...stylex.props(styles.sig)}>ns.delete_where(filter)</dt>
                <dd {...stylex.props(text.body, styles.does)}>Deletes every document matching a filter in one atomic batch and returns the count deleted.</dd>
                <dt {...stylex.props(styles.sig)}>ns.patch(key, attrs=…, remove=…), ns.patch_where(filter, attrs=…)</dt>
                <dd {...stylex.props(text.body, styles.does)}>
                  Updates specific fields of a document while keeping the rest, removing any names listed in <InlineCode>remove</InlineCode>. <InlineCode>patch_where</InlineCode> applies a patch across all documents matching a filter and returns the count updated. Because segments are immutable, patching reads and rewrites the document (requiring stored text).
                </dd>
                <dt {...stylex.props(styles.sig)}>ns.upsert_if(key, condition, …), ns.delete_if(key, condition), ns.patch(key, …, condition=…)</dt>
                <dd {...stylex.props(text.body, styles.does)}>
                  Conditional writes checked atomically under the writer lock for optimistic concurrency (for example, writing only when a revision attribute is below the new one). <InlineCode>upsert_if</InlineCode> also inserts when the key is missing; <InlineCode>patch</InlineCode> and <InlineCode>delete_if</InlineCode> do not. Each returns whether it wrote.
                </dd>
                <dt {...stylex.props(styles.sig)}>ns.fork("sandbox")</dt>
                <dd {...stylex.props(text.body, styles.does)}>
                  Creates an independent namespace initialized from the current state for sandboxes, experiments, or migrations. Files are hard-linked rather than copied, so forking takes milliseconds and uses zero extra disk until writes diverge (3.9 ms on 1M docs; 8.9 ms with 10,000 unflushed docs in memory).
                </dd>
                <dt {...stylex.props(styles.sig)}>timeout=, ns.set_limits(max_batch_bytes=…, flush_bytes=…)</dt>
                <dd {...stylex.props(text.body, styles.does)}>
                  Queries past <InlineCode>timeout</InlineCode> raise <InlineCode>TimeoutError</InlineCode> instead of returning partial hits. Batches over <InlineCode>max_batch_bytes</InlineCode> (1 GiB default) raise <InlineCode>ValueError</InlineCode> before writing. Memtables flush at <InlineCode>flush_bytes</InlineCode> (256 MiB) or 20,000 passages. <InlineCode>ns.stats()</InlineCode> exposes memory use, active background work, and pinned segments.
                </dd>
                <dt {...stylex.props(styles.sig)}>ns.sync()</dt>
                <dd {...stylex.props(text.body, styles.does)}>
                  Flushes prior WAL writes to storage without building segments. Writes after the last successful sync can be lost on OS crash or power loss. Check I/O errors explicitly; cleanup on drop is best-effort. Durability depends on the filesystem and device honoring sync calls (non-Unix directory entries are not synced).
                </dd>
                <dt {...stylex.props(styles.sig)}>ns.optimize(), ns.flush(), ns.stats(), ns.verify(), hinny.key("any string")</dt>
                <dd {...stylex.props(text.body, styles.does)}>Compact into one segment after a bulk load, sync and flush memory to disk, inspect stats, verify every file against its checksums, and hash string IDs into stable 64-bit integer keys.</dd>
              </dl>
              <Note>
                Need a shared service? <InlineCode>hinny-server</InlineCode> exposes the same namespaces, queries, and filters over HTTP as JSON, with optional read replicas. See{' '}
                {GITHUB_URL ? (
                  <ExternalLink href={`${GITHUB_URL}/blob/main/crates/hinny-server/README.md`} {...stylex.props(link.inline, focus.ring)}>
                    <InlineCode>crates/hinny-server/README.md</InlineCode>
                  </ExternalLink>
                ) : (
                  <InlineCode>crates/hinny-server/README.md</InlineCode>
                )}
                .
              </Note>
            </DocSection>

            <DocSection id="rag" title="Retrieval-augmented generation">
              <P>Store chunks as documents with source metadata as attributes. Query with the question as both text and vector.</P>
              <CodeTabs tabs={[{ name: 'Python', code: rag }]} />
            </DocSection>

            <DocSection id="memory" title="Agent memory">
              <P>
                Writes are live on return, so an agent can recall on its next turn what it just stored. Keep each memory as a document, scope it with attributes (user, kind, timestamp), and query with both text and vector under a filter. Reusing a key overwrites in place; <InlineCode>delete_where</InlineCode> drops matching memories in one atomic batch.
              </P>
              <CodeTabs tabs={[{ name: 'Python', code: memory }]} />
            </DocSection>

            <DocSection id="sparse" title="Learned sparse fields">
              <P>Store term weights from SPLADE or OpenSearch inference-free encoders, scored by dot product with the same block-max pruning as BM25. Weights are stored to hundredths.</P>
              <CodeTabs tabs={[{ name: 'Python', code: sparse }]} />
            </DocSection>

            <DocSection id="batches" title="Batches and snapshots">
              <P>Every write is an atomic batch with a monotonic sequence number. Queries read an immutable view and never block writers. <InlineCode>snapshot()</InlineCode> pins a view for as long as you hold it. One process writes at a time; others can open read-only snapshots.</P>
              <CodeTabs tabs={[{ name: 'Python', code: batches }]} />
            </DocSection>

            <DocSection id="how" title="How it works">
              <P>
                A checksummed WAL for fresh writes, immutable memory-mapped segments for everything else, background merges, and 1-bit RaBitQ vectors reranked at half precision only when needed. Full architecture details live in{' '}
                {GITHUB_URL ? (
                  <ExternalLink href={`${GITHUB_URL}/blob/main/DESIGN.md`} {...stylex.props(link.inline, focus.ring)}>
                    <InlineCode>DESIGN.md</InlineCode>
                  </ExternalLink>
                ) : (
                  <InlineCode>DESIGN.md</InlineCode>
                )}
                .
              </P>
              <div {...stylex.props(styles.flow)}>
                <div {...stylex.props(styles.lane)}>
                  <h3 {...stylex.props(text.eyebrow)}>A write</h3>
                  <Steps steps={writes} />
                </div>
                <div {...stylex.props(styles.lane)}>
                  <h3 {...stylex.props(text.eyebrow)}>A query</h3>
                  <Steps steps={reads} />
                </div>
              </div>
            </DocSection>

          </div>
        </div>
      </div>
    </>
  )
}
